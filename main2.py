"""
main.py

CodeT5+ 기반 코드 관련 4개 Task 파이프라인 (기말 프로젝트용)

[지원 Task]
  1) code_summary   : 코드 -> 자연어 설명
  2) bug_detection  : 코드 -> True / False
  3) code_repair    : 버그 있는 코드 -> 수정된 코드
  4) code_generation: 자연어(+시그니처) -> 코드

[기능]
  - 각 Task마다 "별도의" 모델을 학습 (멀티태스크 X)
  - Baseline: 사전학습된 모델 그대로, 내가 만든 데이터로 바로 평가
  - Fine-tuning: 각 Task별로 파인튜닝 후 같은 테스트셋으로 평가
  - 결과:
      checkpoints/<task>/            : Task별 파인튜닝 모델
      outputs/<task>_baseline/*      : Task별 baseline 성능
      outputs/<task>_finetune/*      : Task별 finetune 성능
"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from difflib import SequenceMatcher

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed as hf_set_seed,
)

# ============================================================
# CONFIGURATION
# ============================================================

# ----- 데이터 경로 -----
DATA_PATH = "data/origin.json"      # 학습용 JSON (리스트 형식)
TEST_DATA_PATH = "data/dataset.json"  # 테스트용 JSON (리스트 또는 test_samples 포맷)

# ----- 출력 경로 (루트) -----
CHECKPOINT_ROOT = "./checkpoints"     # 각 task별 모델이 들어갈 루트
RESULT_ROOT = "./outputs"             # 각 task별 평가 결과 루트

# ----- 모델/학습 하이퍼파라미터 -----
MODEL_NAME = "Salesforce/codet5p-220m"
RANDOM_SEED = 42

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5

WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 256

# ----- 실행 모드 -----
RUN_BASELINE = True      # True면 사전학습 모델 그대로 평가
RUN_FINETUNE = True      # True면 파인튜닝 + 평가
TEST_ONLY = False        # True면 파인튜닝 학습은 건너뛰고, 체크포인트에서만 로드해서 평가

# ----- 지원 Task 목록 -----
ALL_TASKS = ["code_summary", "bug_detection", "code_repair", "code_generation"]

# ----- Task별 prefix -----
TASK_PREFIXES = {
    "code_summary": "summarize code:",      # Code -> NL
    "bug_detection": "is it buggy:",        # Code -> True / False
    "code_repair": "fix a bug:",            # Buggy code -> Fixed code
    "code_generation": "generate code:",    # NL -> Code
}

# ----- 텍스트 생성 설정 -----
GENERATION_MAX_LENGTH = 512
GENERATION_MIN_LENGTH = 5
GENERATION_NUM_BEAMS = 4
GENERATION_LENGTH_PENALTY = 0.8
GENERATION_REPETITION_PENALTY = 1.1
GENERATION_NO_REPEAT_NGRAM_SIZE = 0


# ============================================================
# 유틸 함수들
# ============================================================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def load_json(path: str) -> Any:
    print(f"[INFO] Loading JSON from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_str(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_str(a), normalize_str(b)).ratio()


def parse_bool_label(text: str) -> str:
    """
    모델 출력에서 True/False 라벨을 추출.
    - 'true', 'True', 'TRUE' 등 -> 'True'
    - 'false', 'False', 'FALSE' 등 -> 'False'
    - 그 외 -> 원문 그대로
    """
    if text is None:
        return ""
    t = normalize_str(text)
    t_clean = t.replace(".", "").strip()
    if t_clean == "true":
        return "True"
    if t_clean == "false":
        return "False"
    if "true" in t and "false" not in t:
        return "True"
    if "false" in t and "true" not in t:
        return "False"
    return text.strip()


def normalize_code_for_em(s: str) -> str:
    """코드용 relaxed EM: 공백/개행/탭 제거 후 비교."""
    if s is None:
        return ""
    return "".join(str(s).lower().split())


# ============================================================
# 데이터 전처리: Task별 학습 예제 생성
# ============================================================

def build_task_examples_from_train(
    train_items: List[Dict[str, Any]],
    task: str,
) -> List[Dict[str, str]]:
    """
    학습용 JSON의 각 item에서, 특정 task에 해당하는
    input_text / target_text 리스트를 만든다.
    기대 포맷 예시 (item):
      {
        "nl": "설명",
        "pl": "코드",
        "prefix": "def foo(x):",
        "is_buggy": false,
        "fixed_code": ""
      }
    """
    examples: List[Dict[str, str]] = []

    for item in train_items:
        nl = (item.get("nl") or "").strip()
        pl = (item.get("pl") or "").strip()
        is_buggy = bool(item.get("is_buggy", False))
        fixed_code = (item.get("fixed_code") or "").strip()
        prefix = (item.get("prefix") or "").strip()

        if task == "code_summary":
            if not pl or not nl:
                continue
            input_text = f"{TASK_PREFIXES['code_summary']} {pl}"
            target_text = nl
            examples.append({"task": task, "input_text": input_text, "target_text": target_text})

        elif task == "bug_detection":
            if not pl:
                continue
            bug_label = "True" if is_buggy else "False"
            bug_input = (
                f"{TASK_PREFIXES['bug_detection']} {pl}\n"
                "Answer with exactly one word: True or False."
            )
            examples.append({"task": task, "input_text": bug_input, "target_text": bug_label})

        elif task == "code_generation":
            if not nl or not pl:
                continue
            gen_input = f"{TASK_PREFIXES['code_generation']} {nl}"
            if prefix:
                gen_input += f"\nFunction signature: {prefix}"
            examples.append({"task": task, "input_text": gen_input, "target_text": pl})

        elif task == "code_repair":
            if not (is_buggy and pl and fixed_code):
                continue
            input_text = f"{TASK_PREFIXES['code_repair']} {pl}"
            target_text = fixed_code
            examples.append({"task": task, "input_text": input_text, "target_text": target_text})

    random.shuffle(examples)
    print(f"[INFO] Built {len(examples)} training examples for task={task} "
          f"from {len(train_items)} original items.")
    return examples


# ============================================================
# Dataset
# ============================================================

@dataclass
class TaskExample:
    task: str
    input_text: str
    target_text: str


class TaskDataset(Dataset):
    """
    단일 Task용 Dataset (내부적으로 task 이름은 유지하지만 모델은 하나의 task만 학습).
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        examples: List[Dict[str, str]],
        max_source_length: int = 512,
        max_target_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples: List[TaskExample] = [TaskExample(**ex) for ex in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]

        model_inputs = self.tokenizer(
            ex.input_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                ex.target_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        labels_ids = labels["input_ids"].squeeze(0)

        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
            "task": ex.task,
            "raw_input": ex.input_text,
            "raw_target": ex.target_text,
        }


# ============================================================
# 모델/토크나이저
# ============================================================

def setup_model_and_tokenizer(model_name: str) -> Tuple[T5ForConditionalGeneration, AutoTokenizer]:
    print(f"[INFO] Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)

    special_tokens = {
        "additional_special_tokens": [
            "<code>", "</code>",
            "<bug>", "</bug>",
            "<fix>", "</fix>",
            "<summary>", "</summary>",
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"[INFO] Added {num_added} special tokens & resized embeddings.")

    return model, tokenizer


# ============================================================
# Dataset 준비 (Train/Val split)
# ============================================================

def prepare_task_datasets(
    tokenizer: AutoTokenizer,
    train_items: List[Dict[str, Any]],
    task: str,
    val_ratio: float = 0.1,
) -> Tuple[TaskDataset, TaskDataset]:
    examples = build_task_examples_from_train(train_items, task)
    dataset = TaskDataset(
        tokenizer=tokenizer,
        examples=examples,
        max_source_length=MAX_SOURCE_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
    )

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    print(f"[INFO][{task}] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    return train_dataset, val_dataset


# ============================================================
# 테스트셋 빌드 (학습 JSON 형식 또는 test_samples 형식 둘 다 지원)
# ============================================================

def build_test_samples_from_raw(raw: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    1) {"metadata": ..., "test_samples": [ { "task": ..., "input": ..., "expected_output": ... }, ... ]}
    2) 학습 JSON과 동일한 리스트 형식
    """
    if isinstance(raw, dict) and "test_samples" in raw:
        meta = raw.get("metadata", {})
        return raw["test_samples"], meta

    if isinstance(raw, list):
        print("[WARN] TEST_DATA_PATH 가 학습 데이터 형식(list)으로 보입니다. "
              "train과 동일한 방식으로 task별 테스트 샘플을 생성합니다.")
        test_samples: List[Dict[str, Any]] = []
        for item in raw:
            nl = (item.get("nl") or "").strip()
            pl = (item.get("pl") or "").strip()
            is_buggy = bool(item.get("is_buggy", False))
            fixed_code = (item.get("fixed_code") or "").strip()
            prefix = (item.get("prefix") or "").strip()

            # summary
            if nl and pl:
                test_samples.append({
                    "task": "code_summary",
                    "input": f"{TASK_PREFIXES['code_summary']} {pl}",
                    "expected_output": nl,
                })

            # bug detection
            if pl:
                bug_label = "True" if is_buggy else "False"
                bug_input = (
                    f"{TASK_PREFIXES['bug_detection']} {pl}\n"
                    "Answer with exactly one word: True or False."
                )
                test_samples.append({
                    "task": "bug_detection",
                    "input": bug_input,
                    "expected_output": bug_label,
                })

            # code generation
            if nl and pl:
                gen_input = f"{TASK_PREFIXES['code_generation']} {nl}"
                if prefix:
                    gen_input += f"\nFunction signature: {prefix}"
                test_samples.append({
                    "task": "code_generation",
                    "input": gen_input,
                    "expected_output": pl,
                })

            # code repair
            if is_buggy and pl and fixed_code:
                test_samples.append({
                    "task": "code_repair",
                    "input": f"{TASK_PREFIXES['code_repair']} {pl}",
                    "expected_output": fixed_code,
                })

        meta = {"note": "test samples automatically built from training-style data"}
        return test_samples, meta

    raise ValueError("지원하지 않는 TEST JSON 형식입니다.")


# ============================================================
# 텍스트 생성
# ============================================================

def generate_output(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    input_text: str,
    device: torch.device,
) -> str:
    model.eval()
    inputs = tokenizer(
        input_text,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=GENERATION_MAX_LENGTH,
            min_length=GENERATION_MIN_LENGTH,
            num_beams=GENERATION_NUM_BEAMS,
            length_penalty=GENERATION_LENGTH_PENALTY,
            repetition_penalty=GENERATION_REPETITION_PENALTY,
            no_repeat_ngram_size=GENERATION_NO_REPEAT_NGRAM_SIZE,
        )

    output = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return output.strip()


# ============================================================
# 평가 (단일 Task, Baseline or Finetune)
# ============================================================

def evaluate_single_task(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
    test_path: str,
    task: str,
    result_dir: str,
    run_label: str,   # "baseline" or "finetune"
) -> None:
    """
    단일 task에 대해 테스트셋을 평가하고,
    detailed/summary를 result_dir 하위에 저장.
    """
    ensure_dir(result_dir)
    raw = load_json(test_path)
    test_samples, meta = build_test_samples_from_raw(raw)

    # 해당 task에 해당하는 샘플만 필터링
    samples = [s for s in test_samples if s.get("task") == task]
    print(f"[INFO][{task}][{run_label}] #test samples = {len(samples)}")

    records: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        inp = sample.get("input", "")
        expected = sample.get("expected_output", "")

        pred = generate_output(model, tokenizer, inp, device)

        if task == "bug_detection":
            gold_label = parse_bool_label(expected)
            pred_label = parse_bool_label(pred)
            is_correct = normalize_str(gold_label) == normalize_str(pred_label)
            em = is_correct
            sim = similarity(gold_label, pred_label)
        elif task in ["code_generation", "code_repair"]:
            em = normalize_code_for_em(expected) == normalize_code_for_em(pred)
            sim = similarity(expected, pred)
            is_correct = em
        else:
            em = normalize_str(expected) == normalize_str(pred)
            sim = similarity(expected, pred)
            is_correct = em

        record = {
            "index": idx,
            "task": task,
            "input": inp,
            "expected_output": expected,
            "model_output": pred,
            "is_exact_match": bool(em),
            "similarity": float(sim),
        }

        if task == "bug_detection":
            record["gold_label"] = gold_label
            record["pred_label"] = pred_label

        records.append(record)

    # 요약 통계
    n = len(records)
    if n == 0:
        print(f"[WARN][{task}][{run_label}] No test samples. Skip saving.")
        return

    exact_matches = sum(1 for r in records if r["is_exact_match"])
    avg_em = exact_matches / n
    avg_sim = sum(r["similarity"] for r in records) / n

    if task == "bug_detection":
        metric_name = "accuracy"
    else:
        metric_name = "exact_match_rate"

    summary = {
        "metadata": meta,
        "task": task,
        "run_label": run_label,
        "num_samples": n,
        metric_name: avg_em,
        "avg_similarity": avg_sim,
    }

    # 파일 경로
    detailed_path = os.path.join(result_dir, f"{task}_{run_label}_detailed.json")
    summary_path = os.path.join(result_dir, f"{task}_{run_label}_summary.txt")

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Single-Task Evaluation Summary ===\n\n")
        f.write(f"[Task] {task}\n")
        f.write(f"[Run ] {run_label}\n\n")
        f.write(f"  - #samples       : {summary['num_samples']}\n")
        if task == "bug_detection":
            f.write(f"  - accuracy       : {summary['accuracy']:.4f}\n")
        else:
            f.write(f"  - exact_match    : {summary['exact_match_rate']:.4f}\n")
        f.write(f"  - avg_similarity : {summary['avg_similarity']:.4f}\n")

    print(f"[INFO][{task}][{run_label}] Saved detailed to {detailed_path}")
    print(f"[INFO][{task}][{run_label}] Saved summary  to {summary_path}")


# ============================================================
# 단일 Task: Fine-tuning + 평가
# ============================================================

def run_finetune_for_task(
    task: str,
    train_items: List[Dict[str, Any]],
    device: torch.device,
) -> None:
    """
    단일 task에 대해:
      - (TEST_ONLY=False면) 파인튜닝 후 체크포인트 저장
      - 체크포인트에서 모델을 로드해서 테스트셋 평가
    """
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, task)
    result_dir = os.path.join(RESULT_ROOT, f"{task}_finetune")

    # 모델/토크나이저 로드
    if TEST_ONLY and os.path.isdir(checkpoint_dir):
        # 이미 학습된 모델을 사용
        print(f"[INFO][{task}][finetune] TEST_ONLY=True -> Load from {checkpoint_dir}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir, trust_remote_code=True)
    else:
        print(f"[INFO][{task}][finetune] Training from pretrained model.")
        model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)

        train_dataset, val_dataset = prepare_task_datasets(tokenizer, train_items, task, val_ratio=0.1)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
        )

        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        print(f"[INFO][{task}][finetune] Start training...")
        trainer.train()
        print(f"[INFO][{task}][finetune] Training finished.")

        print(f"[INFO][{task}][finetune] Saving model & tokenizer to {checkpoint_dir}")
        ensure_dir(checkpoint_dir)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    model.to(device)

    # 평가
    print(f"[INFO][{task}][finetune] Start evaluation...")
    evaluate_single_task(
        model=model,
        tokenizer=tokenizer,
        device=device,
        test_path=TEST_DATA_PATH,
        task=task,
        result_dir=result_dir,
        run_label="finetune",
    )


# ============================================================
# 단일 Task: Baseline 평가 (파인튜닝 없이)
# ============================================================

def run_baseline_for_task(
    task: str,
    device: torch.device,
) -> None:
    """
    단일 task에 대해 파인튜닝 없이 사전학습 모델로만 평가.
    """
    result_dir = os.path.join(RESULT_ROOT, f"{task}_baseline")

    print(f"[INFO][{task}][baseline] Evaluating pretrained model (no fine-tuning).")
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    model.to(device)

    evaluate_single_task(
        model=model,
        tokenizer=tokenizer,
        device=device,
        test_path=TEST_DATA_PATH,
        task=task,
        result_dir=result_dir,
        run_label="baseline",
    )


# ============================================================
# main
# ============================================================

def main():
    print("=" * 60)
    print(" Per-Task Fine-tuning & Baseline for CodeT5+ ")
    print("=" * 60)
    print(f"[INFO] RUN_BASELINE = {RUN_BASELINE}")
    print(f"[INFO] RUN_FINETUNE = {RUN_FINETUNE}")
    print(f"[INFO] TEST_ONLY    = {TEST_ONLY}")
    print(f"[INFO] TASKS        = {ALL_TASKS}")
    print("=" * 60)

    set_random_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 학습 데이터는 finetune에만 필요
    train_items: List[Dict[str, Any]] = []
    if RUN_FINETUNE and not TEST_ONLY:
        train_raw = load_json(DATA_PATH)
        if not isinstance(train_raw, list):
            raise ValueError("학습 JSON은 리스트(list) 형식이어야 합니다.")
        train_items = train_raw

    # Task별 실행
    for task in ALL_TASKS:
        print("\n" + "-" * 60)
        print(f"[TASK] {task}")
        print("-" * 60)

        if RUN_BASELINE:
            run_baseline_for_task(task, device)

        if RUN_FINETUNE:
            run_finetune_for_task(task, train_items, device)


if __name__ == "__main__":
    main()