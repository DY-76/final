"""
multitask_pipeline.py

멀티태스크 CodeT5+ 파인튜닝 파이프라인 (기말 프로젝트용)

- 입력: 수업에서 사용했던 형식의 JSON 데이터 (nl/pl, is_buggy, fixed_code, prefix 등)
- 4가지 Task를 한 모델로 학습:
    1) Code Summarization  : 코드 -> 자연어 설명
    2) Bug Detection       : 코드 -> True / False
    3) Code Repair         : 버그 있는 코드 -> 수정된 코드
    4) Code Generation     : 자연어 설명(+시그니처) -> 코드
- 출력:
    - 체크포인트 폴더에 파인튜닝된 모델 저장
    - TEST 데이터셋에 대해 Task별 평가 결과 (json + txt 요약) 저장

사용 순서(예시):
1) data/my_train.json       에 학습용 데이터를 직접 구축
2) data/my_multitask_test.json 을 build_multitask_testset.py 같은 스크립트로 생성
3) 아래 CONFIGURATION 부분에서 경로 및 하이퍼파라미터를 필요에 맞게 수정
4) python multitask_pipeline.py  실행
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
    EarlyStoppingCallback,
    set_seed as hf_set_seed,
)

# ============================================================
# CONFIGURATION (이 부분만 바꿔도 전체 파이프라인이 동작합니다)
# ============================================================

'''
Plan:
- 무조건 8시까지 파일 디렉토리 구조 완성
- 10시까지 파이프라인 정상동작 확인하기 (슈퍼컴 필요? - 연구실가서 해야지 뭐 어케)
'''
# ----- 경로 설정 -----
DATA_PATH = "data/origin.json"              # 직접 구축한 학습 JSON
TEST_DATA_PATH = "data/dataset.json"  # 멀티태스크 테스트 JSON
OUTPUT_DIR = "./checkpoints/multitask_model"  # 파인튜닝된 모델 저장 경로
TEST_OUTPUT_DIR = "./output_multitask"        # 평가 결과 저장 경로

# ----- 모델/학습 하이퍼파라미터 -----
## 나중에 여기만 수정해서 파인튜닝하기
MODEL_NAME = "Salesforce/codet5p-220m"  # CodeT5+ 모델 이름
RANDOM_SEED = 42

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5

WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1

MAX_SOURCE_LENGTH = 512   # 입력 토큰 최대 길이
MAX_TARGET_LENGTH = 256   # 출력 토큰 최대 길이

# ----- 실행 모드 -----
TEST_ONLY = False  # True 이면 학습을 건너뛰고 OUTPUT_DIR의 모델로 테스트만 수행

# ----- 평가할 Task (4가지만 사용) -----
EVAL_TASKS = ["code_summary", "bug_detection", "code_repair", "code_generation"]

# ----- Task별 prefix (prompt) -----
TASK_PREFIXES = {
    "code_summary": "summarize code:",      # Code -> NL
    "bug_detection": "is it buggy:",        # Code -> True / False
    "code_repair": "fix a bug:",            # Buggy code -> Fixed code
    "code_generation": "generate code:",    # NL -> Code
}

# ----- 텍스트 생성 설정 (테스트 시) -----
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
    """파이썬, 넘파이, 파이토치, 트랜스포머의 랜덤 시드를 모두 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def load_json(path: str) -> Any:
    """간단한 JSON 로더."""
    print(f"[INFO] Loading JSON from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def ensure_dir(path: str) -> None:
    """폴더가 없으면 생성."""
    os.makedirs(path, exist_ok=True)


def normalize_str(s: str) -> str:
    """간단한 문자열 정규화: 양끝 공백 제거 + 소문자."""
    if s is None:
        return ""
    return str(s).strip().lower()


def similarity(a: str, b: str) -> float:
    """두 문자열 사이의 유사도(0~1)를 difflib으로 측정."""
    return SequenceMatcher(None, normalize_str(a), normalize_str(b)).ratio()


def parse_bool_label(text: str) -> str:
    """
    모델 출력에서 True/False 라벨을 추출.
    - 'true', 'True', 'TRUE' 등 -> 'True'
    - 'false', 'False', 'FALSE' 등 -> 'False'
    - 그 외 -> 원문 그대로 (후에 EM 비교용)
    """
    t = normalize_str(text)
    if "true" in t and "false" not in t:
        return "True"
    if "false" in t and "true" not in t:
        return "False"
    # 둘 다 들어있거나, 아무것도 없으면 원문 유지
    return text.strip()


# ============================================================
# 데이터 전처리: 학습용 JSON -> 멀티태스크 예제들
# ============================================================

def build_multitask_examples_from_train(train_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    학습용 JSON(원소마다 nl, pl, is_buggy, fixed_code, prefix 등을 포함)에 대해
    4개 Task에 해당하는 'input_text' / 'target_text' 예제를 생성한다.

    기대하는 입력 포맷(예시):

    {
        "nl": "주어진 숫자에 1을 더하는 함수 설명",
        "pl": "def add_one(x): return x + 1",
        "prefix": "def add_one(x):",
        "tests": ["assert add_one(1) == 2", ...],
        "is_buggy": false,
        "fixed_code": ""
    }
    """
    examples: List[Dict[str, str]] = []

    for item in train_items:
        nl = item.get("nl", "").strip()
        pl = item.get("pl", "").strip()
        is_buggy = bool(item.get("is_buggy", False))
        fixed_code = item.get("fixed_code", "").strip()
        prefix = item.get("prefix", "").strip()

        if not nl or not pl:
            # nl 또는 pl이 비어있으면 Task를 만들 수 없으므로 스킵
            continue

        # 1) Code Summarization (code -> nl)
        examples.append(
            {
                "task": "code_summary",
                "input_text": f"{TASK_PREFIXES['code_summary']} {pl}",
                "target_text": nl,
            }
        )

        # 2) Bug Detection (code -> True / False)
        bug_label = "True" if is_buggy else "False"
        bug_input = (f"{TASK_PREFIXES['bug_detection']} {pl}\n""!Answer with exactly one word: 'True' or 'False'.")
        examples.append(
            {
                "task": "bug_detection",
                "input_text": bug_input,
                "target_text": bug_label,
            }
        )

        # 3) Code Generation (nl (+ prefix) -> code)
        gen_input = f"{TASK_PREFIXES['code_generation']} {nl}"
        if prefix:
            gen_input += f"\nFunction signature: {prefix}"
        examples.append(
            {
                "task": "code_generation",
                "input_text": gen_input,
                "target_text": pl,
            }
        )

        # 4) Code Repair (buggy code -> fixed code)
        if is_buggy and fixed_code:
            examples.append(
                {
                    "task": "code_repair",
                    "input_text": f"{TASK_PREFIXES['code_repair']} {pl}",
                    "target_text": fixed_code,
                }
            )

    random.shuffle(examples)
    print(f"[INFO] Built {len(examples)} multi-task training examples "
          f"from {len(train_items)} original items.")
    return examples


# ============================================================
# 토치 Dataset 정의
# ============================================================

@dataclass
class MultiTaskExample:
    task: str
    input_text: str
    target_text: str


class MultiTaskDataset(Dataset):
    """
    MultiTaskExample 리스트를 받아서 T5 모델에 들어갈 수 있도록
    토크나이즈하는 PyTorch Dataset.
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
        self.examples: List[MultiTaskExample] = [
            MultiTaskExample(**ex) for ex in examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]

        # 입력 토크나이즈
        model_inputs = self.tokenizer(
            ex.input_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 타깃 토크나이즈
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

        # loss 계산 시 padding 토큰은 -100으로 마스킹
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
            # 아래 두 개는 Trainer가 사용하진 않지만, 디버깅/분석용으로 넣어둠
            "task": ex.task,
            "raw_input": ex.input_text,
            "raw_target": ex.target_text,
        }


# ============================================================
# 모델/토크나이저 세팅
# ============================================================

def setup_model_and_tokenizer(model_name: str) -> Tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """
    CodeT5+ 모델과 토크나이저 로드 및 특수 토큰 추가.
    """
    print(f"[INFO] Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)

    # 코드용 특수 토큰 (필요시 자유롭게 수정 가능)
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
        print(f"[INFO] Added {num_added} special tokens to tokenizer & resized embeddings")

    return model, tokenizer


# ============================================================
# 데이터셋 준비 (train / val split)
# ============================================================

def prepare_datasets(
    tokenizer: AutoTokenizer,
    train_items: List[Dict[str, Any]],
    val_ratio: float = 0.1,
) -> Tuple[MultiTaskDataset, MultiTaskDataset]:
    """
    원본 학습 JSON -> 멀티태스크 예제 -> train / val Dataset 분할.
    """
    examples = build_multitask_examples_from_train(train_items)
    dataset = MultiTaskDataset(
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

    print(f"[INFO] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    return train_dataset, val_dataset


# ============================================================
# 텍스트 생성 & 평가 로직
# ============================================================

def generate_output(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    input_text: str,
    device: torch.device,
) -> str:
    """단일 입력에 대해 모델 출력을 생성."""
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


def build_test_samples_from_raw(
    raw: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    테스트 JSON이 두 가지 케이스 중 하나라고 가정:

    1) {"metadata": ..., "test_samples": [ { "task": ..., "input": ..., "expected_output": ... }, ... ]}
    2) 학습 JSON과 동일한 형식의 리스트 -> 이 경우, train과 동일한 방식으로 4개 Task 테스트 예제를 만든다.
    """
    if isinstance(raw, dict) and "test_samples" in raw:
        meta = raw.get("metadata", {})
        return raw["test_samples"], meta

    if isinstance(raw, list):
        # 학습 데이터 형식과 동일한 것으로 가정하고, 간단히 테스트 샘플 생성
        print("[WARN] TEST_DATA_PATH 가 학습 데이터 형식(list)으로 보입니다. "
              "학습 데이터에서 직접 테스트 예제를 생성합니다.")
        examples = build_multitask_examples_from_train(raw)
        # training용으로 만든 예제(examples)를 테스트에도 재사용 (실제 프로젝트에서는 별도 데이터 권장)
        test_samples = []
        for ex in examples:
            test_samples.append(
                {
                    "task": ex["task"],
                    "input": ex["input_text"],
                    "expected_output": ex["target_text"],
                }
            )
        meta = {"note": "test samples automatically built from training-style data"}
        return test_samples, meta

    raise ValueError("지원하지 않는 TEST JSON 형식입니다.")


## 평가 관련, 차후 개선 필
def evaluate_on_test_set(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
    test_path: str,
    eval_tasks: List[str],
    output_dir: str,
) -> None:
    """
    멀티태스크 테스트셋에 대해 모델을 평가하고,
    - detailed 결과를 JSON
    - summary를 TXT 로 저장한다.
    """
    ensure_dir(output_dir)
    raw = load_json(test_path)
    test_samples, meta = build_test_samples_from_raw(raw)

    # Task 필터링 (EVAL_TASKS 설정에 따라)
    if eval_tasks:
        test_samples = [s for s in test_samples if s.get("task") in eval_tasks]

    print(f"[INFO] Number of test samples: {len(test_samples)}")

    per_task_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for idx, sample in enumerate(test_samples):
        task = sample.get("task")
        if eval_tasks and task not in eval_tasks:
            continue

        inp = sample.get("input", "")
        expected = sample.get("expected_output", "")

        pred = generate_output(model, tokenizer, inp, device)

        # 평가 지표 계산
        if task == "bug_detection":
            # True/False 분류 정확도
            gold_label = parse_bool_label(expected)
            pred_label = parse_bool_label(pred)
            is_correct = normalize_str(gold_label) == normalize_str(pred_label)
            em = is_correct  # EM과 동일
            sim = similarity(gold_label, pred_label)
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

        per_task_records[task].append(record)

    # ----- 요약 통계 계산 -----
    summary = {
        "metadata": meta,
        "total_tasks": list(per_task_records.keys()),
        "task_results": {},
    }

    for task, records in per_task_records.items():
        n = len(records)
        if n == 0:
            continue
        exact_matches = sum(1 for r in records if r["is_exact_match"])
        avg_em = exact_matches / n
        avg_sim = sum(r["similarity"] for r in records) / n

        # Bug Detection은 Accuracy라고 이름을 따로 붙여주면 보기 편함
        if task == "bug_detection":
            metric_name = "accuracy"
        else:
            metric_name = "exact_match_rate"

        summary["task_results"][task] = {
            "num_samples": n,
            metric_name: avg_em,
            "avg_similarity": avg_sim,
        }

    # ----- 파일로 저장 -----
    detailed_path = os.path.join(output_dir, "multitask_results_detailed.json")
    summary_path = os.path.join(output_dir, "multitask_results_summary.txt")

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": meta,
                "task_records": per_task_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Multi-Task Evaluation Summary ===\n\n")
        for task, stats in summary["task_results"].items():
            f.write(f"[Task] {task}\n")
            f.write(f"  - #samples       : {stats['num_samples']}\n")
            if task == "bug_detection":
                f.write(f"  - accuracy       : {stats['accuracy']:.4f}\n")
            else:
                f.write(f"  - exact_match    : {stats['exact_match_rate']:.4f}\n")
            f.write(f"  - avg_similarity : {stats['avg_similarity']:.4f}\n")
            f.write("\n")

    print(f"[INFO] Saved detailed results to: {detailed_path}")
    print(f"[INFO] Saved summary results to:  {summary_path}")


# ============================================================
# 메인 함수: 학습 + 평가 파이프라인
# ============================================================

def main():
    print("=" * 60)
    print(" Multi-Task Fine-tuning for CodeT5+ (T5 기반) ")
    print("=" * 60)
    print(f"[INFO] TEST_ONLY = {TEST_ONLY}")
    print(f"[INFO] EVAL_TASKS = {EVAL_TASKS}")
    print("=" * 60)

    # 랜덤 시드 고정
    set_random_seed(RANDOM_SEED)

    # 디바이스 설정
    # NPU 연동되면 좋은데,, 못하겠다 ㅠ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 토크나이저/모델 로드
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    model.to(device)

    if not TEST_ONLY:
        # ---------- 학습 데이터 로드 ----------
        train_raw = load_json(DATA_PATH)

        if not isinstance(train_raw, list):
            raise ValueError("학습 JSON은 리스트(list) 형식이어야 합니다.")

        # ---------- Dataset 생성 ----------
        train_dataset, val_dataset = prepare_datasets(tokenizer, train_raw, val_ratio=0.1)

        # ---------- Data Collator ----------
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
        )

        # ---------- TrainingArguments ----------
        ## 다날려 1
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            logging_steps=50
        )

        # ---------- Trainer ----------
        ## 다날려 2222
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        # ---------- 학습 ----------
        print("[INFO] Start training...")
        trainer.train()
        print("[INFO] Training finished.")

        # ---------- 최종 모델 저장 ----------
        print(f"[INFO] Saving model and tokenizer to: {OUTPUT_DIR}")
        ensure_dir(OUTPUT_DIR)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    else:
        # TEST_ONLY 모드인 경우, OUTPUT_DIR 에서 모델 로드
        print("[INFO] TEST_ONLY = True -> Load model from checkpoint")
        if not os.path.isdir(OUTPUT_DIR):
            raise ValueError(
                f"TEST_ONLY 모드이지만 OUTPUT_DIR ({OUTPUT_DIR}) 이 존재하지 않습니다."
            )
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
        model.to(device)

    # ---------- 평가 ----------
    print("[INFO] Start evaluation on test set...")
    evaluate_on_test_set(
        model=model,
        tokenizer=tokenizer,
        device=device,
        test_path=TEST_DATA_PATH,
        eval_tasks=EVAL_TASKS,
        output_dir=TEST_OUTPUT_DIR,
    )
    print("[INFO] All done.")

## 얜 왜 _main_으로 안되지,, 나중에 체크? 일단 빼
main()