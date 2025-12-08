''' 
- 이거 왜안돼 (파일 별도로 빼기)
- 오전중에 확인마치기
- 어느정도 데이터도 구축해두기? 짐 좀 덜어줘야지
- 하고 검토만 부탁하ㅣㄱ
'''

# build_multitask_testset.py
import json
import os
import random
from collections import Counter

# === 경로 설정: main.py가 있는 폴더 기준으로 맞추기 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "origin.json")          # 학습용
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.json") # 만들 테스트셋

# 테스트셋에 사용할 "원본 샘플" 개수 (각 샘플에서 최대 4개의 task 예제가 생김)
NUM_BASE_SAMPLES_FOR_TEST = 50

# main.py에서 쓴 것과 동일한 prefix를 유지하는 게 중요함
TASK_PREFIXES = {
    'code_summary': 'summarize code:',
    'bug_detection': 'is it buggy:',
    'code_repair': 'fix a bug:',
    'code_generation': 'generate code:'
}

def load_train_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} items from {path}")
    return data

def build_test_samples(data, num_base_samples=50):
    # 원본 데이터에서 일부만 뽑아서 테스트셋으로 사용
    if len(data) <= num_base_samples:
        base_items = data
    else:
        base_items = random.sample(data, num_base_samples)

    test_samples = []

    for item in base_items:
        nl = (item.get("nl") or "").strip()
        pl = (item.get("pl") or "").strip()
        is_buggy = bool(item.get("is_buggy", False))
        fixed_code = (item.get("fixed_code") or "").strip()
        prefix = (item.get("prefix") or "").strip()

        # nl, pl 둘 다 있어야 4개 task를 만들 수 있음
        if not nl or not pl:
            continue

        # 1) Code Summarization
        test_samples.append({
            "task": "code_summary",
            "input": f"{TASK_PREFIXES['code_summary']} {pl}",
            "expected_output": nl,
        })

        # 2) Bug Detection
        bug_label = "True" if is_buggy else "False"
        bug_input = (f"{TASK_PREFIXES['bug_detection']} {pl}\n""!Answer with exactly one word: 'True' or 'False'.")
        test_samples.append({
            "task": "bug_detection",
            "input": bug_input,
            "expected_output": bug_label,
        })

        # 3) Code Generation (nl (+ prefix) -> pl)
        input_text = f"{TASK_PREFIXES['code_generation']} {nl}"
        if prefix:
            input_text += f"\nFunction signature: {prefix}"
        test_samples.append({
            "task": "code_generation",
            "input": input_text,
            "expected_output": pl,
        })

        # 4) Code Repair (buggy일 때만)
        if is_buggy and fixed_code:
            test_samples.append({
                "task": "code_repair",
                "input": f"{TASK_PREFIXES['code_repair']} {pl}",
                "expected_output": fixed_code,
            })

    counter = Counter(s["task"] for s in test_samples)
    print("[INFO] Task distribution in test set:")
    for t, c in counter.items():
        print(f"  - {t}: {c}")

    metadata = {
        "source_train_file": TRAIN_DATA_PATH,
        "num_original_items": len(data),
        "num_base_items_for_test": len(base_items),
        "num_test_samples": len(test_samples),
        "task_distribution": dict(counter),
    }

    return {
        "metadata": metadata,
        "test_samples": test_samples,
    }

def main():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    data = load_train_data(TRAIN_DATA_PATH)
    test_dataset = build_test_samples(data, NUM_BASE_SAMPLES_FOR_TEST)

    with open(TEST_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Saved multi-task test set to {TEST_DATA_PATH}")

if __name__ == "__main__":
    main()