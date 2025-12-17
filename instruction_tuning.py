"""
Multi-Task Instruction Tuning for CodeT5+

This script provides a simple, configurable multi-task instruction tuning framework for CodeT5+.
All configuration is done by editing the constants below - no command line arguments needed.

Tasks supported:
1. Code Summarization: "summarize code:" + code -> description
2. Bug Detection: "is it buggy:" + code -> True/False  
3. Code Repair: "fix a bug:" + buggy_code -> fixed_code
4. Code Generation: "generate code:" + description -> code
5. Combined Bug Fix: "fix a bug if it is buggy:" + code -> analysis + fix

USAGE:
1. Edit the configuration constants below to match your needs
2. Run: python instruction_tuning.py
3. The script will automatically train and test the model

CONFIGURATION:
- Edit DATA_PATH to point to your dataset
- Adjust training parameters (epochs, batch size, learning rate)
- Set TEST_ONLY = True to test an existing model without training
- All paths and parameters can be customized below
"""

import json
import os
import random
import signal
from datetime import datetime
from collections import Counter

import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
)
import subprocess
import tempfile
import os


# Execution Mode
TEST_ONLY = False           # Set to True to only test existing fine-tuned model (no training)
BASE_TEST = False          # Set to True to evaluate base HF model BEFORE training

# Evaluation Settings
# List of tasks to evaluate. Empty list = evaluate all tasks, or you can specify tasks like the following.
TRAIN_TASKS = ['code_repair']
EVAL_TASKS = ['code_search', 'code_repair', 'code_summary', 'code_generation']


# ========================================
# INSTRUCTION PREFIXES FOR EACH TASK
# ========================================
TASK_PREFIXES = {
    "code_search": "choose the correct implementation:",
    "code_repair": "fix a bug:",
    "code_summary": "summarize code:",
    "code_generation": "generate code:",
    "combined_repair": "fix a bug if it is buggy:"
}


# ========================================
# CONFIGURATION - EDIT THESE VALUES
# ========================================

# Data and Model Settings
DATA_PATH = "./data/patterns_train.json"     # Path to your training dataset
TEST_DATA_PATH = "./data/patterns_multitask_test.json"  # Path to your test dataset
OUTPUT_DIR = "./fine_tuned_model"    # Where to save the trained model
TEST_OUTPUT_DIR = "./output"                   # Where to save test results
# MODEL_NAME = "Salesforce/codet5p-220m"         # Base model to fine-tune
MODEL_NAME = "./fine_tuned_model"  
RANDOM_SEED = 42                               # For reproducible results

# Training Settings
TRAIN_BATCH_SIZE = 8           # Number of samples per training batch (reduce if out of memory)
EVAL_BATCH_SIZE = 8            # Number of samples per validation batch
LEARNING_RATE = 2e-5           # How fast the model learns (lower = more stable)
NUM_EPOCHS = 64                # How many times to go through the entire dataset
MAX_INPUT_LENGTH = 1024        # Maximum tokens for input text (code/descriptions)
MAX_TARGET_LENGTH = 512        # Maximum tokens for output text

# Validation and Early Stopping
EARLY_STOPPING_PATIENCE = 3   # Stop training if no improvement for this many evaluations
EVAL_STEPS = 300              # Evaluate model performance every N training steps
SAVE_STEPS = 300              # Save model checkpoint every N steps
VALIDATION_SIZE = 30         # Number of samples to use for validation

# Text Generation Settings (for inference/testing)
GENERATION_MAX_LENGTH = 1024
GENERATION_MIN_LENGTH = 1
GENERATION_NUM_BEAMS = 4
GENERATION_LENGTH_PENALTY = 0.8
GENERATION_REPETITION_PENALTY = 1.1
GENERATION_NO_REPEAT_NGRAM_SIZE = 0

def load_data(path):
    """Load training data from JSON file"""
    print(f"Loading data from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} data items")
    return data

def load_test_data(path):
    """Load test data from JSON file"""
    print(f"Loading test data from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        test_dataset = json.load(f)
    
    test_samples = test_dataset.get('test_samples', [])
    metadata = test_dataset.get('metadata', {})
    
    print(f"Loaded {len(test_samples)} test samples")
    if metadata:
        print(f"Task distribution: {metadata.get('task_distribution', {})}")
    
    return test_samples

class MultiTaskDataset(Dataset):
    """
    Dataset class for multi-task instruction tuning.
    
    Creates training examples for all 5 tasks from each data item:
    1. Code summarization: code -> description
    2. Bug detection: code -> True/False
    3. Code generation: description -> code  
    4. Code repair: buggy_code -> fixed_code
    5. Combined repair: code -> analysis + fix
    """
    
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=512):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.examples = self._create_examples(data)
        
        print(f"Created {len(self.examples)} training examples from {len(data)} data items")
        
    def _create_examples(self, data):
        """Convert raw data into instruction-tuned training examples"""
        examples = []
        
        for item in data:
            nl = item.get('nl', '')           # Natural language description
            pl = item.get('pl', '')           # Programming language code
            is_buggy = item.get('is_buggy', False)  # Whether code has bugs
            fixed_code = item.get('fixed_code', '') # Fixed version of buggy code
            prefix = item.get('prefix', '')   # Function signature prefix
            
            # Skip items missing essential data
            if not nl or not pl:
                continue
            
            if 'code_summary' in TRAIN_TASKS:
                # Task 1: Code Summarization (code -> natural language)
                if not is_buggy:
                    examples.append({
                        'task': 'code_summary',
                        'input': f"{TASK_PREFIXES['code_summary']} {pl}",
                        'output': nl
                    })
            
            if 'bug_detection' in TRAIN_TASKS:
                # Task 2: Bug Detection (code -> True/False)
                bug_label = "True" if is_buggy else "False"
                examples.append({
                    'task': 'bug_detection', 
                    'input': f"{TASK_PREFIXES['bug_detection']} {pl}",
                    'output': bug_label
                })
            
            if 'code_generation' in TRAIN_TASKS:
                # Task 3: Code Generation (description -> code)
                # Use prefix if available for better code generation
                input_text = f"{TASK_PREFIXES['code_generation']} {nl}"
                if prefix and prefix.strip():
                    input_text += f"\nFunction signature: {prefix}"
                
                if not is_buggy:
                    examples.append({
                        'task': 'code_generation',
                        'input': input_text,
                        'output': pl
                    })

            
            if 'code_repair' in TRAIN_TASKS or 'combined_repair' in TRAIN_TASKS: 
                # Task 4: Code Repair (only for buggy code)
                if is_buggy and fixed_code:
                    examples.append({
                        'task': 'code_repair',
                        'input': f"{TASK_PREFIXES['code_repair']} {pl}",
                        'output': fixed_code
                    })
                    
                    # Task 5: Combined Bug Detection & Repair
                    combined_output = f"Fixed code:\n{fixed_code}"
                    examples.append({
                        'task': 'combined_repair',
                        'input': f"{TASK_PREFIXES['combined_repair']} {pl}",
                        'output': combined_output
                    })
                else:
                    # For non-buggy code, combined task should indicate no bug
                    examples.append({
                        'task': 'combined_repair',
                        'input': f"{TASK_PREFIXES['combined_repair']} {pl}",
                        'output': "No, the code is not buggy."
                    })

            
            if 'code_search' in TRAIN_TASKS and 'choices' in item and 'answer' in item:
                choices = item['choices']
                answer = item['answer']  # 0 / 1 / 2

                # 안전장치: 최소 3개 옵션, answer 범위 체크
                if isinstance(choices, list) and len(choices) >= 3 and answer in [0, 1, 2]:
                    # 인풋 문자열 구성
                    input_text = f"{TASK_PREFIXES['code_search']} \n"
                    input_text += "Description:\n"
                    input_text += nl.strip() + "\n\n"
                    input_text += "Choices:\n"
                    for idx, code_snippet in enumerate(choices):
                        input_text += f"Option {idx}:\n{code_snippet}\n\n"

                    # 타겟은 정답 인덱스 하나 (문자열)
                    output_text = str(answer)

                    examples.append({
                        'task': 'code_search',
                        'input': input_text,
                        'output': output_text
                    })
        
        # Shuffle to mix different tasks
        random.shuffle(examples)
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            example['input'],
            max_length=self.max_input_length,
            padding=False,  # Let collate_fn handle padding
            truncation=True,
            return_tensors=None
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            example['output'],
            max_length=self.max_target_length,
            padding=False,  # Let collate_fn handle padding
            truncation=True,
            return_tensors=None
        )
        
        return {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'labels': target_encoding['input_ids']
        }

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Convert to tensors and pad
    input_ids = [torch.tensor(ids) if not isinstance(ids, torch.Tensor) else ids for ids in input_ids]
    attention_masks = [torch.tensor(mask) if not isinstance(mask, torch.Tensor) else mask for mask in attention_masks]
    labels = [torch.tensor(lbls) if not isinstance(lbls, torch.Tensor) else lbls for lbls in labels]
    
    # Pad sequences to the same length within the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }

def prepare_datasets(data, tokenizer, validation_size=VALIDATION_SIZE, seed=42):
    """Split data into training and validation sets"""
    random.seed(seed)
    random.shuffle(data)
    
    # Create full dataset
    full_dataset = MultiTaskDataset(data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = min(validation_size, dataset_size)
    train_size = dataset_size - val_size
    
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(full_dataset, train_indices)
    eval_dataset = Subset(full_dataset, val_indices)
    
    print(f"Dataset split: {len(train_dataset)} training, {len(eval_dataset)} validation samples")
    
    return train_dataset, eval_dataset

def setup_model_and_tokenizer(model_name="Salesforce/codet5p-220m"):
    """Load and configure the model and tokenizer"""
    print(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)
    
    # Add special tokens that might be useful for code tasks
    special_tokens = {
        "additional_special_tokens": [
            "<code>", "</code>", 
            "<bug>", "</bug>",
            "<fix>", "</fix>",
            "<summary>", "</summary>"
        ]
    }
    
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added} special tokens to vocabulary")
    
    return model, tokenizer

def train_model(
    model, 
    tokenizer, 
    train_dataset, 
    eval_dataset, 
    device, 
    output_dir="./fine_tuned_multitask_model",
    num_epochs=NUM_EPOCHS,
    batch_size=TRAIN_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE
):
    """Train the multi-task model using Hugging Face Trainer"""
    
    # Configure training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_steps=300,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        prediction_loss_only=True,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    )
    
    # Setup early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.001
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[early_stopping],
    )
    
    print("\n" + "="*60)
    print("STARTING MULTI-TASK TRAINING")
    print("="*60)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Max epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*60)
    
    # Train the model
    train_result = trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nTraining completed! Model saved to: {output_dir}")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    
    return trainer, train_result

def generate_response(text, tokenizer, model, device, task_type="general"):
    
    """Generate response for given input text"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=GENERATION_MAX_LENGTH,
            min_length=GENERATION_MIN_LENGTH,
            num_beams=GENERATION_NUM_BEAMS,
            length_penalty=GENERATION_LENGTH_PENALTY,
            repetition_penalty=GENERATION_REPETITION_PENALTY,
            no_repeat_ngram_size=GENERATION_NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
            do_sample=False,  # Use beam search for better quality
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def calculate_bleu_score(reference, candidate, max_n=4):
    """Calculate BLEU-4 score with multiple n-grams (standard for code summarization)"""
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if not cand_words or not ref_words:
        return 0.0
    
    # Calculate precision for each n-gram level (1 to 4)
    precisions = []
    
    for n in range(1, max_n + 1):
        if len(cand_words) < n:
            precisions.append(0.0)
            continue
            
        # Generate n-grams
        ref_ngrams = [tuple(ref_words[i:i+n]) for i in range(len(ref_words) - n + 1)]
        cand_ngrams = [tuple(cand_words[i:i+n]) for i in range(len(cand_words) - n + 1)]
        
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        
        # Count overlaps
        ref_counter = Counter(ref_ngrams)
        cand_counter = Counter(cand_ngrams)
        
        overlap = sum((cand_counter & ref_counter).values())
        precision = overlap / len(cand_ngrams)
        precisions.append(precision)
    
    # Brevity penalty
    bp = 1.0
    if len(cand_words) < len(ref_words):
        bp = np.exp(1 - len(ref_words) / len(cand_words))
    
    # Geometric mean of precisions (BLEU-4 formula)
    if all(p > 0 for p in precisions):
        score = bp * np.exp(np.mean(np.log(precisions)))
    else:
        score = 0.0
    
    return score

def calculate_rouge_l(reference, candidate):
    """Calculate ROUGE-L score (longest common subsequence)"""
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if not ref_words or not cand_words:
        return 0.0
    
    # Dynamic programming for LCS
    m, n = len(ref_words), len(cand_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == cand_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    # ROUGE-L F1 score
    if lcs_length == 0:
        return 0.0
    
    recall = lcs_length / len(ref_words)
    precision = lcs_length / len(cand_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Operation timed out")

def execute_code_safely(code, timeout=5):
    """Execute Python code safely with timeout protection and return success status"""
    try:
        # Set up timeout signal (2 seconds for consistency with code_generation_finetuning.py)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)  # 2 second timeout
        
        try:
            # Create a clean namespace for execution
            namespace = {}
            
            # Execute the code in the namespace
            exec(code, namespace)
            
            return True, "", ""
        finally:
            # Always clean up the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
    except TimeoutError:
        return False, "", "Code execution timeout (possible infinite loop)"
    except Exception as e:
        return False, "", str(e)

# def evaluate_code_repair(fixed_code, expected_code=None):
#     """
#     Evaluate code repair using execution-based testing (Pass@1)
#     Returns: (execution_success, syntax_valid, similarity_score)
#     """
#     # Check syntax validity
#     try:
#         compile(fixed_code, '<string>', 'exec')
#         syntax_valid = True
#     except SyntaxError:
#         syntax_valid = False
    
#     # Test execution
#     execution_success, stdout, stderr = execute_code_safely(fixed_code)
    
#     # Calculate similarity to expected if available
#     similarity_score = 0.0
#     if expected_code:
#         similarity_score = calculate_rouge_l(expected_code, fixed_code)
    
#     return execution_success, syntax_valid, similarity_score

def basic_syntax_sanity(code: str) -> bool:
    """초경량 sanity 체크: 완전히 깨진 코드만 걸러냄 (false negative 최소화 목적)."""
    if not isinstance(code, str) or not code.strip():
        return False
    return (
        code.count("(") == code.count(")") and
        code.count("{") == code.count("}") and
        code.count("<") <= code.count(">") and
        "const Example" in code
    )

def run_static_tests(code: str, tests) -> bool:
    """DSL: must_contain:* 전부 만족하면 True"""
    if not tests:
        # tests가 없으면 execution_success를 판단할 근거가 없으니 True로 두지 말고 False 권장
        return False
    for t in tests:
        if t.startswith("must_contain:"):
            needle = t.replace("must_contain:", "")
            if needle not in code:
                return False
    return True

def node_syntax_check_js(code: str) -> bool:
    """
    Node --check로 JS 문법 검사.
    - TS/JSX는 완벽히 지원하지 않을 수 있어 false negative 가능
    - 그래서 basic_syntax_sanity와 OR로 묶어 사용하는 걸 권장
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".js", mode="w", encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["node", "--check", tmp_path],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
def evaluate_code_repair(fixed_code: str, tests=None, expected_code: str = None):
    """
    Static evaluation for code repair (SS-Pass@1).

    - Syntax validity: (basic sanity) OR (node --check)  [node는 JSX/TS에서 false negative 가능]
    - Execution success: passes all static DSL tests
    - Similarity: ROUGE-L (optional)

    Returns: (execution_success, syntax_valid, similarity_score)
    Success criterion: syntax_valid AND execution_success
    """

    # 1) Syntax validity (static)
    syntax_valid = basic_syntax_sanity(fixed_code) or node_syntax_check_js(fixed_code)

    # 2) Execution success proxy (static structural conformance)
    execution_success = run_static_tests(fixed_code, tests) if tests is not None else False

    # 3) Similarity (optional)
    similarity_score = 0.0
    if expected_code:
        similarity_score = calculate_rouge_l(expected_code, fixed_code)

    return execution_success, syntax_valid, similarity_score


def evaluate_code_search(prediction: str, expected: str):
    """
    Code search evaluation logic:
    - Extract the first digit (0/1/2) from the model's output
    - If the extracted digit matches the expected answer, return True
    - If no digit is found, return False
    """
    if prediction is None:
        return False

    prediction = str(prediction)
    expected = str(expected).strip()

    # Extract the first digit from the model's output
    first_digit = None
    for ch in prediction:
        if ch in "0123456789":   # valid candidate digit
            first_digit = ch
            break

    # If no digit is found, it's incorrect
    if first_digit is None:
        return False

    # Correct if the first digit matches the expected answer
    return first_digit == expected


def evaluate_bug_detection(prediction, expected):
    """Robust evaluation for bug detection (handles messy LLM outputs)."""
    pred_text = prediction.lower()

    # Remove surrounding symbols
    pred_text = pred_text.replace("\n", " ").replace("\t", " ")

    # Find positions of keywords
    pos_true = pred_text.find("true")
    pos_false = pred_text.find("false")

    # Determine model's predicted boolean
    if pos_true == -1 and pos_false == -1:
        # No valid signal found
        return False
    elif pos_false == -1:
        pred_bool = True
    elif pos_true == -1:
        pred_bool = False
    else:
        # Both appear → use the earliest occurrence
        pred_bool = pos_true < pos_false

    # Expected output
    expected_bool = expected.strip().lower() == "true"

    return pred_bool == expected_bool

def evaluate_code_generation(generated_code, expected_output=None, description=None):
    """
    Evaluate code generation using execution and syntax checking
    Returns: (execution_success, syntax_valid, functionality_score)
    """
    # Check syntax validity
    try:
        compile(generated_code, '<string>', 'exec')
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False
    
    # Test execution
    execution_success, stdout, stderr = execute_code_safely(generated_code)
    
    # Calculate functionality score based on expected output or description
    functionality_score = 0.0
    if expected_output:
        functionality_score = calculate_rouge_l(expected_output, generated_code)
    elif description:
        # Simple keyword matching for functionality
        desc_keywords = set(description.lower().split())
        code_words = set(generated_code.lower().split())
        common_keywords = desc_keywords & code_words
        if desc_keywords:
            functionality_score = len(common_keywords) / len(desc_keywords)
    
    return execution_success, syntax_valid, functionality_score

def _get_tests_from_result(result):
    # test_samples 생성 방식에 따라 tests 위치가 달라질 수 있어서 안전하게 찾음
    if isinstance(result.get("original_item"), dict) and result["original_item"].get("tests"):
        return result["original_item"]["tests"]
    if result.get("tests"):
        return result["tests"]
    return None


def evaluate_single_result(task_name, result):
    """Evaluate a single result and return correctness status and score"""
    expected = result['expected_output']
    generated = result['model_output']

    if task_name in ['code_summary', 'code_generation']:
        bleu_score = calculate_bleu_score(expected, generated)
        is_correct = bleu_score > 0.15
        score = bleu_score

    elif task_name == 'bug_detection':
        is_correct = evaluate_bug_detection(generated, expected)
        score = 1.0 if is_correct else 0.0

    elif task_name in ['code_repair', 'combined_repair']:
        tests = _get_tests_from_result(result)

        # combined_repair에서 자연어 답변만 오는 경우 처리
        if task_name == 'combined_repair' and 'Fixed code:' in generated:
            code_part = generated.split('Fixed code:', 1)[1].strip()

        elif task_name == 'combined_repair' and 'No, the code is not buggy.' in generated:
            # "버그 없음" 응답은 코드가 아니라 자연어이므로,
            # SS-Pass@1 평가 대신 유사도만 계산 (정답 여부도 유사도로 판단하고 싶으면 별도 정책 필요)
            score = calculate_rouge_l(generated, expected)
            is_correct = (score > 0.5)  # ✅ 원하는 임계값으로 조정 가능
            return is_correct, score

        else:
            code_part = generated

        # ✅ SS-Pass@1 평가: (execution_success, syntax_valid, similarity)
        execution_success, syntax_valid, score = evaluate_code_repair(
            code_part,
            tests=tests,
            expected_code=expected
        )

        # One-time sanity log for code_repair path
        if not getattr(evaluate_single_result, "_code_repair_logged", False):
            print(f"[debug][code_repair] tests_missing={tests is None}, execution_success={execution_success}, syntax_valid={syntax_valid}")
            evaluate_single_result._code_repair_logged = True

        is_correct = bool(syntax_valid and execution_success)

    elif task_name == 'code_search':
        is_correct = evaluate_code_search(generated, expected)
        score = 1.0 if is_correct else 0.0

    return is_correct, score
def evaluate_task_results(task_name, results):
    """Evaluate results for a specific task and return metrics"""
    if not results:
        return {}

    metrics = {
        'total_samples': len(results),
        'success_rate': 0.0,
        'avg_score': 0.0,
        'additional_metrics': {}
    }

    successful_samples = 0
    total_score = 0.0

    if task_name in ['code_summary', 'code_generation']:
        bleu_scores = []
        rouge_scores = []
        structure_passes = 0
        structure_tests_available = 0
        syntax_valid_count = 0

        for result in results:
            expected = result['expected_output']
            generated = result['model_output']
            tests = _get_tests_from_result(result)

            bleu_score = calculate_bleu_score(expected, generated)
            rouge_score = calculate_rouge_l(expected, generated)

            bleu_scores.append(bleu_score)
            rouge_scores.append(rouge_score)

            if bleu_score > 0.15:
                successful_samples += 1

            total_score += bleu_score

            if tests:
                structure_tests_available += 1
                if run_static_tests(generated, tests):
                    structure_passes += 1

            if basic_syntax_sanity(generated) or node_syntax_check_js(generated):
                syntax_valid_count += 1

        metrics['additional_metrics'] = {
            'avg_bleu_score': round(np.mean(bleu_scores), 3),
            'avg_rouge_l': round(np.mean(rouge_scores), 3),
            'max_bleu_score': round(max(bleu_scores), 3),
            'min_bleu_score': round(min(bleu_scores), 3),
            'bleu_above_015': sum(1 for s in bleu_scores if s > 0.15),
            'bleu_above_025': sum(1 for s in bleu_scores if s > 0.25),
            'success_threshold': 0.15,
            'structure_pass_rate': round(structure_passes / structure_tests_available, 3) if structure_tests_available else 0.0,
            'structure_tests_available': structure_tests_available,
            'syntax_validity_rate': round(syntax_valid_count / len(results), 3),
        }

    elif task_name == 'bug_detection':
        correct_predictions = 0

        for result in results:
            expected = result['expected_output']
            predicted = result['model_output']

            is_correct = evaluate_bug_detection(predicted, expected)
            if is_correct:
                correct_predictions += 1
                successful_samples += 1

            total_score += 1.0 if is_correct else 0.0

        metrics['additional_metrics'] = {
            'accuracy': round(correct_predictions / len(results), 3),
            'correct_predictions': correct_predictions
        }

    elif task_name == 'code_search':
        correct_predictions = 0

        for result in results:
            expected = result['expected_output']
            predicted = result['model_output']

            is_correct = evaluate_code_search(predicted, expected)
            if is_correct:
                correct_predictions += 1
                successful_samples += 1

            total_score += 1.0 if is_correct else 0.0

        metrics['additional_metrics'] = {
            'accuracy': round(correct_predictions / len(results), 3),
            'correct_predictions': correct_predictions
        }

    elif task_name in ['code_repair', 'combined_repair']:
        # ✅ SS-Pass@1 평가
        pass_at_1_count = 0
        syntax_valid_count = 0
        exec_success_count = 0
        similarity_scores = []

        for result in results:
            expected = result['expected_output']
            generated = result['model_output']
            tests = _get_tests_from_result(result)

            # combined_repair에서 자연어만 답하는 케이스 처리
            if task_name == 'combined_repair' and 'Fixed code:' in generated:
                code_part = generated.split('Fixed code:', 1)[1].strip()

            elif task_name == 'combined_repair' and 'No, the code is not buggy.' in generated:
                # 자연어 응답은 구조 tests가 적용 불가 → 유사도만
                similarity = calculate_rouge_l(generated, expected)
                similarity_scores.append(similarity)
                total_score += similarity

                # 성공 판정 정책: 유사도 임계값 기반(원하면 조정)
                is_pass = similarity > 0.5
                if is_pass:
                    pass_at_1_count += 1
                    successful_samples += 1
                continue

            else:
                code_part = generated

            execution_success, syntax_valid, similarity = evaluate_code_repair(
                code_part,
                tests=tests,
                expected_code=expected
            )

            if syntax_valid:
                syntax_valid_count += 1
            if execution_success:
                exec_success_count += 1

            is_pass = bool(syntax_valid and execution_success)
            if is_pass:
                pass_at_1_count += 1
                successful_samples += 1

            similarity_scores.append(similarity)
            total_score += similarity

        metrics['additional_metrics'] = {
            'pass_at_1': round(pass_at_1_count / len(results), 3),
            'syntax_validity_rate': round(syntax_valid_count / len(results), 3),
            'execution_success_rate': round(exec_success_count / len(results), 3),
            'avg_similarity': round(float(np.mean(similarity_scores)) if similarity_scores else 0.0, 3),
        }

    metrics['success_rate'] = round(successful_samples / len(results), 3)
    metrics['avg_score'] = round(total_score / len(results), 3)

    return metrics


def test_multitask_model(model, tokenizer, device, test_samples=None):
    """Test the trained model on all test samples"""
    
    if test_samples is None:
        # Load test samples from file if not provided
        if os.path.exists(TEST_DATA_PATH):
            test_samples = load_test_data(TEST_DATA_PATH)
        else:
            print(f"Error: Test data file {TEST_DATA_PATH} not found")
            return
    
    print("\n" + "="*60)
    print("TESTING MULTI-TASK MODEL")
    print("="*60)
    print(f"Testing with {len(test_samples)} samples")
    
    # Count samples by task
    task_counts = {}
    for sample in test_samples:
        task = sample['task']
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print("Test samples by task:")
    for task, count in task_counts.items():
        print(f"  {task}: {count} samples")
    print("="*60)
    
    model.eval()
    
    # Prepare detailed results for file
    detailed_results = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_dir': OUTPUT_DIR,
            'total_samples': len(test_samples),
            'task_distribution': task_counts
        },
        'results_by_task': {},
        'evaluation_metrics': {}
    }
    
    # Test all samples and collect results
    task_summaries = {}
    
    # Determine which tasks to evaluate
    tasks_to_evaluate = EVAL_TASKS if EVAL_TASKS else list(TASK_PREFIXES.keys())
    
    # Validate task names
    invalid_tasks = [task for task in tasks_to_evaluate if task not in TASK_PREFIXES]
    if invalid_tasks:
        print(f"ERROR: Invalid task names in EVAL_TASKS: {invalid_tasks}")
        print(f"Available tasks: {list(TASK_PREFIXES.keys())}")
        return
    
    print(f"Evaluating tasks: {tasks_to_evaluate}")
    if EVAL_TASKS:
        print(f"Note: Only evaluating {len(EVAL_TASKS)} out of {len(TASK_PREFIXES)} available tasks")
    
    for task_name in tasks_to_evaluate:
        task_samples = [s for s in test_samples if s['task'] == task_name]
        if task_samples:
            print(f"Testing {task_name}: {len(task_samples)} samples...", end="", flush=True)
            
            task_results = []
            
            for i, sample in enumerate(task_samples):
                response = generate_response(sample['input'], tokenizer, model, device, sample['task'])
                
                # Normalize tests passthrough (avoid nulls in results JSON)
                tests_from_sample = sample.get('tests') or sample.get('original_item', {}).get('tests')

                result = {
                    'sample_id': i + 1,
                    'input': sample['input'],
                    'expected_output': sample.get('expected_output', ''),
                    'model_output': response,
                    'input_length': len(sample['input']),
                    'output_length': len(response),
                    # Pass-through for evaluators (needed for code_repair tests)
                    'original_item': sample.get('original_item', {}),
                }

                # Add correctness evaluation
                is_correct, score = evaluate_single_result(task_name, result)
                result['correct'] = bool(is_correct)
                result['score'] = round(score, 3)
                
                task_results.append(result)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f" {i + 1}", end="", flush=True)
            
            print(" ✓")
            
            # Store detailed results
            detailed_results['results_by_task'][task_name] = task_results
            
            # Evaluate task performance using appropriate metrics
            print(f"  Evaluating {task_name}...", end="", flush=True)
            task_metrics = evaluate_task_results(task_name, task_results)
            print(" ✓")
            
            # Store evaluation metrics
            task_summaries[task_name] = task_metrics
            detailed_results['evaluation_metrics'][task_name] = task_metrics
    
    # Calculate overall success rate across all tasks
    overall_success_rate = 0.0
    total_samples_evaluated = 0
    weighted_success = 0.0
    
    for task_name, metrics in task_summaries.items():
        task_samples = metrics['total_samples']
        task_success = metrics['success_rate']
        
        weighted_success += task_success * task_samples
        total_samples_evaluated += task_samples
    
    if total_samples_evaluated > 0:
        overall_success_rate = weighted_success / total_samples_evaluated
    
    # Add overall metrics to results
    detailed_results['evaluation_metrics']['overall'] = {
        'overall_success_rate': round(overall_success_rate, 3),
        'total_samples_evaluated': total_samples_evaluated,
        'tasks_evaluated': len(task_summaries)
    }
    
    # Save detailed results to file
    if BASE_TEST:
        file_name = f"base_model_results"
    else:
        file_name = f"tuning_model_results"

    results_file = os.path.join(TEST_OUTPUT_DIR, f'{file_name}_detailed.json')
    results_summary_file = os.path.join(TEST_OUTPUT_DIR, f'{file_name}_summary.txt')
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Create human-readable summary file
    with open(results_summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-TASK MODEL TEST RESULTS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Directory: {OUTPUT_DIR}\n")
        f.write(f"Total samples tested: {len(test_samples)}\n")
        f.write(f"Tasks tested: {len(task_counts)}\n")
        f.write(f"Overall Success Rate: {overall_success_rate:.1%}\n\n")
        
        f.write("Task Distribution:\n")
        for task, count in task_counts.items():
            f.write(f"  {task}: {count} samples\n")
        f.write("\n")
        
        f.write("EVALUATION METRICS BY TASK:\n")
        f.write("="*80 + "\n")
        for task_name, metrics in task_summaries.items():
            f.write(f"\n{task_name.upper()}:\n")
            f.write(f"  Samples tested: {metrics['total_samples']}\n")
            f.write(f"  Success rate: {metrics['success_rate']:.1%}\n")
            f.write(f"  Average score: {metrics['avg_score']:.3f}\n")
            
            # Task-specific metrics
            additional = metrics.get('additional_metrics', {})
            # if task_name == 'code_summary':
            if task_name in ['code_summary', 'code_generation']:
                f.write(f"  Average BLEU-4 score: {additional.get('avg_bleu_score', 0):.3f}\n")
                f.write(f"  Max BLEU-4: {additional.get('max_bleu_score', 0):.3f}\n")
                f.write(f"  BLEU > 0.15 (good): {additional.get('bleu_above_015', 0)}/{metrics['total_samples']}\n")
                f.write(f"  BLEU > 0.25 (high quality): {additional.get('bleu_above_025', 0)}/{metrics['total_samples']}\n")
                f.write(f"  Average ROUGE-L: {additional.get('avg_rouge_l', 0):.3f}\n")
            # elif task_name == 'bug_detection':
            elif task_name in ['bug_detection', 'code_search']:
                f.write(f"  Accuracy: {additional.get('accuracy', 0):.1%}\n")
                f.write(f"  Correct predictions: {additional.get('correct_predictions', 0)}\n")    
            # elif task_name in ['code_repair', 'code_generation', 'combined_repair']:
            elif task_name in ['code_repair', 'combined_repair']:
                f.write(f"  Pass@1 (execution): {additional.get('pass_at_1', 0):.1%}\n")
                f.write(f"  Syntax validity: {additional.get('syntax_validity_rate', 0):.1%}\n")
                # f.write(f"  Execution success: {additional.get('execution_success_rate', 0):.1%}\n")
                if 'avg_similarity' in additional:
                    f.write(f"  Average similarity: {additional['avg_similarity']:.3f}\n")
                if 'avg_functionality' in additional:
                    f.write(f"  Average functionality: {additional['avg_functionality']:.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SAMPLE OUTPUTS (First 3 per task)\n")
        f.write("="*80 + "\n")
        
        for task_name in tasks_to_evaluate:
            if task_name in detailed_results['results_by_task']:
                task_results = detailed_results['results_by_task'][task_name]
                f.write(f"\n{task_name.upper()}:\n")
                f.write("-" * 40 + "\n")
                
                for i, result in enumerate(task_results[:3]):  # Show first 3 samples
                    f.write(f"\nSample {i+1}:\n")
                    f.write(f"Input: {result['input'][:200]}...\n")
                    f.write(f"Expected: {result['expected_output'][:200]}...\n")
                    f.write(f"Output: {result['model_output'][:200]}...\n")
                    f.write("-" * 20 + "\n")
    
    # Show summary on screen
    print(f"\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"Overall Success Rate: {overall_success_rate:.1%}")
    print(f"Total samples tested: {len(test_samples)}")
    print(f"Tasks evaluated: {len(tasks_to_evaluate)}/{len(TASK_PREFIXES)} ({', '.join(tasks_to_evaluate)})")
    if EVAL_TASKS:
        excluded_tasks = [task for task in TASK_PREFIXES.keys() if task not in tasks_to_evaluate]
        if excluded_tasks:
            print(f"Tasks excluded: {', '.join(excluded_tasks)}")
    print()
    print("Per-task performance:")
    for task_name, metrics in task_summaries.items():
        additional = metrics.get('additional_metrics', {})
        print(f"  {task_name}:")
        print(f"    Success rate: {metrics['success_rate']:.1%}")
        print(f"    Samples: {metrics['total_samples']}")
        
        # Show key metric for each task
        if task_name in ['code_summary', 'code_generation']:
            print(f"    BLEU-4: {additional.get('avg_bleu_score', 0):.3f}")
        elif task_name in ['bug_detection', 'code_search']:
            print(f"    Accuracy: {additional.get('accuracy', 0):.1%}")
        elif task_name in ['code_repair', 'combined_repair']:
            print(f"    Pass@1: {additional.get('pass_at_1', 0):.1%}")
    
    print(f"\nDetailed results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  Text: {results_summary_file}")
    print("="*60)

def save_training_config(output_dir, config):
    """Save training configuration for future reference"""
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved to: {config_path}")

def main():
    """
    Main function - runs the multi-task instruction tuning.
    Edit the configuration constants at the top of the file to customize training.
    """
    print("="*60)
    print("MULTI-TASK INSTRUCTION TUNING FOR CODET5+")
    print("="*60)
    print(f"Data path: {DATA_PATH}")
    print(f"Test data path: {TEST_DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Validation size: {VALIDATION_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Test only mode: {TEST_ONLY}")
    print(f"Based model test: {BASE_TEST}")
    print("="*60)
    
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    # Determine device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU for training")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS for training")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    model.to(device)
    
    if TEST_ONLY:
        if BASE_TEST:
            # Evaluate base model before training
            print("\nBASE_TEST mode: evaluating base HF model (no fine-tuned weights).")
            test_multitask_model(model, tokenizer, device)
            return

        # Only test existing model
        print("Test-only mode: Loading existing model")
        if os.path.exists(OUTPUT_DIR):
            model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR)
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
            model.to(device)
            test_multitask_model(model, tokenizer, device)
        else:
            print(f"Error: Model directory {OUTPUT_DIR} not found")
        return
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found: {DATA_PATH}")
        return
    
    # Load and prepare training data
    data = load_data(DATA_PATH)
    train_dataset, eval_dataset = prepare_datasets(data, tokenizer, VALIDATION_SIZE, RANDOM_SEED)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train the model
    trainer, train_result = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        device=device,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Save training configuration for reference
    config = {
        'model_name': MODEL_NAME,
        'data_path': DATA_PATH,
        'num_epochs': NUM_EPOCHS,
        'batch_size': TRAIN_BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'validation_size': VALIDATION_SIZE,
        'seed': RANDOM_SEED,
        'final_training_loss': train_result.training_loss,
        'task_prefixes': TASK_PREFIXES,
        'total_training_samples': len(train_dataset),
        'total_validation_samples': len(eval_dataset)
    }
    save_training_config(OUTPUT_DIR, config)
    
    # Test the trained model
    print("\nTesting the trained multi-task model...")
    test_multitask_model(model, tokenizer, device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
