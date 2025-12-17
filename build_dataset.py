"""
Make dataset for Code LLM

1) 웹 크롤링해서 총 1,101개 링크 확보 (patterns_urls.txt)
(이 파일이 이미 있으면 크롤링 단계 스킵)
"""
import random
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from urllib.parse import urljoin
from collections import defaultdict, Counter

from playwright.async_api import async_playwright



# CONFIG
# ----------------------------

URL_PATH = Path("./data/patterns_urls.txt")
DATA_PATH = Path("./data/patterns_data.json")
DATA_PRE_PATH = Path("./data/patterns_data_pre.json")
TRAIN_PATH = Path("data/patterns_train.json")
TEST_PATH  = Path("data/patterns_test.json")
MULTITASK_TEST_PATH = Path("data/patterns_multitask_test.json")

TASK_PREFIXES = {
    "code_search": "choose the correct implementation:",
    "code_repair": "fix a bug:",
    "code_summary": "summarize code:",
    "code_generation": "generate code:",
    "bug_detection": "is it buggy:",
}

NUMBER_OF_TEST = 30

#----------------------------

BASE = "https://www.shadcn.io"
START = "https://www.shadcn.io/patterns"

DESC_SELECTOR = "p.mb-8.text-lg.text-fd-muted-foreground"

SEED = 42 
random.seed(SEED)

# ----------------------------
# 1) URL 수집 파트
# ----------------------------

def clean_pattern_href(h: str) -> str | None:
    if not h:
        return None
    if not h.startswith("/patterns/"):
        return None
    if h == "/patterns":
        return None
    h = h.split("?")[0].split("#")[0].rstrip("/")
    if h.count("/") == 2:  # /patterns/<slug>
        return h
    return None


async def collect_pattern_urls() -> list[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("1) 페이지 여는 중...")
        await page.goto(START, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)

        sidebar_viewport = page.locator("div[data-radix-scroll-area-viewport]").first
        if await sidebar_viewport.count() == 0:
            await browser.close()
            raise RuntimeError("사이드바 스크롤 영역을 못 찾았어. 셀렉터가 달라졌을 수 있어.")

        print("2) 왼쪽 메뉴를 '전부 펼치는 중'...")
        for _ in range(50):
            closed_buttons = sidebar_viewport.locator('button[aria-expanded="false"]')
            n = await closed_buttons.count()
            if n == 0:
                break
            for i in range(min(n, 20)):
                try:
                    await closed_buttons.nth(i).click(timeout=2000)
                    await page.wait_for_timeout(50)
                except Exception:
                    pass

        print("3) 사이드바를 끝까지 스크롤하면서 링크 수집 중...")
        urls: set[str] = set()
        last_count = -1

        for _ in range(200):
            links = sidebar_viewport.locator("a[href^='/patterns/']")
            hrefs = await links.evaluate_all(
                "els => els.map(e => e.getAttribute('href')).filter(Boolean)"
            )

            for h in hrefs:
                ch = clean_pattern_href(h)
                if ch:
                    urls.add(urljoin(BASE, ch))

            print("현재 링크 수:", len(urls))

            if len(urls) == last_count:
                before = await sidebar_viewport.evaluate("el => el.scrollTop")
                await sidebar_viewport.evaluate("el => { el.scrollTop = el.scrollTop + el.clientHeight * 2; }")
                await page.wait_for_timeout(200)
                after = await sidebar_viewport.evaluate("el => el.scrollTop")
                if after == before:
                    break
            else:
                last_count = len(urls)
                await sidebar_viewport.evaluate("el => { el.scrollTop = el.scrollTop + el.clientHeight * 2; }")
                await page.wait_for_timeout(200)

        await browser.close()
        return sorted(urls)


def load_urls_from_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_urls_to_file(path: Path, urls: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # ✅ ./data 없으면 생성
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(urls) + "\n")

async def ensure_urls_file() -> list[str]:
    if URL_PATH.exists() and URL_PATH.stat().st_size > 0:
        urls = load_urls_from_file(URL_PATH)
        print(f"1-3) 데이터 URL 수집이 이미 완료되어있어서 크롤링 생략: {URL_PATH} (COUNT={len(urls)})")
        return urls

    urls = await collect_pattern_urls()
    save_urls_to_file(URL_PATH, urls)
    print(f"✅ 새로 저장 완료: {URL_PATH} (COUNT={len(urls)})")
    return urls

# ----------------------------
# 2) nl/pl/prefix 생성 파트
# ----------------------------
@dataclass
class Row:
    nl: str
    pl: str
    prefix: str
    component_type: str
    url: str
    

def extract_prefix(tsx: str) -> str:
    lines = tsx.splitlines()

    # 'const Example = () => (' 형태를 정확히 찾기 (공백 변형 허용)
    example_pat = re.compile(r"^\s*const\s+Example\s*=\s*\(\s*\)\s*=>\s*\(\s*$")

    out: List[str] = []
    for line in lines:
        out.append(line)
        if example_pat.match(line):
            break

    # 만약 못 찾으면(예외 케이스), fallback: 그냥 import만이라도 반환
    if not any(example_pat.match(l) for l in lines):
        out = []
        for line in lines:
            out.append(line)
            # 첫 컴포넌트 선언에서 멈추기(대충)
            if re.match(r"^\s*(const|function|export\s+function|export\s+default\s+function)\s+\w+", line):
                break

    return "\n".join(out).rstrip() + "\n"



async def fetch_one_pattern(page, url: str) -> Optional[Row]:
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(800)

    # nl: description paragraph
    desc = ""
    try:
        loc = page.locator(DESC_SELECTOR).first
        if await loc.count():
            desc = (await loc.inner_text()).strip()
    except Exception:
        desc = ""

    # Code 탭 클릭
    try:
        await page.get_by_role("tab", name="Code").click(timeout=3000)
    except Exception:
        try:
            await page.get_by_text("Code", exact=True).click(timeout=3000)
        except Exception:
            pass

    await page.wait_for_timeout(500)

    # pl: code block (pre)
    pl = ""
    try:
        pre = page.locator("pre").first
        if await pre.count():
            pl = (await pre.inner_text()).strip()
    except Exception:
        pl = ""

    if not pl:
        return None

    nl = desc if desc else "Implement this UI pattern."
    nl += "."

    prefix = extract_prefix(pl)
    
    component_type = extract_component_type(url)

    return Row(nl=nl, pl=pl, prefix=prefix, component_type=component_type, url=url)

def extract_component_type(url: str) -> str:
    """
    https://www.shadcn.io/patterns/aspect-ratio-standard-1
    -> aspect
    """
    try:
        slug = url.split("/patterns/")[1]
        return slug.split("-")[0]
    except Exception:
        return "unknown"

async def build_dataset(urls: list[str]) -> int:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = [] 

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        written = 0
        for i, url in enumerate(urls, 1):
            try:
                row = await fetch_one_pattern(page, url)
            except Exception as e:
                print(f"[{i}/{len(urls)}] FAIL: {url} ({type(e).__name__})")
                continue

            if row is None:
                print(f"[{i}/{len(urls)}] SKIP(no code): {url}")
                continue

            rows.append(row.__dict__)
            written += 1

            if i % 25 == 0:
                print(f"[{i}/{len(urls)}] progress... written={written}")

        await browser.close()

    with DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return written


# ----------------------------
# 3) 데이터 정제 파트
# ----------------------------

def preprocessing_data(ratio=1.0, seed=42):
    """
    DATA_PATH를 읽어서
    1) choices/answer 추가
    2) tests 추가
    3) buggy 데이터 추가 (정상 + buggy 같이 포함)
    결과를 DATA_PRE_PATH에 저장
    """
    random.seed(seed)

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    data = add_choices_and_answer(data)
    data = add_tests(data)
    data = add_buggy_dataset(data, ratio=ratio)

    DATA_PRE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PRE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ preprocessed dataset saved to {DATA_PRE_PATH} (count={len(data)})")
    return data


def add_choices_and_answer(data):
    n = len(data)
    if n < 3:
        raise ValueError("choices를 만들려면 데이터가 최소 3개 필요해요.")

    out = []
    for i, item in enumerate(data):
        correct = item["pl"]

        candidates = [j for j in range(n) if j != i]
        wrong_idxs = random.sample(candidates, 2)
        wrongs = [data[j]["pl"] for j in wrong_idxs]

        answer = random.randint(0, 2)

        choices = [None] * 3
        choices[answer] = correct

        wi = 0
        for k in range(3):
            if choices[k] is None:
                choices[k] = wrongs[wi]
                wi += 1

        new_item = dict(item)  # shallow copy OK (우린 list를 공유하지 않음)
        new_item["choices"] = choices
        new_item["answer"] = answer
        out.append(new_item)

    return out


def add_tests(data):
    out = []
    for item in data:
        pl_code = item["pl"]

        tests = ["must_contain:export default Example"]

        jsx_tags = extract_jsx_component_tags(pl_code)
        for tag in jsx_tags[:4]:
            tests.append(f"must_contain:<{tag}")

        while len(tests) < 5:
            tests.append("must_contain:<")

        new_item = dict(item)
        new_item["tests"] = tests
        new_item["is_buggy"] = False
        new_item["fixed_code"] = ""  # ✅ 스키마 통일
        out.append(new_item)

    return out


def extract_jsx_component_tags(code):
    pattern = re.compile(r"<([A-Z][A-Za-z0-9]*)\b")
    seen = set()
    ordered = []
    for match in pattern.finditer(code):
        tag = match.group(1)
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def run_static_tests(code, tests):
    for t in tests:
        if t.startswith("must_contain:"):
            needle = t.replace("must_contain:", "")
            if needle not in code:
                return False
    return True


def add_buggy_dataset(data, ratio=1.0):
    """
    입력: 정상 데이터(이미 tests 포함)
    출력: 정상 + buggy가 섞인 리스트
    - ratio=1.0: 각 정상 item마다 buggy 1개를 시도
    - 실패하면(생성 불가/검증 실패) buggy는 스킵하고 정상만 유지
    """
    out = []
    made = 0
    skipped = 0

    for item in data:
        out.append(item)  # 정상 유지

        if random.random() < ratio:
            buggy = make_buggy_item_safe(item)
            if buggy is not None:
                out.append(buggy)
                made += 1
            else:
                skipped += 1

    print(f"✅ buggy added: {made}, skipped: {skipped}")
    return out


def make_buggy_item_safe(item):
    """
    buggy 생성 중 오류/검증 실패가 나도 전체 파이프라인이 죽지 않게 안전 처리.
    """
    try:
        return make_buggy_item(item)
    except Exception:
        return None


def make_buggy_item(item):
    fixed_code = item["pl"]
    tests = item["tests"]

    # 원본이 tests를 통과하지 못하면 buggy 만들 의미가 없음 (노이즈 방지)
    if not run_static_tests(fixed_code, tests):
        raise ValueError("Original code does not pass tests; skip buggy generation")

    buggy_code = make_buggy_code(fixed_code, tests)

    # 실패 보장
    if run_static_tests(buggy_code, tests):
        raise ValueError("Buggy code unexpectedly passes tests")

    new_item = dict(item)
    new_item["pl"] = buggy_code
    new_item["fixed_code"] = fixed_code
    new_item["is_buggy"] = True
    return new_item


def make_buggy_code(pl, tests):
    # (1) export 제거: 항상 확실
    if "must_contain:export default Example" in tests and "export default Example" in pl:
        return pl.replace("export default Example", "", 1)

    # (2) JSX must_contain 중 하나 깨기
    jsx_tests = [t for t in tests if t.startswith("must_contain:<")]
    if not jsx_tests:
        raise ValueError("No JSX-based tests found")

    target = random.choice(jsx_tests)
    tag = target.replace("must_contain:<", "")  # e.g. AccordionTrigger

    # A) typo: <Tag 를 <Ta(마지막글자 제거)로
    if f"<{tag}" in pl:
        typo_tag = tag[:-1] if len(tag) > 3 else (tag + "x")
        return pl.replace(f"<{tag}", f"<{typo_tag}", 1)

    # B) remove component block
    pattern = re.compile(rf"<{tag}\b[\s\S]*?</{tag}>", re.MULTILINE)
    buggy, n = re.subn(pattern, "", pl, count=1)
    if n > 0:
        return buggy

    raise RuntimeError("Failed to generate buggy code")


# ----------------------------
# 4) train+val / test 분할 코드
# ----------------------------
def split_train_test_by_component(data, test_ratio=0.1):    
    groups = defaultdict(list)
    for item in data:
        ct = item.get("component_type", "__MISSING__")
        groups[ct].append(item)

    train, test = [], []
    train_only = 0

    for ct, items in groups.items():
        items = items[:]  # copy
        random.shuffle(items)
        m = len(items)

        if m == 1:
            train.append(items[0])
            train_only += 1
            continue

        target_test = int(round(m * test_ratio))
        k = max(1, min(m - 1, target_test))

        test.extend(items[:k])
        train.extend(items[k:])

    random.shuffle(train)
    random.shuffle(test)

    print("✅ split report")
    print(f"- total: {len(data)}")
    print(f"- train: {len(train)}")
    print(f"- test : {len(test)}")
    print(f"- component_types: {len(groups)}")
    print(f"- train-only component_types (size=1): {train_only}")
    
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    TRAIN_PATH.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    TEST_PATH.write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return train, test


# ----------------------------
# 5) Multi-task 테스트셋 생성 (test split 전체 사용)
# ----------------------------
def build_multitask_testset(test_data, output_path=MULTITASK_TEST_PATH, samples_per_task=NUMBER_OF_TEST):
    """
    test split 전체를 instruction_tuning.py에서 바로 쓸 수 있는 multi-task 테스트셋 포맷으로 변환한다.
    
    task list
    - code search
    - code repair
    - code summary
    - code generation
    - bug detection
    """
    if not test_data:
        raise ValueError("빈 데이터로는 multi-task 테스트셋을 만들 수 없습니다.")

    task_counts = Counter()
    test_samples = []
    for item in test_data:
        nl = (item.get("nl") or "").strip()
        pl = (item.get("pl") or "").strip()
        is_buggy = bool(item.get("is_buggy", False))
        fixed_code = (item.get("fixed_code") or "").strip()
        prefix = (item.get("prefix") or "").strip()

        choices = item.get("choices") or []
        answer = item.get("answer", None)

        tests = item.get("tests") or []


        if not nl:
            continue

        original_item = {
            "nl": nl,
            "pl": pl,
            "is_buggy": is_buggy,
            "has_fixed_code": bool(fixed_code),
            "tests": tests,
        }

        # ---------- 1) Code search ----------
        if isinstance(choices, list) and len(choices) >= 3 and answer in [0, 1, 2]:
            if task_counts["code_search"] < samples_per_task:
                input_text = TASK_PREFIXES['code_search'] + "\n"
                input_text += "Description:\n"
                input_text += nl + "\n\n"
                input_text += "Choices:\n"
                for idx, code_snippet in enumerate(choices):
                    input_text += f"Option {idx}:\n{code_snippet}\n\n"

                test_samples.append({
                    "task": "code_search",
                    "input": input_text,
                    "expected_output": str(answer),
                    "original_item": original_item,
                })
                task_counts["code_search"] += 1


        # ---------- 2) Code repair ----------
        if pl and is_buggy and fixed_code:
            if task_counts["code_repair"] < samples_per_task:
                test_samples.append({
                    "task": "code_repair",
                    "input": f"{TASK_PREFIXES['code_repair']} {pl}",
                    "expected_output": fixed_code,
                    "original_item": original_item,
                })
                task_counts["code_repair"] += 1

        # ---------- 3) Code summary ----------
        if pl:
            if task_counts["code_summary"] < samples_per_task:
                test_samples.append({
                    "task": "code_summary",
                    "input": f"{TASK_PREFIXES['code_summary']} {pl}",
                    "expected_output": nl,
                    "original_item": original_item,
                })
                task_counts["code_summary"] += 1

        # ---------- 4) Code generation ----------

        if pl:
            if task_counts["code_generation"] < samples_per_task:
                input_text = f"{TASK_PREFIXES['code_generation']} {nl}"
                if prefix:
                    input_text += f"\nFunction signature: {prefix}"
                test_samples.append({
                    "task": "code_generation",
                    "input": input_text,
                    "expected_output": pl,
                    "original_item": original_item,
                })
                task_counts["code_generation"] += 1

        # ---------- 5) Bug detection ----------
        if pl:
            if task_counts["bug_detection"] < samples_per_task:
                bug_label = "True" if is_buggy else "False"
                test_samples.append({
                    "task": "bug_detection",
                    "input": f"{TASK_PREFIXES['bug_detection']} {pl}",
                    "expected_output": bug_label,
                    "original_item": original_item,
                })
                task_counts["bug_detection"] += 1


    counter = Counter(s["task"] for s in test_samples)
    metadata = {
        "source": "patterns_test split (full)",
        "num_original_items": len(test_data),
        "num_test_samples": len(test_samples),
        "task_distribution": dict(counter),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "metadata": metadata,
        "test_samples": test_samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ multi-task test set saved to {output_path}")
    for t, c in counter.items():
        print(f"- {t}: {c}")


# ----------------------------
# main
# ----------------------------
async def main():
    # 1️⃣ 원본 dataset이 없으면 생성
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        urls = await ensure_urls_file()
        print("4) dataset.json 생성 중...")
        written = await build_dataset(urls)
        print(f"✅ done. written={written} -> {DATA_PATH}")
    else:
        print(f"✅ DATA_PATH already exists: {DATA_PATH}")

    # 2️⃣ 전처리 + buggy 생성 (항상 실행)
    processed = preprocessing_data(ratio=1.0)

    train, test = split_train_test_by_component(processed, test_ratio=0.1)

    # multi-task 테스트셋 생성 (test split 전체 사용)
    build_multitask_testset(test)


if __name__ == "__main__":
    asyncio.run(main())
