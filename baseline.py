# baseline.py - baseline tra bảng (không có mạng nơ-ron), chấm bằng cùng bộ chấm với mô hình
# Chạy ở thư mục dự án: python baseline.py
# Cách làm: (1) dựng bảng (chủ thể, thuộc tính) -> giá trị từ các sự thật CÓ trong train;
# (2) tìm chủ thể trong câu hỏi bằng khớp chuỗi; (3) đoán thuộc tính bằng Naive Bayes trên các từ
# của câu hỏi train (đã bỏ từ của chủ thể); (4) tra bảng, không thấy thì từ chối.
import csv
import json
import math
import re
from collections import Counter, defaultdict

from metrics import has_value, is_refusal, load_tests, norm

REFUSAL = "Mình chưa biết."


def words(s):
    return re.findall(r"\w+", norm(s))


CLASSIFIERS = {"con", "quả", "cái", "đôi"}   # giống gen_data.py: bỏ loại từ để khớp cả "mèo" lẫn "con mèo"


def core(subject):
    parts = norm(subject).split(" ")
    if parts[0] in CLASSIFIERS and len(parts) > 1:
        return " ".join(parts[1:])
    return " ".join(parts)


# ---- dựng bảng từ các sự thật có trong train ----
facts = {}
with open("facts.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        facts[r["id"]] = r
train = [json.loads(line) for line in open("data/train.jsonl", encoding="utf-8") if line.strip()]
train_fact_rows = [r for r in train if r["kind"] == "fact" and r["fact_id"] in facts]
table = {}
for fid in {r["fact_id"] for r in train_fact_rows}:
    fr = facts[fid]
    table[(core(fr["subject"]), fr["attribute"])] = fr["value"]
subjects = sorted({s for s, _ in table}, key=len, reverse=True)   # chủ thể dài khớp trước

# ---- Naive Bayes đoán thuộc tính từ câu hỏi (đã bỏ từ của chủ thể) ----
counts = defaultdict(Counter)
for r in train_fact_rows:
    sub_words = set(words(core(facts[r["fact_id"]]["subject"])))
    counts[r["attribute"]].update(w for w in words(r["q"]) if w not in sub_words)
vocab = {w for c in counts.values() for w in c}
total = {a: sum(c.values()) for a, c in counts.items()}


# Từ khóa lấy từ chính TÊN thuộc tính (không nhìn test); chỉ tin khi đúng một thuộc tính khớp,
# còn lại dùng Naive Bayes. "nên" và "dùng" tách "nên làm gì" khỏi "công dụng" (cùng có "làm gì").
CUES = [
    ("nên làm gì", ["nên"]),
    ("công dụng", ["dùng", "công dụng"]),
    ("tiếng kêu", ["kêu"]),
    ("ăn gì", ["ăn"]),
    ("số chân", ["chân"]),
    ("sống ở đâu", ["sống", "ở đâu", "nhà", "nơi"]),
    ("màu sắc", ["màu"]),
    ("vị", ["vị"]),
]


def attribute_by_cue(qn):
    hits = [a for a, cues in CUES if a in counts
            and any(re.search(rf"(?<!\w){re.escape(c)}(?!\w)", qn) for c in cues)]
    return hits[0] if len(hits) == 1 else None


def attribute_of(q, sub_words):
    ws = [w for w in words(q) if w not in sub_words and w in vocab]
    best, best_score = None, -1e18
    for a in counts:
        score = sum(math.log((counts[a][w] + 1) / (total[a] + len(vocab))) for w in ws)
        if score > best_score:
            best, best_score = a, score
    return best


def answer(q):
    qn = norm(q)
    sub = next((s for s in subjects if re.search(rf"(?<!\w){re.escape(s)}(?!\w)", qn)), None)
    if sub is None:
        return REFUSAL
    attr = attribute_by_cue(qn) or attribute_of(q, set(words(sub)))
    value = table.get((sub, attr))
    return f"{value}." if value else REFUSAL


# ---- chấm A, B, C ----
tests = load_tests()
print(f"bảng có {len(table)} sự thật từ {len(train_fact_rows)} cặp train; {len(subjects)} chủ thể")

ka, by_attr, kinds, wrong = 0, defaultdict(lambda: [0, 0]), defaultdict(int), []
for r in tests["A"]:
    g = answer(r["q"])
    ok = has_value(g, r["value"])
    ka += ok
    by_attr[r["attribute"]][0] += ok
    by_attr[r["attribute"]][1] += 1
    kind = "đúng" if ok else ("từ chối nhầm" if is_refusal(g) else "trả lời sai")
    kinds[kind] += 1
    if not ok and len(wrong) < 6:
        wrong.append((r["q"], r["value"], g, kind))
print(f"test A       : {100 * ka / len(tests['A']):.1f}% ({ka}/{len(tests['A'])})")
for attr, (k, n) in sorted(by_attr.items()):
    print(f"   {attr:<12} {100 * k / n:5.1f}% ({k}/{n})")
print(f"   câu sai gồm: từ chối nhầm {kinds['từ chối nhầm']} | trả lời sai {kinds['trả lời sai']}")

cnt = {True: defaultdict(int), False: defaultdict(int)}
for r in tests["B"]:
    g = answer(r["q"])
    kind = "đúng" if has_value(g, r["value"]) else ("từ chối" if is_refusal(g) else "bịa")
    cnt[r["value_seen_in_train"]][kind] += 1
    cnt[r["value_seen_in_train"]]["n"] += 1
u, s = cnt[False], cnt[True]
print(f"test B unseen: đúng {u['đúng']} | từ chối {u['từ chối']} | bịa {u['bịa']} (trên {u['n']})")
print(f"test B seen  : đúng {s['đúng']} | từ chối {s['từ chối']} | bịa {s['bịa']} (trên {s['n']})")

kc = sum(is_refusal(answer(r["q"])) for r in tests["C"])
print(f"test C       : {kc}/{len(tests['C'])}")

if "D" in tests:   # chỉ có khi dữ liệu sinh bằng gen_data.py --v2
    kd = sum(is_refusal(answer(r["q"])) for r in tests["D"])
    print(f"test D       : {kd}/{len(tests['D'])}")

print("\n-- vài câu A sai --")
for q, v, g, kind in wrong:
    print(f"  Hỏi: {q} | cần '{v}' | baseline: {g}  [{kind}]")
