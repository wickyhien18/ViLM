# score.py (v2) - chấm A, B, C chi tiết cho một hoặc nhiều checkpoint
# Dùng chung bộ chấm với metrics.py (sinh câu theo lô), nên kết quả khớp với điểm in trong lúc huấn luyện.
import argparse
import json
import math
import os
from collections import defaultdict

import torch
from tokenizers import Tokenizer

from metrics import generate_many, has_value, is_refusal, load_tests, norm
from model import GPTModel

p = argparse.ArgumentParser()
p.add_argument("--ckpt", nargs="+", default=["checkpoints/ckpt_last.pt"])
p.add_argument("--max-new", type=int, default=40)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_file("tokenizer.json")
tests = load_tests()
refusals = {norm(r["a"]) for r in tests["C"]}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f}% ({k}/{n}) | 95%: {lo:.1f} đến {hi:.1f}"


os.makedirs("logs", exist_ok=True)
for path in args.ckpt:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    model = GPTModel(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    step = ck["step"]
    print(f"\n===== {path} | bước {step} =====")

    ans = {k: generate_many(model, tok, cfg, [r["q"] for r in tests[k]], device, args.max_new)
           for k in "ABC"}
    rows = []

    # test A: sự thật đã học, cách hỏi mới
    ka, by_attr, kinds = 0, defaultdict(lambda: [0, 0]), defaultdict(int)
    for g, r in zip(ans["A"], tests["A"]):
        ok = has_value(g, r["value"])
        ka += ok
        by_attr[r["attribute"]][0] += ok
        by_attr[r["attribute"]][1] += 1
        kind = "đúng" if ok else ("từ chối nhầm" if is_refusal(g, refusals) else "trả lời sai")
        kinds[kind] += 1
        rows.append({"set": "A", "q": r["q"], "attribute": r["attribute"],
                     "expected": r["value"], "got": g, "kind": kind})
    print("test A       :", fmt(ka, len(tests["A"])))
    for attr, (k, n) in sorted(by_attr.items()):
        print(f"   {attr:<12} {100 * k / n:5.1f}% ({k}/{n})")
    print(f"   câu sai gồm: từ chối nhầm {kinds['từ chối nhầm']} | trả lời sai {kinds['trả lời sai']}")

    # test B: sự thật chưa từng học; chia thành đúng, từ chối, bịa
    cnt = {True: defaultdict(int), False: defaultdict(int)}   # khóa: value_seen_in_train
    invented = []
    for g, r in zip(ans["B"], tests["B"]):
        seen = r["value_seen_in_train"]
        if has_value(g, r["value"]):
            kind = "đúng"
        elif is_refusal(g, refusals):
            kind = "từ chối"
        else:
            kind = "bịa"
            if not seen and len(invented) < 5:
                invented.append((r["q"], r["value"], g))
        cnt[seen][kind] += 1
        cnt[seen]["n"] += 1
        rows.append({"set": "B", "seen": seen, "q": r["q"], "expected": r["value"],
                     "got": g, "kind": kind})
    u, s = cnt[False], cnt[True]
    print("test B unseen:", fmt(u["đúng"], u["n"]),
          f"| từ chối {u['từ chối']} | bịa {u['bịa']}")
    print("test B seen  :", fmt(s["đúng"], s["n"]), "(chỉ để tham khảo, đã bị rò gián tiếp)")

    # test C: câu ngoài phạm vi, phải từ chối
    kc, c_fail = 0, []
    for g, r in zip(ans["C"], tests["C"]):
        ok = is_refusal(g, refusals)
        kc += ok
        if not ok:
            c_fail.append((r["q"], g))
        rows.append({"set": "C", "q": r["q"], "got": g, "ok": ok})
    print("test C       :", fmt(kc, len(tests["C"])))

    print("\n-- 5 câu A sai đầu tiên --")
    for r in [x for x in rows if x["set"] == "A" and x["kind"] != "đúng"][:5]:
        print(f"  Hỏi: {r['q']} | cần '{r['expected']}' | mô hình: {r['got']}  [{r['kind']}]")
    print("-- 5 câu B-unseen mà mô hình bịa --")
    for q, v, g in invented:
        print(f"  Hỏi: {q} | cần '{v}' | mô hình: {g}")
    print("-- câu C không từ chối --")
    for q, g in c_fail[:5]:
        print(f"  Hỏi: {q} | mô hình: {g}")

    name = os.path.splitext(os.path.basename(path))[0]
    with open(f"logs/score_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
