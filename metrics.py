# metrics.py - đo điểm test A, B, C bằng cách sinh câu theo lô (nhanh hơn score.py)
# Dùng chung cho train.py. Chạy riêng: uv run python metrics.py --ckpt checkpoints/ckpt_last.pt
import json
import re
import unicodedata
from collections import defaultdict

import torch


def norm(s):
    return unicodedata.normalize("NFC", s).lower().strip()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tests(data_dir="data"):
    return {k: load_jsonl(f"{data_dir}/test_{k}.jsonl") for k in "ABC"}


def has_value(answer, value):
    # khớp theo ranh giới từ, giống score.py
    return re.search(rf"(?<!\w){re.escape(norm(value))}(?!\w)", norm(answer)) is not None


# Dữ liệu train có 4 cách từ chối, mỗi cách khoảng 25%: "Mình chưa học cái này.", "Mình không biết.",
# "Mình chưa biết câu này.", "Mình chưa biết." (bản cũ chỉ nhận hai cách có "chưa biết" nên đếm thiếu)
REFUSAL_PHRASES = ("chưa biết", "không biết", "chưa học")


def is_refusal(answer, refusals=None):
    # tham số refusals giữ lại để các script cũ gọi không bị lỗi; không còn dùng
    a = norm(answer)
    return any(p in a for p in REFUSAL_PHRASES)


@torch.no_grad()
def generate_many(model, tok, cfg, questions, device, max_new=40):
    """Chọn token xác suất cao nhất, dừng ở </s>. Gom các prompt cùng độ dài
    thành một lô để không cần đệm (padding): mô hình dùng embedding vị trí tuyệt đối."""
    bos, eos = tok.token_to_id("<s>"), tok.token_to_id("</s>")
    ctx = cfg["context_length"]
    prompts = [[bos] + tok.encode(f"Hỏi: {q}\nĐáp:").ids for q in questions]
    by_len = defaultdict(list)
    for i, pr in enumerate(prompts):
        by_len[len(pr)].append(i)

    answers = [""] * len(questions)
    was_training = model.training
    model.eval()
    for idxs in by_len.values():
        x = torch.tensor([prompts[i] for i in idxs], device=device)
        gen = []
        for _ in range(max_new):
            nxt = model(x[:, -ctx:])[:, -1].argmax(-1)
            gen.append(nxt)
            x = torch.cat([x, nxt[:, None]], dim=1)
            if (torch.stack(gen, 1) == eos).any(1).all():
                break
        for row, i in zip(torch.stack(gen, 1).tolist(), idxs):
            if eos in row:
                row = row[:row.index(eos)]
            answers[i] = tok.decode(row).strip()
    if was_training:
        model.train()
    return answers


def score_model(model, tok, cfg, tests, device, max_new=40):
    refusals = {norm(r["a"]) for r in tests["C"]}
    ans = {k: generate_many(model, tok, cfg, [r["q"] for r in tests[k]], device, max_new)
           for k in "ABC"}

    a_ok = sum(has_value(g, r["value"]) for g, r in zip(ans["A"], tests["A"]))

    b = {"correct": 0, "refuse": 0, "invent": 0, "n": 0}
    for g, r in zip(ans["B"], tests["B"]):
        if r["value_seen_in_train"]:
            continue  # chỉ tính dòng unseen
        b["n"] += 1
        if has_value(g, r["value"]):
            b["correct"] += 1
        elif is_refusal(g, refusals):
            b["refuse"] += 1
        else:
            b["invent"] += 1

    c_ok = sum(is_refusal(g, refusals) for g in ans["C"])
    return {
        "A": a_ok, "A_n": len(tests["A"]),
        "B_correct": b["correct"], "B_refuse": b["refuse"], "B_invent": b["invent"], "B_n": b["n"],
        "C": c_ok, "C_n": len(tests["C"]),
    }


def format_score(s):
    return (f"A {s['A']}/{s['A_n']} ({100 * s['A'] / s['A_n']:.1f}%) | "
            f"B-unseen: đúng {s['B_correct']}, từ chối {s['B_refuse']}, bịa {s['B_invent']} (trên {s['B_n']}) | "
            f"C {s['C']}/{s['C_n']}")


if __name__ == "__main__":
    import argparse
    import time
    from tokenizers import Tokenizer
    from model import GPTModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file("tokenizer.json")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = GPTModel(ck["cfg"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    t0 = time.time()
    s = score_model(model, tok, ck["cfg"], load_tests(), device, args.max_new)
    print(f"{args.ckpt} | bước {ck['step']} | {time.time() - t0:.1f}s")
    print(format_score(s))
