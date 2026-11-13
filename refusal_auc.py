# refusal_auc.py - chẩn đoán "núm vặn từ chối"
# Mọi cách từ chối đều mở đầu bằng "Mình". Ta lấy xác suất mô hình chọn token đó ngay sau "Đáp:" làm điểm từ chối,
# rồi xem điểm này có TÁCH được câu nên từ chối (D, B-unseen, C) khỏi câu nên trả lời (A) hay không.
# AUC = xác suất một câu nên từ chối có điểm cao hơn một câu nên trả lời (1,0 là tách hoàn hảo, 0,5 là ngẫu nhiên).
# Chạy: python refusal_auc.py --ckpt ckpt1.pt ckpt2.pt
import argparse
from collections import defaultdict

THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


def auc(pos, neg):
    """Mann-Whitney: P(điểm dương > điểm âm), hòa tính 0,5."""
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def rate_above(scores, tau):
    return sum(s > tau for s in scores) / len(scores)


def main():
    import torch
    from tokenizers import Tokenizer
    from metrics import load_tests
    from model import GPTModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file("tokenizer.json")
    BOS = tok.token_to_id("<s>")
    tests = load_tests()

    # token mở đầu của mọi cách từ chối: token ngay sau tiền tố "Hỏi: ...\nĐáp:"
    prefix = tok.encode("Hỏi: x\nĐáp:").ids
    starts = set()
    for r in ("Mình chưa học cái này.", "Mình không biết.", "Mình chưa biết câu này.", "Mình chưa biết."):
        full = tok.encode(f"Hỏi: x\nĐáp: {r}").ids
        assert full[:len(prefix)] == prefix, "khuôn tiền tố không khớp"
        starts.add(full[len(prefix)])
    assert len(starts) == 1, f"các cách từ chối mở đầu bằng token khác nhau: {starts}"
    REFUSE_TOKEN = starts.pop()
    print("token mở đầu câu từ chối:", repr(tok.decode([REFUSE_TOKEN])), "| id", REFUSE_TOKEN)

    @torch.no_grad()
    def refusal_scores(model, cfg, questions):
        prompts = [[BOS] + tok.encode(f"Hỏi: {q}\nĐáp:").ids for q in questions]
        by_len = defaultdict(list)
        for i, pr in enumerate(prompts):
            by_len[len(pr)].append(i)
        out = [0.0] * len(prompts)
        for idxs in by_len.values():
            x = torch.tensor([prompts[i] for i in idxs], device=device)
            probs = torch.softmax(model(x[:, -cfg["context_length"]:])[:, -1].float(), dim=-1)[:, REFUSE_TOKEN]
            for i, v in zip(idxs, probs.tolist()):
                out[i] = v
        return out

    for path in args.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ck["cfg"]
        model = GPTModel(cfg)
        model.load_state_dict(ck["model"])
        model.to(device).eval()

        groups = {
            "A (nên trả lời)": [r["q"] for r in tests["A"]],
            "B-unseen (nên từ chối)": [r["q"] for r in tests["B"] if not r["value_seen_in_train"]],
            "C (nên từ chối)": [r["q"] for r in tests["C"]],
        }
        if "D" in tests:
            groups["D (nên từ chối)"] = [r["q"] for r in tests["D"]]
        scores = {name: refusal_scores(model, cfg, qs) for name, qs in groups.items()}

        print(f"\n===== {path} | bước {ck['step']} =====")
        a_name = "A (nên trả lời)"
        for name, sc in scores.items():
            line = f"{name:<24} điểm từ chối trung bình {sum(sc) / len(sc):.3f}"
            if name != a_name:
                line += f" | AUC so với A: {auc(sc, scores[a_name]):.3f}"
            print(line)
        print("\nNếu từ chối khi điểm > ngưỡng (tỉ lệ câu bị từ chối):")
        header = f"{'ngưỡng':>8} | " + " | ".join(f"{n.split(' (')[0]:>9}" for n in scores)
        print(header)
        for tau in THRESHOLDS:
            print(f"{tau:>8.2f} | " + " | ".join(f"{100 * rate_above(sc, tau):>8.1f}%" for sc in scores.values()))
        print("(cột A là tỉ lệ từ chối NHẦM, nên càng thấp càng tốt; các cột còn lại càng cao càng tốt)")


if __name__ == "__main__":
    main()
