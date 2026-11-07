# generate.py - việc 8: sinh câu trả lời từ checkpoint
import argparse
import torch
from tokenizers import Tokenizer
from model import GPTModel

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="checkpoints/ckpt_best.pt")
p.add_argument("--q", action="append", help="câu hỏi; dùng nhiều lần được")
p.add_argument("--max-new", type=int, default=40)
args = p.parse_args()

tok = Tokenizer.from_file("tokenizer.json")
BOS, EOS = tok.token_to_id("<s>"), tok.token_to_id("</s>")

ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
cfg = ck["cfg"]
model = GPTModel(cfg)
model.load_state_dict(ck["model"])
model.eval()
print(f"checkpoint: {args.ckpt} | bước {ck['step']} | val {ck['val_loss']:.4f}")

# kiểm tra khuôn: prompt phải là phần đầu của một cặp thật
q0 = "Con mèo kêu như thế nào?"
prompt0 = tok.encode(f"Hỏi: {q0}\nĐáp:").ids
full0 = tok.encode(f"Hỏi: {q0}\nĐáp: Con mèo kêu meo meo.").ids
print("khuôn prompt khớp huấn luyện:", "ĐÚNG" if full0[:len(prompt0)] == prompt0 else "SAI")


@torch.no_grad()
def answer(question):
    ids = [BOS] + tok.encode(f"Hỏi: {question}\nĐáp:").ids
    out, stopped = [], False
    for _ in range(args.max_new):
        x = torch.tensor([ids[-cfg["context_length"]:]])
        nxt = model(x)[0, -1].argmax().item()
        if nxt == EOS:
            stopped = True
            break
        ids.append(nxt)
        out.append(nxt)
    return tok.decode(out).strip(), stopped


questions = args.q or [
    "Cái mũ dùng để làm gì?",
    "Hãy cho biết con chuột kêu như thế nào.",
    "Con ngựa có mấy chân vậy?",
    "Con mèo kêu như thế nào?",
    "Hôm nay giá vàng là bao nhiêu?",   # ngoài phạm vi, chỉ để thăm dò
]
for q in questions:
    a, stopped = answer(q)
    print(f"\nHỏi: {q}\nĐáp: {a}   [{'dừng đúng' if stopped else f'bị cắt ở {args.max_new} token'}]")