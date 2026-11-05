# Kiểm tra mô hình: số tham số, hình dạng đầu ra, độ dài các cặp và loss ban đầu.
import numpy as np
import torch
from tokenizers import Tokenizer
from model import GPTModel

tok = Tokenizer.from_file("tokenizer.json")
V = tok.get_vocab_size()
cfg = {
    "vocab_size": V,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "tie_weights": True,
}

torch.manual_seed(123)
model = GPTModel(cfg)
n_params = sum(p.numel() for p in model.parameters())
print("vocab_size:", V, "| số tham số:", n_params)

data = np.fromfile("data/train.bin", dtype=np.uint16).astype(np.int64)

# độ dài mỗi cặp: từ token <s> tới hết token </s>
bos = tok.token_to_id("<s>")
starts = np.where(data == bos)[0]
lengths = np.diff(np.append(starts, len(data)))
print("số cặp:", len(lengths),
      "| dài nhất:", int(lengths.max()),
      "| trung bình:", round(float(lengths.mean()), 1),
      "| số cặp dài hơn ngữ cảnh", cfg["context_length"], ":", int((lengths > cfg["context_length"]).sum()))

ctx = cfg["context_length"]
x = torch.from_numpy(data[:ctx]).unsqueeze(0)
y = torch.from_numpy(data[1:ctx + 1]).unsqueeze(0)
model.eval()
with torch.no_grad():
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
print("hình dạng logits:", tuple(logits.shape), "| mong đợi:", (1, ctx, V))
print("loss ban đầu:", round(loss.item(), 3), "| ln(vocab) =", round(float(np.log(V)), 3))
