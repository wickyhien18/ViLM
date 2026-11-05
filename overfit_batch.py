# overfit_batch.py - việc 6: kiểm tra mặt nạ nhân quả và cho mô hình học thuộc 1 batch
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPTModel
from dataset import load_tokens, get_train_batch

tok = Tokenizer.from_file("tokenizer.json")
V = tok.get_vocab_size()
cfg = {
    "vocab_size": V,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.0,   # tắt dropout để học thuộc được hoàn toàn; sẽ bật lại ở việc 7
    "qkv_bias": False,
    "tie_weights": True,
}

torch.manual_seed(123)
model = GPTModel(cfg)
train = load_tokens("data/train.bin")
x, y = get_train_batch(train, 4)

# Phép thử 1: mặt nạ nhân quả. Đổi token từ vị trí k trở đi.
model.eval()
k = 40
x2 = x.clone()
x2[:, k:] = (x2[:, k:] + 1) % V
with torch.no_grad():
    a, b = model(x), model(x2)
print(f"trước vị trí {k} KHÔNG đổi (phải True):", torch.allclose(a[:, :k], b[:, :k], atol=1e-4))
print(f"từ vị trí {k} CÓ đổi (phải True):", not torch.allclose(a[:, k:], b[:, k:], atol=1e-4))

# Phép thử 2: học thuộc đúng một batch
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
for step in range(301):
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 25 == 0:
        print(f"bước {step:3d} | loss {loss.item():.4f}")

model.eval()
with torch.no_grad():
    acc = (model(x).argmax(-1) == y).float().mean().item()
print(f"đoán đúng token kế tiếp trên batch: {acc:.1%}")