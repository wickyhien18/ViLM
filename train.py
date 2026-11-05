# train.py - việc 7: vòng lặp huấn luyện đầy đủ
import argparse, csv, math, os, time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPTModel
from dataset import load_tokens, get_train_batch, val_batches

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=300)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval-every", type=int, default=50)
p.add_argument("--out", default="checkpoints")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_file("tokenizer.json")
V = tok.get_vocab_size()
cfg = {
    "vocab_size": V, "context_length": 64, "emb_dim": 128, "n_heads": 4,
    "n_layers": 4, "drop_rate": 0.1, "qkv_bias": False, "tie_weights": True,
}

torch.manual_seed(123)
model = GPTModel(cfg).to(device)
train = load_tokens("data/train.bin")
val = load_tokens("data/val.bin")

# weight decay chỉ áp cho ma trận (ndim >= 2), không áp cho bias và chuẩn hóa
decay = [q for q in model.parameters() if q.ndim >= 2]
no_decay = [q for q in model.parameters() if q.ndim < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95),
)

warmup = max(10, args.steps // 20)
min_lr = args.lr * 0.1


def lr_at(step):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    t = (step - warmup) / max(1, args.steps - warmup)
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate():
    model.eval()
    total, count = 0.0, 0
    for x, y in val_batches(val, 16, device=device):
        logits = model(x)
        total += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        count += y.numel()
    model.train()
    return total / count


os.makedirs(args.out, exist_ok=True)
os.makedirs("logs", exist_ok=True)
log = open("logs/train_log.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "seconds"])

best = float("inf")
run_loss, run_n = 0.0, 0
t0 = time.time()
model.train()
for step in range(args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    if step % args.eval_every == 0 or step == args.steps:
        vl = evaluate()
        tl = run_loss / run_n if run_n else float("nan")
        run_loss, run_n = 0.0, 0
        elapsed = time.time() - t0
        print(f"bước {step:5d} | train {tl:.4f} | val {vl:.4f} | lr {lr_at(step):.2e} | {elapsed:.0f}s")
        writer.writerow([step, tl, vl, lr_at(step), "", round(elapsed, 1)])
        log.flush()
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, "val_loss": vl},
                       f"{args.out}/ckpt_best.pt")
    if step == args.steps:
        break
    x, y = get_train_batch(train, args.batch, device=device)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step()
    run_loss += loss.item()
    run_n += 1
    if step < 3:
        print(f"  bước {step}: loss {loss.item():.3f} | grad_norm {gn:.2f} | {time.time() - t0:.2f}s tính đến giờ")

torch.save({"model": model.state_dict(), "cfg": cfg, "step": args.steps, "val_loss": vl},
           f"{args.out}/ckpt_last.pt")
print(f"xong: val tốt nhất {best:.4f} | tổng {time.time() - t0:.0f}s")