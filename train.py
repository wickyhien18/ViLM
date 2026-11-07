# train.py (v3) - vòng lặp huấn luyện
# v2: đo điểm A/B/C định kỳ, lưu checkpoint theo điểm A, lưu mốc định kỳ, hỗ trợ fp16 trên GPU.
# v3: cấu hình mô hình lấy từ config.py, thêm --seed, in tên GPU.
import argparse
import csv
import math
import os
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import make_cfg
from dataset import load_tokens, get_train_batch, val_batches
from metrics import load_tests, score_model, format_score
from model import GPTModel

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=300)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval-every", type=int, default=50, help="đo val loss mỗi N bước")
p.add_argument("--score-every", type=int, default=0, help="đo điểm A/B/C mỗi N bước; 0 là không đo")
p.add_argument("--save-every", type=int, default=0, help="lưu mốc ckpt_stepN.pt mỗi N bước; 0 là không lưu")
p.add_argument("--amp", action="store_true", help="fp16 (chỉ có tác dụng trên GPU)")
p.add_argument("--out", default="checkpoints")
p.add_argument("--log", default="logs/train_log.csv")
p.add_argument("--seed", type=int, default=123, help="seed cho khởi tạo, thứ tự batch và dropout")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = args.amp and device == "cuda"
print(f"thiết bị: {device} | fp16: {use_amp} | seed: {args.seed}")
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

tok = Tokenizer.from_file("tokenizer.json")
V = tok.get_vocab_size()
cfg = make_cfg(V)

torch.manual_seed(args.seed)
model = GPTModel(cfg).to(device)
train = load_tokens("data/train.bin")
val = load_tokens("data/val.bin")
tests = load_tests()

# weight decay chỉ áp cho ma trận (ndim >= 2), không áp cho bias và chuẩn hóa
decay = [q for q in model.parameters() if q.ndim >= 2]
no_decay = [q for q in model.parameters() if q.ndim < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95),
)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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


def save(name, step, vl):
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, "val_loss": vl, "seed": args.seed},
               f"{args.out}/{name}")


os.makedirs(args.out, exist_ok=True)
os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
log = open(args.log, "w", newline="")
writer = csv.writer(log)
writer.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "seconds",
                 "A", "B_correct", "B_refuse", "B_invent", "C"])

best_val, best_a = float("inf"), -1
run_loss, run_n = 0.0, 0
gn = float("nan")
t0 = time.time()
model.train()
for step in range(args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)

    do_score = args.score_every > 0 and (step % args.score_every == 0 or step == args.steps)
    do_eval = do_score or step % args.eval_every == 0 or step == args.steps
    if do_eval:
        vl = evaluate()
        tl = run_loss / run_n if run_n else float("nan")
        run_loss, run_n = 0.0, 0
        s = score_model(model, tok, cfg, tests, device) if do_score else None
        elapsed = time.time() - t0
        line = f"bước {step:5d} | train {tl:.4f} | val {vl:.4f} | lr {lr_at(step):.2e} | {elapsed:.0f}s"
        if s:
            line += "\n           " + format_score(s)
        print(line, flush=True)
        writer.writerow([step, tl, vl, lr_at(step), round(gn, 3), round(elapsed, 1)]
                        + ([s["A"], s["B_correct"], s["B_refuse"], s["B_invent"], s["C"]] if s else [""] * 5))
        log.flush()
        if vl < best_val:
            best_val = vl
            save("ckpt_best.pt", step, vl)          # theo val loss thấp nhất (như v1)
        if s and s["A"] > best_a:
            best_a = s["A"]
            save("ckpt_bestA.pt", step, vl)         # theo điểm A cao nhất
        if args.save_every and step > 0 and step % args.save_every == 0:
            save(f"ckpt_step{step}.pt", step, vl)

    if step == args.steps:
        break

    x, y = get_train_batch(train, args.batch, device=device)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        logits = model(x)
    loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    scaler.step(opt)
    scaler.update()
    run_loss += loss.item()
    run_n += 1
    if step < 3:
        print(f"  bước {step}: loss {loss.item():.3f} | grad_norm {gn:.2f} | {time.time() - t0:.2f}s tính đến giờ",
              flush=True)

save("ckpt_last.pt", args.steps, vl)
print(f"xong: val tốt nhất {best_val:.4f} | A tốt nhất {best_a} | tổng {time.time() - t0:.0f}s")
