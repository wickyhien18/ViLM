# summarize_seeds.py - tổng hợp log của nhiều lần chạy (nhiều seed): điểm cuối, độ ổn định, trung bình và độ lệch
# Dùng: python summarize_seeds.py run1/train_log.csv run2/train_log.csv run3/train_log.csv
import argparse
import csv
import statistics as st

p = argparse.ArgumentParser()
p.add_argument("logs", nargs="+", help="các file train_log.csv, mỗi file một lần chạy")
p.add_argument("--a-n", type=int, default=302, help="số câu test A")
p.add_argument("--b-n", type=int, default=80, help="số câu B-unseen")
p.add_argument("--c-n", type=int, default=15, help="số câu test C")
p.add_argument("--d-n", type=int, default=96, help="số câu test D (nếu log có cột D)")
p.add_argument("--from-step", type=int, default=8000, help="cửa sổ cuối dùng để xem độ ổn định")
args = p.parse_args()


def read(path):
    scored, grad = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["grad_norm"] not in ("", "nan"):
                grad.append(float(r["grad_norm"]))
            if r["A"] == "":
                continue
            scored.append({
                "step": int(r["step"]),
                "A": 100 * int(r["A"]) / args.a_n,
                "refuse": int(r["B_refuse"]),
                "invent": int(r["B_invent"]),
                "C": int(r["C"]),
                "A_refused": int(r["A_refused"]) if r.get("A_refused") not in (None, "") else None,
                "D": int(r["D"]) if r.get("D") not in (None, "") else None,
            })
    return scored, grad


def ms(values):
    if len(values) < 2:
        return f"{values[0]:.1f}" if values else "-"
    return f"{st.mean(values):.1f} ± {st.stdev(values):.1f} (từ {min(values):.1f} đến {max(values):.1f})"


finals = []
print(f"{'file':<46} {'bước':>6} {'A cuối':>8} {'từ chối':>8} {'bịa':>5} {'C':>6} | A trong cửa sổ cuối (thấp, tb, cao) | C thấp nhất")
for path in args.logs:
    scored, grad = read(path)
    if not scored:
        print(f"{path}: không có dòng nào có điểm A")
        continue
    last = scored[-1]
    win = [r for r in scored if r["step"] >= args.from_step]
    wa = [r["A"] for r in win]
    wc = [r["C"] for r in win]
    finals.append(last)
    print(f"{path[-46:]:<46} {last['step']:>6} {last['A']:>7.1f}% {last['refuse']:>8} {last['invent']:>5} "
          f"{last['C']:>3}/{args.c_n} | {min(wa):.1f}, {st.mean(wa):.1f}, {max(wa):.1f} | {min(wc)}/{args.c_n}")
    gtxt = f"{len(grad)} giá trị, đầu tiên {grad[0]:.2f}" if grad else "cột trống"
    extra = ""
    if last["A_refused"] is not None:
        extra += f" | A từ chối nhầm {last['A_refused']}/{args.a_n}"
    if last["D"] is not None:
        extra += f" | D {last['D']}/{args.d_n}"
    print(f"{'':<46} grad_norm trong log: {gtxt}{extra}")

if len(finals) >= 2:
    print(f"\nTổng hợp trên {len(finals)} lần chạy (điểm ở mốc cuối của mỗi lần):")
    print("  A (%)            :", ms([r["A"] for r in finals]))
    print(f"  B-unseen bịa (/{args.b_n}):", ms([float(r["invent"]) for r in finals]))
    print(f"  B-unseen từ chối  :", ms([float(r["refuse"]) for r in finals]))
    print(f"  C (/{args.c_n})          :", ms([float(r["C"]) for r in finals]))
    if all(r["A_refused"] is not None for r in finals):
        print(f"  A từ chối nhầm (/{args.a_n}):", ms([float(r["A_refused"]) for r in finals]))
    if all(r["D"] is not None for r in finals):
        print(f"  D (/{args.d_n})          :", ms([float(r["D"]) for r in finals]))
    print("\nCách đọc: khi so hai cấu hình khác nhau, chỉ tin chênh lệch của trung bình nếu nó lớn hơn rõ rệt")
    print("độ lệch ở trên (tham khảo: ít nhất khoảng 2 lần). Với 3 lần chạy thì độ lệch bản thân cũng chỉ là ước lượng thô.")
else:
    print("\nCần ít nhất 2 lần chạy để tính độ lệch.")
