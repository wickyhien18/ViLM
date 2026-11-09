#!/usr/bin/env python3
"""
Sinh dữ liệu hỏi–đáp tiếng Việt mức mầm non từ bảng sự thật (facts.csv) bằng template.

Nguyên tắc:
  - Đáp án đúng do bảng sự thật quyết định, không để LLM tự bịa kiến thức.
  - Tách tập kiểm tra TRƯỚC khi sinh train:
      test_A: sự thật đã học, nhưng hỏi bằng template CHƯA TỪNG thấy khi train
      test_B: sự thật bị GIỮ LẠI hoàn toàn (không có trong train)
      test_C: câu hỏi ngoài phạm vi (mong đợi: từ chối)
  - Chạy lại với cùng --seed sẽ cho đúng cùng dữ liệu.

Chạy:  python3 gen_data.py            (chỉ dùng thư viện chuẩn)
"""
import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CLASSIFIERS = {"con", "quả", "cái", "đôi"}   # bỏ loại từ để tạo cách gọi ngắn: "con mèo" -> "mèo"
N_HELDOUT_Q = 2                              # 2 template cuối của mỗi thuộc tính dành cho test_A

REFUSALS = [
    "Mình chưa biết.",
    "Mình không biết.",
    "Mình chưa học cái này.",
    "Mình chưa biết câu này.",
]

# Mỗi thuộc tính: 8 template hỏi (2 cái cuối bị giữ lại cho test_A) và 4 template đáp.
# {S} = chủ thể, {V} = giá trị trong bảng sự thật.
TEMPLATES = {
    "tiếng kêu": {
        "q": [
            "{S} kêu như thế nào?",
            "{S} kêu ra sao?",
            "Tiếng kêu của {S} là gì?",
            "{S} kêu thế nào nhỉ?",
            "Bạn có biết {S} kêu thế nào không?",
            "Hãy cho biết {S} kêu như thế nào.",
            "Nghe thử xem, {S} kêu như thế nào?",
            "Khi kêu, {S} phát ra tiếng gì?",
        ],
        "a": [
            "{S} kêu {V}.",
            "Tiếng kêu của {S} là {V}.",
            "{S} kêu {V} đó.",
            "{V}! Đó là tiếng {S} kêu.",
        ],
    },
    "màu sắc": {
        "q": [
            "{S} có màu gì?",
            "{S} màu gì?",
            "Màu của {S} là gì?",
            "{S} có màu gì vậy?",
            "Bạn có biết {S} có màu gì không?",
            "Hãy cho biết {S} có màu gì.",
            "Nhìn xem, {S} có màu gì?",
            "{S} thường có màu nào?",
        ],
        "a": [
            "{S} có màu {V}.",
            "Màu của {S} là {V}.",
            "{S} màu {V}.",
            "{S} có màu {V} đó.",
        ],
    },
    "ăn gì": {
        "q": [
            "{S} ăn gì?",
            "{S} thích ăn gì?",
            "Thức ăn của {S} là gì?",
            "{S} thường ăn gì vậy?",
            "Bạn có biết {S} ăn gì không?",
            "Hãy cho biết {S} ăn gì.",
            "Bữa ăn của {S} có gì?",
            "Nói xem, {S} ăn gì nhỉ?",
        ],
        "a": [
            "{S} ăn {V}.",
            "{S} thích ăn {V}.",
            "Thức ăn của {S} là {V}.",
            "{S} ăn {V} đó.",
        ],
    },
    "số chân": {
        "q": [
            "{S} có mấy chân?",
            "{S} có bao nhiêu chân?",
            "Số chân của {S} là bao nhiêu?",
            "{S} có mấy chân vậy?",
            "Bạn có biết {S} có mấy chân không?",
            "Hãy đếm xem {S} có mấy chân.",
            "Nói xem, {S} có mấy cái chân nhỉ?",
            "{S} đi bằng mấy chân?",
        ],
        "a": [
            "{S} có {V}.",
            "{S} có {V} đó.",
            "Số chân của {S} là {V}.",
            "Đếm thử xem, {S} có {V}.",
        ],
    },
    "sống ở đâu": {
        "q": [
            "{S} sống ở đâu?",
            "Nhà của {S} ở đâu?",
            "{S} ở đâu vậy?",
            "Bạn có biết {S} sống ở đâu không?",
            "{S} thường sống ở chỗ nào?",
            "Hãy cho biết {S} sống ở đâu.",
            "Nói xem, {S} sống ở đâu nhỉ?",
            "Nơi ở của {S} là đâu?",
        ],
        "a": [
            "{S} sống {V}.",
            "{S} sống {V} đó.",
            "{S} thường sống {V}.",
            "Nhà của {S} nằm {V}.",
        ],
    },
    "vị": {
        "q": [
            "{S} có vị gì?",
            "{S} vị gì?",
            "Vị của {S} là gì?",
            "{S} có vị như thế nào?",
            "Bạn có biết {S} có vị gì không?",
            "Hãy cho biết {S} có vị gì.",
            "Nếm thử {S} thì thấy vị gì?",
            "Nói xem, {S} có vị gì nhỉ?",
        ],
        "a": [
            "{S} có vị {V}.",
            "Vị của {S} là {V}.",
            "{S} có vị {V} đó.",
            "Nếm thử sẽ thấy {S} có vị {V}.",
        ],
    },
    "công dụng": {
        "q": [
            "{S} dùng để làm gì?",
            "Dùng {S} để làm gì?",
            "Người ta dùng {S} để làm gì?",
            "{S} dùng làm gì vậy?",
            "Bạn có biết {S} dùng để làm gì không?",
            "Hãy cho biết {S} dùng để làm gì.",
            "Nói xem, {S} dùng để làm gì nhỉ?",
            "Mình dùng {S} để làm gì?",
        ],
        "a": [
            "{S} dùng để {V}.",
            "Mình dùng {S} để {V}.",
            "Người ta dùng {S} để {V}.",
            "{S} dùng để {V} đó.",
        ],
    },
    "nên làm gì": {
        "q": [
            "{S} thì nên làm gì?",
            "{S} thì mình nên làm gì?",
            "Khi {S} thì phải làm sao?",
            "Nếu {S} thì nên làm gì?",
            "Bạn có biết {S} thì nên làm gì không?",
            "{S} thì cần làm gì?",
            "Hãy cho biết {S} thì nên làm gì.",
            "Nói xem, {S} thì nên làm gì nhỉ?",
        ],
        "a": [
            "{S} thì nên {V}.",
            "{S} thì mình nên {V}.",
            "Khi {S}, nên {V}.",
            "{S} thì nên {V} đó.",
        ],
    },
}

# Câu hỏi ngoài phạm vi: tập train và tập test phải KHÔNG trùng nhau.
OOD_TRAIN = [
    "Thủ đô của nước Pháp là gì?", "Ai là tổng thống nước Mỹ?", "Hôm nay là thứ mấy?",
    "Giá vàng hôm nay là bao nhiêu?", "Python là ngôn ngữ gì?", "Mặt trăng cách Trái Đất bao xa?",
    "Ai đã phát minh ra bóng đèn?", "Con khủng long biến mất khi nào?",
    "Núi cao nhất thế giới là núi nào?", "Dòng sông dài nhất thế giới là sông nào?",
    "Trái Đất có bao nhiêu tuổi?", "Cách nấu phở bò như thế nào?",
    "Làm sao để đăng ký tài khoản ngân hàng?", "Cầu thủ nào ghi nhiều bàn nhất?",
    "Bộ phim nào hay nhất năm nay?", "Hà Nội có bao nhiêu quận?", "Ai viết truyện Kiều?",
    "Nước nào đông dân nhất?", "Điện thoại thông minh hoạt động như thế nào?",
    "Máy bay bay bằng cách nào?", "Cách sửa máy tính bị hỏng?",
    "Thuế thu nhập tính như thế nào?", "Vì sao có nhật thực?",
    "Lịch sử của nước Việt Nam bắt đầu khi nào?", "Con người đã lên Mặt Trăng chưa?",
    "Ai là người giàu nhất thế giới?", "Làm thế nào để học giỏi tiếng Anh?", "Bitcoin là gì?",
    "Thời tiết ngày mai ở Hà Nội thế nào?", "Ngày Quốc khánh nước ta là ngày nào?",
    "Máy giặt hoạt động như thế nào?", "Vitamin C có tác dụng gì?",
    "Phương trình bậc hai giải thế nào?", "Tại sao biển có sóng?", "Sao Hỏa có sự sống không?",
    "Ai là vua đầu tiên của nước ta?", "Cách đi từ Hà Nội vào Sài Gòn?",
    "Làm sao để viết một bài văn hay?", "Trí tuệ nhân tạo là gì?", "Hôm nay là ngày bao nhiêu?",
]
OOD_TEST = [
    "Thủ đô của nước Nhật là gì?", "Ai là chủ tịch nước hiện nay?", "Giá xăng hôm nay bao nhiêu?",
    "Java là ngôn ngữ gì?", "Sao Kim cách Mặt Trời bao xa?", "Ai phát minh ra điện thoại?",
    "Đỉnh Fansipan cao bao nhiêu mét?", "Cách nấu bún chả như thế nào?",
    "Cầu thủ nào nổi tiếng nhất Việt Nam?", "Thành phố Hồ Chí Minh có bao nhiêu quận?",
    "Làm sao để mở tài khoản chứng khoán?", "Tại sao có mưa đá?", "Tivi hoạt động như thế nào?",
    "Trí tuệ nhân tạo học như thế nào?", "Ai là tác giả của bài thơ nổi tiếng nhất nước ta?",
]


# ---- v2: câu từ chối viết theo mẫu hỏi QUEN, với chủ thể chưa có trong bảng sự thật ----
# Mục đích: câu ngoài phạm vi và câu trong phạm vi dùng cùng cách diễn đạt, nên cách hỏi không còn là
# dấu hiệu để từ chối; chỉ còn nội dung (chủ thể có trong bảng hay không).
# Mỗi nhóm có danh sách "train" và "test" KHÔNG trùng nhau; chủ thể nào đã có trong facts.csv bị loại tự động.
_ANIMALS = {
    "train": ["con voi", "con sư tử", "con hổ", "con hươu cao cổ", "con cá heo", "con rùa", "con thỏ",
              "con dê", "con cừu", "con sói", "con gấu", "con cáo", "con nai", "con lạc đà"],
    "test": ["con cá sấu", "con báo", "con tê giác", "con hà mã", "con đà điểu", "con công",
             "con chim cánh cụt", "con ếch", "con cua", "con ốc sên"],
}
_FRUITS = {
    "train": ["quả dưa hấu", "quả nho", "quả dâu tây", "quả xoài", "quả bưởi", "quả ổi", "quả cà tím",
              "quả mận", "quả lê", "quả đào"],
    "test": ["quả chanh", "quả cam", "quả dứa", "quả vải", "quả nhãn", "quả mít", "quả sầu riêng",
             "quả măng cụt"],
}
_TOOLS = {
    "train": ["cái bàn", "cái ghế", "cái bút", "cái cốc", "cái kéo", "cái chổi", "cái thìa", "cái đĩa",
              "cái nồi", "cái quạt", "cái gối", "cái khăn"],
    "test": ["cái gương", "cái lược", "cái chăn", "cái kính", "cái đồng hồ", "cái tủ", "cái dao", "cái bát"],
}
_SITUATIONS = {
    "train": ["đến ngày 8/3", "đến ngày 20/10", "đến Tết Trung thu", "trời nắng", "trời lạnh",
              "trời có sương mù", "trời gió to"],
    "test": ["đến ngày 20/11", "đến ngày 1/6", "trời có bão", "trời nóng", "trời rét đậm"],
}
POOLS = {
    "tiếng kêu": _ANIMALS, "ăn gì": _ANIMALS, "số chân": _ANIMALS, "sống ở đâu": _ANIMALS,
    "màu sắc": _FRUITS, "vị": _FRUITS, "công dụng": _TOOLS, "nên làm gì": _SITUATIONS,
}


def norm(s: str) -> str:
    """Chuẩn hóa Unicode NFC (bắt buộc với tiếng Việt) và gọn khoảng trắng."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def render(template: str, subject: str, value: str) -> str:
    return cap(norm(template.replace("{S}", subject).replace("{V}", value)))


def subject_forms(subject: str):
    """'con mèo' -> ['con mèo', 'mèo']; 'trời mưa' -> ['trời mưa']."""
    forms = [subject]
    parts = subject.split(" ")
    if parts[0] in CLASSIFIERS and len(parts) > 1:
        forms.append(" ".join(parts[1:]))
    return forms


def make_text(q: str, a: str) -> str:
    return f"Hỏi: {q}\nĐáp: {a}"


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def syllables(text: str):
    return re.findall(r"\w+", text.lower())


def _contains(hay: str, needle: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay) is not None


def usable_pool(pool, known_forms):
    """Bỏ chủ thể đã có trong bảng, và cả chủ thể chứa hoặc nằm trong một chủ thể của bảng theo ranh giới từ
    (ví dụ 'con chim cánh cụt' so với 'con chim', 'con cá heo' so với 'con cá'): những chủ thể đó mơ hồ,
    câu trả lời mong đợi (từ chối) không còn rõ ràng."""
    known_low = {k.lower() for k in known_forms}
    out = []
    for s in pool:
        forms = [f.lower() for f in subject_forms(s)]
        if any(f == k or _contains(f, k) or _contains(k, f) for f in forms for k in known_low):
            continue
        out.append(s)
    return out


def make_unknown_refusals(rng2, known_forms, per_attr):
    """Train v2: câu hỏi theo mẫu QUEN (không dùng 2 mẫu giữ lại cho test_A) với chủ thể chưa biết -> từ chối."""
    rows = []
    for attr, tpl in TEMPLATES.items():
        pool = usable_pool(POOLS[attr]["train"], known_forms)
        if len(pool) < 4:
            print(f"CẢNH BÁO: thuộc tính {attr!r} chỉ còn {len(pool)} chủ thể lạ để làm câu từ chối")
        qs = sorted({render(qt, s, "") for s0 in pool for s in subject_forms(s0)
                     for qt in tpl["q"][:-N_HELDOUT_Q]})
        rng2.shuffle(qs)
        for q in qs[:per_attr]:
            rows.append({"q": q, "a": rng2.choice(REFUSALS), "kind": "ood_unknown", "fact_id": None,
                         "attribute": None, "value": None})
    return rows


def make_test_d(rng2, known_forms, per_attr):
    """test_D: chủ thể lạ (không dùng ở train) với mẫu hỏi quen hoặc mẫu giữ lại; mong đợi từ chối."""
    rows = []
    for attr, tpl in TEMPLATES.items():
        pool = usable_pool(POOLS[attr]["test"], known_forms)
        n_fam = len(tpl["q"]) - N_HELDOUT_Q
        cands = sorted({(render(qt, s, ""), "familiar" if i < n_fam else "novel")
                        for s0 in pool for s in subject_forms(s0) for i, qt in enumerate(tpl["q"])})
        rng2.shuffle(cands)
        for q, kind in cands[:per_attr]:
            rows.append({"q": q, "a": REFUSALS[0], "attribute": attr, "template_kind": kind,
                         "text": make_text(q, REFUSALS[0])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="facts.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-fact", type=int, default=40, help="số cặp train tối đa mỗi sự thật")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--heldout-frac", type=float, default=0.15, help="tỉ lệ sự thật giữ lại cho test_B")
    ap.add_argument("--test-b-per-fact", type=int, default=8)
    ap.add_argument("--v2", action="store_true",
                    help="thêm câu từ chối theo mẫu hỏi quen (chủ thể lạ) vào train và tạo test_D; test_A/B/C giữ nguyên")
    ap.add_argument("--v2-pairs-per-attr", type=int, default=60, help="số cặp từ chối v2 mỗi thuộc tính")
    ap.add_argument("--test-d-per-attr", type=int, default=12, help="số dòng test_D mỗi thuộc tính")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.facts, encoding="utf-8", newline="") as f:
        facts = [{k: norm(v) for k, v in row.items()} for row in csv.DictReader(f)]
    for fact in facts:
        if fact["attribute"] not in TEMPLATES:
            raise SystemExit(f"Thuộc tính chưa có template: {fact['attribute']!r} (id={fact['id']})")

    # 1) Giữ lại sự thật cho test_B, chia đều theo từng thuộc tính.
    by_attr = defaultdict(list)
    for fact in facts:
        by_attr[fact["attribute"]].append(fact)
    heldout_ids = set()
    for attr, rows in by_attr.items():
        ids = [r["id"] for r in rows]
        rng.shuffle(ids)
        n_hold = max(1, round(args.heldout_frac * len(ids))) if len(ids) >= 5 else 0
        heldout_ids.update(ids[:n_hold])
    train_facts = [x for x in facts if x["id"] not in heldout_ids]
    held_facts = [x for x in facts if x["id"] in heldout_ids]
    train_vals = defaultdict(set)   # giá trị đã xuất hiện ở chủ thể khác trong train (model có thể đoán đúng nhờ đó)
    for x in train_facts:
        train_vals[x["attribute"]].add(x["value"])

    with (out / "facts_split.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "subject", "attribute", "value", "split"])
        for x in facts:
            w.writerow([x["id"], x["subject"], x["attribute"], x["value"],
                        "heldout" if x["id"] in heldout_ids else "train"])

    # 2) Sinh train (+ val) bằng các template hỏi KHÔNG bị giữ lại.
    pairs = []
    for fact in train_facts:
        tpl = TEMPLATES[fact["attribute"]]
        q_train = tpl["q"][:-N_HELDOUT_Q]
        combos = set()
        for s in subject_forms(fact["subject"]):
            for qt in q_train:
                for at in tpl["a"]:
                    q, a = render(qt, s, fact["value"]), render(at, s, fact["value"])
                    combos.add((q, a))
        combos = sorted(combos)
        rng.shuffle(combos)
        for q, a in combos[: args.max_per_fact]:
            pairs.append({"q": q, "a": a, "kind": "fact", "fact_id": fact["id"],
                          "attribute": fact["attribute"], "value": fact["value"]})

    for q in OOD_TRAIN:
        for a in REFUSALS:
            pairs.append({"q": norm(q), "a": a, "kind": "ood", "fact_id": None,
                          "attribute": None, "value": None})

    rng.shuffle(pairs)
    n_val = int(len(pairs) * args.val_frac)
    val_rows, train_rows = pairs[:n_val], pairs[n_val:]
    test_d = []
    if args.v2:
        # RNG riêng để test_A/B/C giống hệt lần chạy không có --v2 (rng chính không bị dùng thêm)
        rng2 = random.Random(args.seed + 1000)
        known_forms = {frm for x in facts for frm in subject_forms(x["subject"])}
        extra = make_unknown_refusals(rng2, known_forms, args.v2_pairs_per_attr)
        rng2.shuffle(extra)
        n_val_x = int(len(extra) * args.val_frac)
        val_rows = val_rows + extra[:n_val_x]
        train_rows = train_rows + extra[n_val_x:]
        rng2.shuffle(train_rows)
        rng2.shuffle(val_rows)
        test_d = make_test_d(rng2, known_forms, args.test_d_per_attr)
    for r in train_rows + val_rows:
        r["text"] = make_text(r["q"], r["a"])

    # 3) test_A: sự thật đã học + template hỏi chưa từng thấy.
    test_a = []
    for fact in train_facts:
        tpl = TEMPLATES[fact["attribute"]]
        for s in subject_forms(fact["subject"]):
            for qt in tpl["q"][-N_HELDOUT_Q:]:
                q = render(qt, s, fact["value"])
                a = render(tpl["a"][0], s, fact["value"])
                test_a.append({"q": q, "a": a, "value": fact["value"], "fact_id": fact["id"],
                               "attribute": fact["attribute"], "text": make_text(q, a)})

    # 4) test_B: sự thật giữ lại, hỏi bằng mọi template.
    test_b = []
    for fact in held_facts:
        tpl = TEMPLATES[fact["attribute"]]
        cands = []
        for s in subject_forms(fact["subject"]):
            for qt in tpl["q"]:
                q = render(qt, s, fact["value"])
                a = render(tpl["a"][0], s, fact["value"])
                cands.append((q, a))
        rng.shuffle(cands)
        for q, a in cands[: args.test_b_per_fact]:
            test_b.append({"q": q, "a": a, "value": fact["value"], "fact_id": fact["id"],
                           "attribute": fact["attribute"],
                           "value_seen_in_train": fact["value"] in train_vals[fact["attribute"]],
                           "text": make_text(q, a)})

    # 5) test_C: ngoài phạm vi, mong đợi từ chối.
    test_c = [{"q": norm(q), "a": REFUSALS[0], "text": make_text(norm(q), REFUSALS[0])} for q in OOD_TEST]

    # 6) Kiểm tra rò rỉ: dừng ngay nếu có.
    train_q = {r["q"] for r in train_rows}
    train_fact_ids = {r["fact_id"] for r in train_rows if r["fact_id"]}
    val_pairs = {(r["q"], r["a"]) for r in val_rows}
    train_pairs = {(r["q"], r["a"]) for r in train_rows}
    assert not val_pairs & train_pairs, "val trùng train"
    assert not {r["q"] for r in test_a} & train_q, "test_A có câu hỏi đã xuất hiện trong train"
    assert not {r["fact_id"] for r in test_b} & train_fact_ids, "test_B có sự thật nằm trong train"
    assert not {r["q"] for r in test_c} & train_q, "test_C có câu hỏi trùng train"
    assert not set(OOD_TRAIN) & set(OOD_TEST), "OOD train/test trùng nhau"
    assert not {r["q"] for r in test_d} & train_q, "test_D có câu hỏi trùng train"

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    write_jsonl(out / "test_A.jsonl", test_a)
    write_jsonl(out / "test_B.jsonl", test_b)
    write_jsonl(out / "test_C.jsonl", test_c)
    if args.v2:
        write_jsonl(out / "test_D.jsonl", test_d)

    # 7) Thống kê để tự kiểm tra bằng mắt.
    all_text = " ".join(r["text"] for r in train_rows)
    syl = Counter(syllables(all_text))
    q_len = [len(syllables(r["q"])) for r in train_rows]
    a_len = [len(syllables(r["a"])) for r in train_rows]
    first2 = Counter(" ".join(syllables(r["a"])[:2]) for r in train_rows if r["kind"] == "fact")
    n_fact_rows = sum(1 for r in train_rows if r["kind"] == "fact")
    per_attr = Counter(r["attribute"] for r in train_rows if r["kind"] == "fact")
    stats = {
        "facts_total": len(facts),
        "facts_train": len(train_facts),
        "facts_heldout": len(held_facts),
        "pairs": {"train": len(train_rows), "val": len(val_rows), "test_A": len(test_a),
                  "test_B": len(test_b), "test_C": len(test_c)},
        "ood_share_in_train": round(sum(1 for r in train_rows if r["kind"] == "ood") / len(train_rows), 3),
        "unique_syllables_in_train": len(syl),
        "avg_question_syllables": round(sum(q_len) / len(q_len), 2),
        "avg_answer_syllables": round(sum(a_len) / len(a_len), 2),
        "approx_train_tokens_syllable_level": sum(syl.values()),
        "top_answer_openings": [(k, round(v / n_fact_rows, 3)) for k, v in first2.most_common(5)],
        "train_pairs_per_attribute": dict(per_attr),
        "test_B_rows_value_seen_in_train": {
            "seen (có thể đoán đúng)": sum(1 for r in test_b if r["value_seen_in_train"]),
            "unseen (không thể đoán)": sum(1 for r in test_b if not r["value_seen_in_train"]),
        },
        "heldout_facts": [f"{x['id']}: {x['subject']} / {x['attribute']}" for x in held_facts],
        "seed": args.seed,
    }
    if args.v2:
        stats["v2"] = {
            "ood_unknown_pairs_in_train": sum(1 for r in train_rows if r["kind"] == "ood_unknown"),
            "ood_unknown_distinct_questions": len({r["q"] for r in train_rows if r["kind"] == "ood_unknown"}),
            "refusal_share_in_train": round(sum(1 for r in train_rows if r["kind"] != "fact") / len(train_rows), 3),
            "test_D_rows": len(test_d),
            "test_D_by_template_kind": dict(Counter(r["template_kind"] for r in test_d)),
        }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
