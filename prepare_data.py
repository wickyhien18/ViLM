import json
from array import array
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
BOS, EOS = tok.token_to_id("<s>"), tok.token_to_id("</s>")

def encode_file(name):
    stream = array("H")          # H = số nguyên 16 bit không dấu (đủ cho vocab 1068)
    n = 0
    for line in open(f"data/{name}.jsonl", encoding="utf-8"):
        text = json.loads(line)["text"]
        stream.append(BOS)
        stream.extend(tok.encode(text).ids)
        stream.append(EOS)
        n += 1
    with open(f"data/{name}.bin", "wb") as f:
        stream.tofile(f)
    return n, len(stream)

for name in ("train", "val"):
    n, total = encode_file(name)
    print(f"{name}: {n} cặp -> {total} token (trung bình {total / n:.1f} token/cặp)")
