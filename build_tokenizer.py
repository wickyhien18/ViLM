import json, re
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

texts = [json.loads(l)["text"] for l in open("data/train.jsonl", encoding="utf-8")]
syl = lambda t: re.findall(r"\w+", t.lower())
vocab_syl = sorted({w for t in texts for w in syl(t)})

def train_tok(vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    return tok

Path("tokenizers").mkdir(exist_ok=True)
sample = "Hỏi: Con mèo kêu như thế nào?\nĐáp: Con mèo kêu meo meo."
for cap in (1024, 2048):
    tok = train_tok(cap)
    tok.save(f"tokenizers/tok_{cap}.json")
    split = [i for i, w in enumerate(vocab_syl) if len(tok.encode(" " + w).ids) != 1]
    print(f"\n=== đặt vocab_size = {cap} -> kích thước THẬT = {tok.get_vocab_size()}")
    print("id đặc biệt:", {t: tok.token_to_id(t) for t in ["<pad>", "<s>", "</s>", "<unk>"]})
    print("âm tiết bị cắt vụn (tối đa 25):", [vocab_syl[i] for i in split[:25]])
    ids = tok.encode(sample).ids
    print("câu mẫu thành", len(ids), "token:", [tok.decode([i]) for i in ids])
