import json, re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

texts = [json.loads(l)["text"] for l in open("data/train.jsonl", encoding="utf-8")]
syl = lambda t: re.findall(r"\w+", t.lower())
n_syl = sum(len(syl(t)) for t in texts)
vocab_syl = sorted({w for t in texts for w in syl(t)})

def train_tok(vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    return tok

print("số âm tiết trong train:", n_syl, "| số âm tiết khác nhau:", len(vocab_syl))
for v in (512, 1024, 2048):
    tok = train_tok(v)
    n_tok = sum(len(tok.encode(t).ids) for t in texts)
    one = sum(1 for w in vocab_syl if len(tok.encode(" " + w).ids) == 1)
    print(f"vocab {v}: token/âm tiết = {n_tok / n_syl:.2f} | {100 * one / len(vocab_syl):.0f}% âm tiết khác nhau chỉ là 1 token")
