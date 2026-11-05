# dataset.py - bộ nạp dữ liệu từ data/train.bin và data/val.bin
import numpy as np
import torch

CONTEXT = 64  # phải khớp với model.py


def load_tokens(path):
    # mỗi token 2 byte không dấu (uint16), xem prepare_data.py
    return np.fromfile(path, dtype=np.uint16)


def _window(data, start, length):
    return torch.from_numpy(data[start:start + length].astype(np.int64))


def get_train_batch(data, batch_size, context=CONTEXT, device="cpu"):
    # chọn ngẫu nhiên điểm bắt đầu; y là x dịch sang phải đúng 1 token
    starts = torch.randint(0, len(data) - context, (batch_size,)).tolist()
    x = torch.stack([_window(data, i, context) for i in starts])
    y = torch.stack([_window(data, i + 1, context) for i in starts])
    return x.to(device), y.to(device)


def val_batches(data, batch_size, context=CONTEXT, device="cpu"):
    # các đoạn không chồng lên nhau, phủ gần hết val (bỏ phần dư < context)
    n = (len(data) - 1) // context
    for b in range(0, n, batch_size):
        idx = range(b, min(b + batch_size, n))
        x = torch.stack([_window(data, i * context, context) for i in idx])
        y = torch.stack([_window(data, i * context + 1, context) for i in idx])
        yield x.to(device), y.to(device)


if __name__ == "__main__":
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("tokenizer.json")
    vocab = tok.get_vocab_size()
    train = load_tokens("data/train.bin")
    val = load_tokens("data/val.bin")
    print(f"vocab {vocab} | train {len(train)} token | val {len(val)} token")
    print(f"id lớn nhất: train {train.max()} | val {val.max()} | phải < {vocab}")
    assert train.max() < vocab and val.max() < vocab

    x, y = get_train_batch(train, 4)
    print("hình dạng x, y:", tuple(x.shape), tuple(y.shape), x.dtype)
    assert torch.equal(y[:, :-1], x[:, 1:])
    print("đích = đầu vào dịch 1 token: ĐÚNG")
    print("--- giải mã hàng đầu của x ---")
    print(tok.decode(x[0].tolist(), skip_special_tokens=False))

    n = sum(xb.shape[0] for xb, _ in val_batches(val, 16))
    print(f"val: {n} đoạn, phủ {n * CONTEXT}/{len(val)} token")