# config.py - cấu hình mô hình dùng chung (thay cho dict cfg bị chép ở nhiều script)
def make_cfg(vocab_size, drop_rate=0.1):
    return {
        "vocab_size": vocab_size,
        "context_length": 64,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": drop_rate,
        "qkv_bias": False,
        "tie_weights": True,
    }
