import torch

from kda_train.data import ByteDataset
from kda_train.model import KDAForCausalLM, ModelConfig


def test_reference_kda_forward_and_backward() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=32,
        dim=16,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2,
        max_seq_len=6,
        kda_backend="reference",
    )
    model = KDAForCausalLM(config)
    inputs = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    targets = torch.randint(0, config.vocab_size, (2, config.max_seq_len))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, config.max_seq_len, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_byte_dataset_returns_next_byte_targets(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(bytes(range(80)))
    dataset = ByteDataset.from_text(corpus, validation_fraction=0.25)

    x, y = dataset.get_batch("train", batch_size=3, seq_len=8, device=torch.device("cpu"))

    assert x.shape == y.shape == (3, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
