import torch

from minigpt.model import GPT, GPTConfig, sample_next

CONFIG = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def make_model(seed=0):
    torch.manual_seed(seed)
    return GPT(CONFIG).eval()


def test_forward_shapes_and_loss():
    model = make_model()
    idx = torch.randint(50, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, 50)
    assert loss is None
    _, loss = model(idx, idx)
    # An untrained model should be close to uniform: loss ~ ln(vocab_size).
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.5


def test_ignore_index_is_skipped():
    model = make_model()
    idx = torch.randint(50, (1, 8))
    targets = torch.full((1, 8), -100)
    targets[0, 3] = 7
    _, loss = model(idx, targets)
    logits, _ = model(idx)
    expected = torch.nn.functional.cross_entropy(logits[0, 3:4], torch.tensor([7]))
    assert torch.allclose(loss, expected)


def test_attention_is_causal():
    model = make_model()
    a = torch.randint(50, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 50  # change only the future
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5)
    assert not torch.allclose(la[0, 8:], lb[0, 8:])


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = GPT(CONFIG)
    opt = model.configure_optimizer(learning_rate=3e-3, weight_decay=0.0)
    x = torch.randint(50, (4, 16))
    y = torch.randint(50, (4, 16))
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_limits_and_stop_tokens():
    model = make_model()
    prompt = torch.tensor([[1, 2, 3]])
    out = list(model.generate(prompt, max_new_tokens=40, temperature=1.0))
    assert len(out) == 40  # longer than block_size: the context window slides

    greedy = list(model.generate(prompt, max_new_tokens=5, temperature=0))
    stopped = list(model.generate(prompt, max_new_tokens=5, temperature=0, stop_ids={greedy[2]}))
    assert stopped == greedy[:greedy.index(greedy[2])]


def test_sampling_filters():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
    assert sample_next(logits, temperature=0).item() == 1
    assert all(sample_next(logits, top_k=1).item() == 1 for _ in range(20))
    assert all(sample_next(logits, top_p=0.01).item() == 1 for _ in range(20))
    samples = {sample_next(logits, top_k=2).item() for _ in range(200)}
    assert samples == {1, 2}


def test_save_and_load(tmp_path):
    model = make_model()
    path = tmp_path / "ckpt.pt"
    model.save(path, step=7)
    loaded, ckpt = GPT.load(path)
    assert ckpt["step"] == 7
    idx = torch.randint(50, (2, 10))
    assert torch.equal(model(idx)[0], loaded.eval()(idx)[0])
    # Weight tying survives the round trip.
    assert loaded.lm_head.weight is loaded.tok_emb.weight
