from cadvmh.training.stages import configure_stage


def test_freezing_policy(tiny_model):
    configure_stage(tiny_model, 1)
    assert any(p.requires_grad for p in tiny_model.encoder.parameters())
    assert not any(p.requires_grad for p in tiny_model.selector.parameters())
    configure_stage(tiny_model, 2)
    assert all(not p.requires_grad for p in tiny_model.encoder.parameters())
    assert all(p.requires_grad for p in tiny_model.selector.parameters())
    configure_stage(tiny_model, 3)
    assert all(p.requires_grad for p in tiny_model.parameters())
