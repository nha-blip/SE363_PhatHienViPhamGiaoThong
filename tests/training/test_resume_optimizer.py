import pytest
import torch

from scripts.resume_vehicle_local import normalize_optimizer_state, use_nonfused_adamw


def test_mixed_saved_moments_are_preserved_and_upcast():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    average = torch.tensor([0.25, 0.5], dtype=torch.float16)
    squares = torch.tensor([0.125, 0.25], dtype=torch.float32)
    optimizer.state[parameter] = {"step": torch.tensor(5.0), "exp_avg": average.clone(), "exp_avg_sq": squares.clone()}
    changes = normalize_optimizer_state(optimizer)
    assert len(changes) == 1
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], average.float())
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], squares)
    assert optimizer.state[parameter]["step"].item() == 5
    parameter.grad = torch.tensor([0.1, 0.2])
    optimizer.step()
    assert optimizer.state[parameter]["step"].item() == 6


def test_nonfinite_saved_state_is_not_silently_reset():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter] = {"exp_avg": torch.tensor([float('inf')])}
    with pytest.raises(ValueError):
        normalize_optimizer_state(optimizer)


def test_nonfused_policy_keeps_saved_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    optimizer.param_groups[0]["fused"] = True
    optimizer.state[parameter] = {"step": torch.tensor(5.0), "exp_avg": torch.tensor([0.25]), "exp_avg_sq": torch.tensor([0.125])}
    use_nonfused_adamw(optimizer)
    assert optimizer.param_groups[0]["fused"] is False
    assert optimizer.param_groups[0]["foreach"] is False
    assert optimizer.state[parameter]["step"].item() == 5
    assert optimizer.state[parameter]["exp_avg"].item() == 0.25
    parameter.grad = torch.tensor([0.1])
    optimizer.step()
    assert optimizer.state[parameter]["step"].item() == 6


def test_checkpoint_channels_last_moments_match_local_parameter_layout():
    parameter = torch.nn.Parameter(torch.ones(2, 3, 4, 5))
    optimizer = torch.optim.AdamW([parameter])
    average = torch.full_like(parameter, 0.25).to(memory_format=torch.channels_last)
    squares = torch.full_like(parameter, 0.125).to(memory_format=torch.channels_last)
    optimizer.state[parameter] = {"step": torch.tensor(5.0), "exp_avg": average, "exp_avg_sq": squares}
    assert average.stride() != parameter.stride()
    changes = normalize_optimizer_state(optimizer)
    assert len(changes) == 2
    assert optimizer.state[parameter]["exp_avg"].stride() == parameter.stride()
    assert optimizer.state[parameter]["exp_avg_sq"].stride() == parameter.stride()
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], average)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], squares)
    assert optimizer.state[parameter]["step"].item() == 5
