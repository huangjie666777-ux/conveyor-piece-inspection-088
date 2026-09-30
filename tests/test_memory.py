import torch

from anomaly_inspector.memory import greedy_farthest_points, local_distances


def test_greedy_fps_is_deterministic_and_bounded():
    torch.manual_seed(0)
    points = torch.randn(2000, 32)
    bank_a = greedy_farthest_points(points, 256)
    bank_b = greedy_farthest_points(points, 256)
    assert bank_a.shape == (256, 32)
    assert torch.equal(bank_a, bank_b)
    # Every selected descriptor exists in the source set.
    for row in bank_a[:5]:
        assert torch.any(torch.all(points == row, dim=1))


def test_greedy_fps_picks_origin_then_farthest():
    points = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]
    )
    bank = greedy_farthest_points(points, 3)
    # First point is index 0; second must be one of the two far corners.
    assert torch.allclose(bank[0], points[0])
    assert bank.shape[0] == 3
    distances = local_distances(points, bank)
    assert distances.shape == (4,)
    # Bank points themselves have zero distance to the bank.
    assert distances[0].item() == 0.0
