import numpy as np
import anygoal  # noqa: F401
from gymnasium import make

from anygoal.datasets.tworoom.support import (
    TwoRoomHDF5Writer,
    make_episode,
    validate_dataset,
)
from anygoal.envs.tworoom_route.env import (
    ROUTE_POINTS,
    ROUTES,
)
from anygoal.envs.tworoom_route.policy import TwoRoomRoutePolicy


def test_routes_use_declared_points():
    assert ROUTE_POINTS == {
        1: (36.0, 188.0),
        2: (97.0, 188.0),
        3: (36.0, 80.0),
        4: (36.0, 36.0),
        5: (97.0, 36.0),
        6: (127.0, 188.0),
        7: (188.0, 188.0),
        8: (188.0, 80.0),
        9: (127.0, 36.0),
        10: (188.0, 36.0),
    }
    assert len(ROUTES) == 8
    for route in ROUTES.values():
        assert len(route) == 3
        assert all(point in ROUTE_POINTS for point in route)


def test_constant_policy_preserves_sampled_magnitude_with_noise():
    policy = TwoRoomRoutePolicy(
        route_id=1,
        condition="noisy_constant",
        rng=np.random.default_rng(0),
    )
    actions = [policy.action(np.array(ROUTE_POINTS[2]))[0] for _ in range(100)]
    magnitudes = np.linalg.norm(actions, axis=1)
    np.testing.assert_allclose(magnitudes, policy.sampled_magnitude, atol=1e-6)
    assert np.all((magnitudes >= 0.25) & (magnitudes <= 1.0))


def test_varying_policy_samples_magnitudes_per_step():
    policy = TwoRoomRoutePolicy(
        route_id=1,
        condition="clean_varying",
        rng=np.random.default_rng(0),
    )
    actions = [policy.action(np.array(ROUTE_POINTS[2]))[0] for _ in range(20)]
    magnitudes = np.linalg.norm(actions, axis=1)
    assert np.ptp(magnitudes) > 0.1
    assert np.all((magnitudes >= 0.25) & (magnitudes <= 1.0))


def test_registered_env_keeps_tworoom_success_and_hides_target():
    env = make("anygoal/TwoRoom-Route-v1", render_mode="rgb_array")
    _, _ = env.reset(
        seed=0,
        options={
            "variation": [],
            "state": np.array((60.0, 112.0), dtype=np.float32),
            "target_state": np.array((64.0, 112.0), dtype=np.float32),
        },
    )
    frame = env.render()
    _, _, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated
    assert not np.any(frame[..., 1] > frame[..., 0])
    env.close()


def test_tworoom_support_writes_and_validates_canonical_schema(tmp_path):
    path = tmp_path / "tworoom.h5"
    columns = {}
    for key, (tail_shape, dtype) in TwoRoomHDF5Writer.PER_STEP_SPECS.items():
        columns[key] = np.zeros((2, *tail_shape), dtype=dtype).tolist()
    columns["terminated"][-1] = True
    episode = make_episode(columns, 1, "clean_constant", 0.5)

    writer = TwoRoomHDF5Writer(path)
    writer.write(episode)
    writer.close({"environment": "anygoal/TwoRoom-Route-v1"})
    validate_dataset(path, expected_episodes=1)
