from gymnasium.envs.registration import register, registry


ENVIRONMENTS = {
    "anygoal/AvoidT-v1": "anygoal.envs.avoid_t:make_avoidt_env",
    "anygoal/CubeHop-v0": "anygoal.envs.cube_hop:make_cube_hop_env",
    "anygoal/CubeDiagonal-v0": (
        "anygoal.envs.cube_diagonal:make_cube_diagonal_env"
    ),
    "anygoal/ManipulatorInsertPeg-v0": (
        "anygoal.envs.manipulator_insert_peg:"
        "make_manipulator_insert_peg_env"
    ),
    "anygoal/StackT-v2": "anygoal.envs.stack_t:make_stackt_v2_env",
    "anygoal/TwoRoom-Tri-v1": (
        "anygoal.envs.tworoom_triangle:make_tworoom_triangle_env"
    ),
    "anygoal/PushTAntigoal-v3": (
        "anygoal.envs.push_t_antigoal:make_push_t_antigoal_v3_env"
    ),
}


def register_envs() -> None:
    for env_id, entry_point in ENVIRONMENTS.items():
        if env_id not in registry:
            register(id=env_id, entry_point=entry_point)
