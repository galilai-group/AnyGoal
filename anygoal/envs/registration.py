from gymnasium.envs.registration import register, registry


ENVIRONMENTS = {
    "anygoal/AvoidT-v1": "anygoal.envs.avoidt.env:make_avoidt_env",
    "anygoal/CubeHop-v0": "anygoal.envs.cube_hop.env:make_cube_hop_env",
    "anygoal/CubeDiagonal-v0": (
        "anygoal.envs.cube_diagonal.env:make_cube_diagonal_env"
    ),
    "anygoal/InsertPeg-v0": (
        "anygoal.envs.insert_peg.env:"
        "make_insert_peg_env"
    ),
    "anygoal/StackT-v2": "anygoal.envs.stackt.env:make_stackt_v2_env",
    "anygoal/TwoRoom-Route-v1": (
        "anygoal.envs.tworoom_route.env:make_tworoom_route_env"
    ),
    "anygoal/PushTAntigoal-v3": (
        "anygoal.envs.pusht_antigoal.env:make_push_t_antigoal_v3_env"
    ),
}


def register_envs() -> None:
    for env_id, entry_point in ENVIRONMENTS.items():
        if env_id not in registry:
            register(id=env_id, entry_point=entry_point)
