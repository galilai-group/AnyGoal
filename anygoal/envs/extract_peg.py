"""Insert-Peg environment with a releasable hand-to-peg grasp constraint."""

from dm_control import suite
from dm_control.rl import control
from dm_control.suite import manipulator as insert_peg_suite

from anygoal.envs.insert_peg import InsertPegEnv


class ExtractPegEnv(InsertPegEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_xml, assets = insert_peg_suite.make_model(use_peg=True, insert=True)
        model_xml = model_xml.replace(
            b"</equality>",
            b'<weld name="extract_grasp" body1="hand" body2="peg" '
            b'active="false" solref=".005 1"/></equality>',
        )
        physics = insert_peg_suite.Physics.from_xml_string(model_xml, assets)
        task = insert_peg_suite.Bring(
            use_peg=True, insert=True, fully_observable=True, random=None
        )
        self.dmc_env = control.Environment(
            physics,
            task,
            control_timestep=insert_peg_suite._CONTROL_TIMESTEP,
            time_limit=insert_peg_suite._TIME_LIMIT,
        )
        self.env_name = "ExtractPeg"
        for geom_name in ("floor", "slot_0", "slot_1", "slot_2"):
            self.physics.named.model.geom_contype[geom_name] = 1
            self.physics.named.model.geom_conaffinity[geom_name] = 1
        self.physics.named.model.geom_friction[
            ["slot_0", "slot_1", "slot_2"]
        ] = (0.7, 0.005, 0.0001)
        self.physics.forward()
