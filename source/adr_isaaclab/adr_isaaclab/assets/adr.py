import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


KINOVA_BIMANUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/alvaro/adr_isaaclab/source/adr_isaaclab/adr_isaaclab/assets/data/ADR.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            ".*right_joint_1": 0.0, #base rotation 
            ".*right_joint_2": 0.7854,# shoulder
            ".*right_joint_3": 0.0, #half arm 1
            ".*right_joint_4": 1.5708, #half arm 2
            ".*right_joint_5": 0.0, #forearm
            ".*right_joint_6": 0.2618, #wrist 1
            ".*right_joint_7": 0.0, #wrist 2
            ".*left_joint_1": 0.0, #base rotation
            ".*left_joint_2": -0.7854,# shoulder
            ".*left_joint_3": 0.0, #half arm 1
            ".*left_joint_4": -1.5708, #half arm 2
            ".*left_joint_5": 0.0, #forearm
            ".*left_joint_6": -0.2618, #wrist 1
            ".*left_joint_7": 0.0, #wrist 2
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # Data retrieved from: https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf
        "large": IdealPDActuatorCfg(
            joint_names_expr=[".*joint_1", ".*joint_2", ".*joint_3", ".*joint_4"],
            effort_limit={
                ".*joint_1": 39.0, #Nm
                ".*joint_2": 39.0,
                ".*joint_3": 39.0,
                ".*joint_4": 39.0,
            },
            velocity_limit=1.39, #rad/s
            stiffness=20.0,
            damping=1.0,
        ),
        "small": IdealPDActuatorCfg(
            joint_names_expr=[".*joint_5", ".*joint_6", ".*joint_7",],
            effort_limit={
                ".*joint_5": 13.0, #Nm
                ".*joint_6": 13.0,
                ".*joint_7": 13.0,
            },
            velocity_limit=1.22, #rad/s
            stiffness=5.0,
            damping=0.5,
        ),
    },
)
"""Configuration for the dual-arm Kinova Gen3 robots in Isaac Lab."""
