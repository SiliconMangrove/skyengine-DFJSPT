"""Event-driven joint scheduling, uncertainty estimation and rolling traffic."""
from .policy import DEFAULT_NODE_DIMS, JointPolicy, DFJSPTPolicyPlugin
from .reward import SMDPMakespanReward
from .trainer import CTDEPPOTrainer

ALGORITHM_METADATA = {
    "name": "DFJSP-T graph rolling PPO", "version": "0.2.0",
    "observation_schema": 2, "action_schema": 2, "backends": ["formal_headless"],
    "policy": "dfjsp_t_rl.policy:DFJSPTPolicyPlugin",
    "reward": "dfjsp_t_rl.reward:SMDPMakespanReward", "trainer": "dfjsp_t_rl.trainer:CTDEPPOTrainer",
}

def register(platform_registry=None):
    if platform_registry is None:
        from experiment.rl_platform.registry import register_algorithm
    else:
        register_algorithm = platform_registry
    register_algorithm(ALGORITHM_METADATA["name"], ALGORITHM_METADATA["policy"],
                       hyperparameters={"policy": DFJSPTPolicyPlugin.hyperparameter_schema()},
                       reward=ALGORITHM_METADATA["reward"], trainer=ALGORITHM_METADATA["trainer"],
                       version=ALGORITHM_METADATA["version"], backends=ALGORITHM_METADATA["backends"],
                       metrics=CTDEPPOTrainer.METRICS)
    return ALGORITHM_METADATA
