from grid2op.Action import TopologyAndDispatchAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend

thermal_limits = {
    "0_1_0": 384.9001770019531,
    "0_4_1": 384.9001770019531,
    "8_9_2": 228997.109375,
    "8_13_3": 228997.109375,
    "9_10_4": 228997.109375,
    "11_12_5": 15266.4736328125,
    "12_13_6": 228997.109375,
    "1_2_7": 384.9001770019531,
    "1_3_8": 384.9001770019531,
    "1_4_9": 183.28579711914062,
    "2_3_10": 384.9001770019531,
    "3_4_11": 384.9001770019531,
    "5_10_12": 228997.109375,
    "5_11_13": 228997.109375,
    "5_12_14": 69393.0625,
    "3_6_15": 384.9001770019531,
    "3_8_16": 384.9001770019531,
    "4_5_17": 240.56260681152344,
    "6_7_18": 3402.24267578125,
    "6_8_19": 3402.24267578125,
}

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAndDispatchAction,
    "observation_class": None,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "thermal_limits": thermal_limits,
    "names_chronics_to_grid": None,
}
