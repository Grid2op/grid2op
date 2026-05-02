from grid2op.Action import PlayableAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend


th_limits = {
    "0_1_0": 541.0,
    "0_4_1": 450.0,
    "1_2_2": 375.0,
    "1_3_3": 636.0,
    "1_4_4": 175.0,
    "2_3_5": 285.0,
    "3_4_6": 335.0,
    "5_10_7": 657.0,
    "5_11_8": 496.0,
    "5_12_9": 827.0,
    "8_9_10": 442.0,
    "8_13_11": 641.0,
    "9_10_12": 840.0,
    "11_12_13": 156.0,
    "12_13_14": 664.0,
    "3_6_15": 235.0,
    "3_8_16": 119.0,
    "4_5_17": 179.0,
    "6_7_18": 1986.0,
    "6_8_19": 1572.0,
}

config = {
    "backend": PandaPowerBackend,
    "action_class": PlayableAction,
    "observation_class": None,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "thermal_limits": th_limits,
    "names_chronics_to_grid": None,
}
