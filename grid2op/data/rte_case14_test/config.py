from grid2op.Action import TopologyAndDispatchAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend

thermal_limits = {
    "0_1_0": 352.8251647949219,
    "0_4_1": 352.8251647949219,
    "8_9_2": 183197.6875,
    "8_13_3": 183197.6875,
    "9_10_4": 183197.6875,
    "11_12_5": 12213.1787109375,
    "12_13_6": 183197.6875,
    "1_2_7": 352.8251647949219,
    "1_3_8": 352.8251647949219,
    "1_4_9": 352.8251647949219,
    "2_3_10": 352.8251647949219,
    "3_4_11": 352.8251647949219,
    "5_10_12": 183197.6875,
    "5_11_13": 183197.6875,
    "5_12_14": 183197.6875,
    "3_6_15": 352.8251647949219,
    "3_8_16": 352.8251647949219,
    "4_5_17": 352.8251647949219,
    "6_7_18": 2721.794189453125,
    "6_8_19": 2721.794189453125,
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
