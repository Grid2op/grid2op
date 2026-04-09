from grid2op.Action import TopologyAndDispatchAction, PowerlineSetAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget

thermal_limits = {
    "0_1_0": 384.9001770019531,
    "0_4_1": 384.9001770019531,
    "1_2_2": 380.0,
    "1_3_3": 380.0,
    "1_4_4": 157.0,
    "2_3_5": 380.0,
    "3_4_6": 380.0,
    "5_10_7": 1077.720458984375,
    "5_11_8": 461.8802185058594,
    "5_12_9": 769.8003540039062,
    "8_9_10": 269.43011474609375,
    "8_13_11": 384.9001770019531,
    "9_10_12": 760.0,
    "11_12_13": 380.0,
    "12_13_14": 760.0,
    "3_6_15": 384.9001770019531,
    "3_8_16": 230.9401092529297,
    "4_5_17": 170.79945373535156,
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
    "opponent_attack_cooldown": 12 * 24,
    "opponent_attack_duration": 12 * 4,
    "opponent_budget_per_ts": 0.5,
    "opponent_init_budget": 0.0,
    "opponent_action_class": PowerlineSetAction,
    "opponent_class": RandomLineOpponent,
    "opponent_budget_class": BaseActionBudget,
    "kwargs_opponent": {
        "lines_attacked": [
            "1_3_3",
            "1_4_4",
            "3_6_15",
            "9_10_12",
            "11_12_13",
            "12_13_14",
        ]
    },
}
