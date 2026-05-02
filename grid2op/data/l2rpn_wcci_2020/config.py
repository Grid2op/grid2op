from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecastsWithMaintenance
from grid2op.Backend import PandaPowerBackend


try:
    from grid2op.l2rpn_utils import ActionWCCI2020, ObservationWCCI2020
except ImportError:
    from grid2op.Action import TopologyAndDispatchAction
    from grid2op.Observation import CompleteObservation
    import warnings

    warnings.warn(
        "The grid2op version you are trying to use is too old for this environment. Please upgrade it."
    )
    ActionWCCI2020 = TopologyAndDispatchAction
    ObservationWCCI2020 = CompleteObservation

thermal_limits = {
    "2_3_0": 43.29999923706055,
    "2_4_1": 205.1999969482422,
    "0_4_2": 341.20001220703125,
    "1_3_3": 204.0,
    "1_4_4": 601.4000244140625,
    "4_6_5": 347.1000061035156,
    "4_7_6": 319.6000061035156,
    "6_7_7": 301.3999938964844,
    "7_8_8": 330.29998779296875,
    "7_9_9": 274.1000061035156,
    "8_9_10": 307.3999938964844,
    "10_11_11": 172.3000030517578,
    "1_10_12": 354.29998779296875,
    "11_12_13": 127.9000015258789,
    "12_13_14": 174.89999389648438,
    "13_14_15": 152.60000610351562,
    "13_15_16": 81.80000305175781,
    "14_16_17": 204.3000030517578,
    "9_16_18": 561.5,
    "9_16_19": 561.5,
    "12_16_20": 98.69999694824219,
    "15_16_21": 179.8000030517578,
    "16_17_22": 193.39999389648438,
    "16_18_23": 239.89999389648438,
    "18_19_24": 164.8000030517578,
    "19_20_25": 100.4000015258789,
    "20_21_26": 125.69999694824219,
    "16_21_27": 278.20001220703125,
    "16_21_28": 274.0,
    "21_22_29": 89.9000015258789,
    "21_23_30": 352.1000061035156,
    "22_23_31": 157.10000610351562,
    "23_24_32": 124.4000015258789,
    "17_24_33": 154.60000610351562,
    "23_25_34": 86.0999984741211,
    "18_25_35": 106.69999694824219,
    "21_26_36": 148.5,
    "23_26_37": 129.60000610351562,
    "23_26_38": 136.10000610351562,
    "22_26_39": 86.0,
    "26_27_40": 313.20001220703125,
    "26_28_41": 198.5,
    "27_28_42": 599.0999755859375,
    "27_29_43": 206.8000030517578,
    "28_29_44": 233.6999969482422,
    "30_31_45": 395.79998779296875,
    "5_32_46": 516.7000122070312,
    "31_32_47": 656.4000244140625,
    "16_33_48": 583.0,
    "16_33_49": 583.0,
    "29_33_50": 263.1000061035156,
    "29_34_51": 222.60000610351562,
    "33_34_52": 322.79998779296875,
    "14_35_53": 340.6000061035156,
    "16_35_54": 305.20001220703125,
    "4_5_55": 360.1000061035156,
    "26_30_56": 395.79998779296875,
    "28_31_57": 274.20001220703125,
    "32_33_58": 605.5,
}

config = {
    "backend": PandaPowerBackend,
    "action_class": ActionWCCI2020,
    "observation_class": ObservationWCCI2020,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecastsWithMaintenance,
    "volagecontroler_class": None,
    "names_chronics_to_grid": {},
    "thermal_limits": thermal_limits,
}
