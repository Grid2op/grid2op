from grid2op.Action import PowerlineSetAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import WeightedRandomOpponent, BaseActionBudget


try:
    from grid2op.l2rpn_utils import ActionNeurips2020, ObservationNeurips2020
except ImportError:
    from grid2op.Action import TopologyAndDispatchAction
    from grid2op.Observation import CompleteObservation
    import warnings

    warnings.warn(
        "The grid2op version you are trying to use is too old for this environment. Please upgrade it."
    )
    ActionNeurips2020 = TopologyAndDispatchAction
    ObservationNeurips2020 = CompleteObservation


lines_attacked = [
    "62_58_180",
    "62_63_160",
    "48_50_136",
    "48_53_141",
    "41_48_131",
    "39_41_121",
    "43_44_125",
    "44_45_126",
    "34_35_110",
    "54_58_154",
]
rho_normalization = [0.45, 0.45, 0.6, 0.35, 0.3, 0.2, 0.55, 0.3, 0.45, 0.55]
opponent_attack_cooldown = 12 * 24  # 24 hours, 1 hour being 12 time steps
opponent_attack_duration = 12 * 4  # 4 hours
opponent_budget_per_ts = (
    0.16667  # opponent_attack_duration / opponent_attack_cooldown + epsilon
)
opponent_init_budget = 144.0  # no need to attack straightfully, it can attack starting at midday the first day

thermal_limits = {
    "34_35_110": 60.900001525878906,
    "34_36_111": 231.89999389648438,
    "32_36_112": 272.6000061035156,
    "33_35_113": 212.8000030517578,
    "33_36_114": 749.2000122070312,
    "36_38_115": 332.3999938964844,
    "36_39_116": 348.0,
    "38_39_119": 414.3999938964844,
    "39_40_120": 310.1000061035156,
    "39_41_121": 371.3999938964844,
    "40_41_122": 401.20001220703125,
    "42_43_123": 124.30000305175781,
    "33_42_124": 298.5,
    "43_44_125": 86.4000015258789,
    "44_45_126": 213.89999389648438,
    "45_46_127": 160.8000030517578,
    "45_47_128": 112.19999694824219,
    "46_48_130": 291.3999938964844,
    "41_48_131": 489.0,
    "41_48_132": 489.0,
    "44_48_133": 124.5999984741211,
    "47_48_134": 196.6999969482422,
    "48_49_135": 191.89999389648438,
    "48_50_136": 238.39999389648438,
    "50_51_137": 174.1999969482422,
    "51_52_138": 105.5999984741211,
    "52_53_139": 143.6999969482422,
    "48_53_141": 293.3999938964844,
    "48_53_142": 288.8999938964844,
    "53_54_143": 107.69999694824219,
    "53_55_144": 415.5,
    "54_55_145": 148.1999969482422,
    "55_56_146": 124.19999694824219,
    "49_56_147": 154.39999389648438,
    "55_57_148": 85.9000015258789,
    "50_57_149": 106.5,
    "53_58_150": 142.0,
    "55_58_152": 124.0,
    "55_58_153": 130.1999969482422,
    "54_58_154": 86.19999694824219,
    "58_59_155": 278.1000061035156,
    "58_60_156": 182.0,
    "59_60_157": 592.0999755859375,
    "59_61_158": 173.10000610351562,
    "60_61_159": 249.8000030517578,
    "62_63_160": 441.0,
    "37_64_161": 344.20001220703125,
    "63_64_163": 722.7999877929688,
    "48_65_164": 494.6000061035156,
    "48_65_165": 494.6000061035156,
    "61_65_166": 196.6999969482422,
    "61_66_167": 151.8000030517578,
    "65_66_168": 263.3999938964844,
    "46_68_169": 364.1000061035156,
    "48_68_170": 327.0,
    "37_36_179": 370.5,
    "62_58_180": 441.0,
    "63_60_181": 300.29998779296875,
    "64_65_182": 656.2000122070312,
}

config = {
    "backend": PandaPowerBackend,
    "action_class": ActionNeurips2020,
    "observation_class": ObservationNeurips2020,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "thermal_limits": thermal_limits,
    "opponent_attack_cooldown": opponent_attack_cooldown,
    "opponent_attack_duration": opponent_attack_duration,
    "opponent_budget_per_ts": opponent_budget_per_ts,
    "opponent_init_budget": opponent_init_budget,
    "opponent_action_class": PowerlineSetAction,
    "opponent_class": WeightedRandomOpponent,
    "opponent_budget_class": BaseActionBudget,
    "kwargs_opponent": {
        "lines_attacked": lines_attacked,
        "rho_normalization": rho_normalization,
        "attack_period": opponent_attack_cooldown,
    },
}
