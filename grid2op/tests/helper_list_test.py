#!/usr/bin/env python3
import unittest
li_tested_elsewhere = [
    # agent (approx 1 min)
    "test_Agent", "test_AgentsFast", "test_recopowerlineperarea",
    "test_Opponent", "test_baseline_alert",
    # converter (approx 45s)
    "test_AgentConverter", "test_Converter", "test_BackendConverter",
    # Runner / EpisodeData / "score (3 mins)"
    "test_EpisodeData",
    "test_runner_kwargs_backend",
    "test_Runner",
    "test_RunnerFast",
    "test_score_idf_2023_assistant",
    "test_score_idf_2023_nres",
    "test_score_idf_2023",
    "test_score_wcci_2022",
    "test_AlarmScore",
    "test_RewardAlertCostScore",
    "test_RewardNewRenewableSourcesUsageScore",
    "test_utils",
    "test_CompactEpisodeData",
    "test_reset_options_runner",
    # env in general (1 min)
    "test_attached_envs",
    "test_attached_envs_compat",
    "test_l2rpn_idf_2023",
    "test_MultiMix",
    "test_timeOutEnvironment",
    "test_MaskedEnvironment",
    "test_MakeEnv",
    "test_multi_steps_env",
    "test_simenv_blackout",
    "test_get_default_env_kwargs",
    "test_Environment",
    "test_EnvironmentCpy",
    "test_generate_classes",
    "test_basic_env_ls",
    # alert / alarm ( 1min)
    "test_AlarmFeature",
    "test_alert_gym_compat",
    "test_alert_obs_act",
    "test_alert_trust_score",
    "test_AlertReward",
    # time_series / backend
    "test_ts_handlers",
    "test_ChronicsHandler",
    "test_env_from_episode",
    # backend
    "test_basicBackendInterface",
    "test_action_set_orig_state",
    "test_PandaPowerBackendDefaultFunc",
    "test_PandaPowerBackend",
    "test_BackendAction"
    
    # TODO simulate
    # TODO curtailment
    ]


def print_suite(suite):
    if hasattr(suite, "__iter__"):
        for x in suite:
            print_suite(x)
    else:
        testmodule = suite.__class__.__module__
        testsuite = suite.__class__.__name__
        testmethod = suite._testMethodName
        do_print = True
        if testmodule.startswith("test_issue_"):
            # the test_issue_* will be tested elsewhere
            do_print = False
        else:
            for el in li_tested_elsewhere:
                if testmodule == el:
                    do_print = False
                    break
        test_name = "{}.{}.{}".format(testmodule, testsuite, testmethod)
        
        # sometimes tests are included in other tests
        # and so they are executed like "grid2op.tests.test_Agent.TestAgent.test_0_donothing"
        # this prevents it
        for el in li_tested_elsewhere:
            if test_name.startswith("grid2op.tests") and el in test_name:
                do_print = False
                break
            
        if do_print:
            print(test_name)


print_suite(unittest.defaultTestLoader.discover("."))
