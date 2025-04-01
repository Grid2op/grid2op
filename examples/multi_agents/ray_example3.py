# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""example with local observation and local actions"""

from ray.rllib.policy.policy import PolicySpec
    
import grid2op
from grid2op.Action import PlayableAction

from lightsim2grid import LightSimBackend
from grid2op.multi_agent import LouvainClustering

from ray_example2 import MAEnvWrapper

ENV_NAME = "l2rpn_case14_sandbox"
DO_NOTHING_EPISODES = -1  # 200

env_for_cls = grid2op.make(ENV_NAME,
                           action_class=PlayableAction,
                           backend=LightSimBackend())


# Get ACTION_DOMAINS by clustering the substations
ACTION_DOMAINS = LouvainClustering.cluster_substations(env_for_cls)

# Get OBSERVATION_DOMAINS by clustering the substations
OBSERVATION_DOMAINS = LouvainClustering.cluster_substations(env_for_cls)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id


if __name__ == "__main__":
    import ray
    # from ray.rllib.agents.ppo import ppo
    from ray.rllib.algorithms.ppo import PPO, PPOConfig
    import json
    import os
    import shutil
    
    ray_ma_env = MAEnvWrapper()
    
    checkpoint_root = "./ma_ppo_test_2ndsetting"
    
    # Where checkpoints are written:
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)

    # Where some data will be written and used by Tensorboard below:
    ray_results = f'{os.getenv("HOME")}/ray_results/'
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    info = ray.init(ignore_reinit_error=True)
    print("Dashboard URL: http://{}".format(info.address_info["webui_url"]))
    
    # #Configs (see ray's doc for more information)
    SELECT_ENV = MAEnvWrapper                            # Specifies the OpenAI Gym environment for Cart Pole
    N_ITER = 1000                                     # Number of training runs.
    
    # see ray doc for this...
    # syntax changes every ray major version apparently...
    config = PPOConfig()
    config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=1)
   
    # multi agent parts
    config.multi_agent(policies={
        "agent_0" : PolicySpec(
            action_space=ray_ma_env.action_space["agent_0"],
            observation_space=ray_ma_env.observation_space["agent_0"]
        ),
        "agent_1" : PolicySpec(
            action_space=ray_ma_env.action_space["agent_1"],
            observation_space=ray_ma_env.observation_space["agent_1"],
        )
        }, 
                    policy_mapping_fn = policy_mapping_fn, 
                    policies_to_train= ["agent_0", "agent_1"])
         
    #Trainer
    agent = PPO(config=config, env=SELECT_ENV)

    results = []
    episode_data = []
    episode_json = []

    for n in range(N_ITER):
        result = agent.train()
        results.append(result)
        
        episode = {'n': n, 
                   'episode_reward_min': result['episode_reward_min'], 
                   'episode_reward_mean': result['episode_reward_mean'], 
                   'episode_reward_max': result['episode_reward_max'],  
                   'episode_len_mean': result['episode_len_mean']
                  }
        
        episode_data.append(episode)
        episode_json.append(json.dumps(episode))
        file_name = agent.save(checkpoint_root)
        
        print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}. Checkpoint saved to {file_name}')

        with open(f'{ray_results}/rewards.json', 'w') as outfile:
            json.dump(episode_json, outfile)
