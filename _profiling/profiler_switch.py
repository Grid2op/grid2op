# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This file should be used to assess the performance of grid2op in "runner" mode: the loading time are not studied,
neither are the import times.

Data are loaded only once, when the environment is "done" the programm stops.

This corresponds to the situation: you have a trained agent, you want to assess its performance using the runner
"""

import os

from grid2op import make
from grid2op.Action import CompleteAction
from grid2op.Space import AddDetailedTopoIEEE
from grid2op.Parameters import Parameters
from grid2op.Agent import BaseAgent
from grid2op.Rules import AlwaysLegal
from grid2op.Backend import PandaPowerBackend
import cProfile

from utils_benchmark import run_env, str2bool

try:
    from lightsim2grid import LightSimBackend
    light_sim_avail = True
except ImportError:
    light_sim_avail = False

ENV_NAME = "rte_case5_example"
ENV_NAME = "rte_case14_realistic"
MAX_TS = 1000


class RandomBBsSwitchAgent(BaseAgent):
    def act(self, observation, reward, done=False):
        sub_id = self.space_prng.randint(type(self.action_space).n_sub)
        new_position = self.space_prng.choice([-1, 1], size=1)
        return self.action_space({"set_switch": [(sub_id, new_position)]})


def main(max_ts, name, use_lightsim=False, test_env=True):
    param = Parameters()
    if use_lightsim:
        if light_sim_avail:
            class ThisBackendClass(AddDetailedTopoIEEE, LightSimBackend):
                pass
        else:
            raise RuntimeError("LightSimBackend not available")
    else:
        class ThisBackendClass(AddDetailedTopoIEEE, PandaPowerBackend):
            pass
        
    backend = ThisBackendClass()

    param.init_from_dict(
        {"NO_OVERFLOW_DISCONNECTION": True,
         "STOP_EP_IF_GEN_BREAK_CONSTRAINTS": False,
         "ENV_DOES_REDISPATCHING": False})

    env_klu = make(name,
                   backend=backend,
                   param=param,
                   gamerules_class=AlwaysLegal,
                   test=test_env,
                   action_class=CompleteAction)
    assert type(env_klu).detailed_topo_desc is not None
    agent = RandomBBsSwitchAgent(action_space=env_klu.action_space)

    cp = cProfile.Profile()
    cp.enable()
    nb_ts_klu, time_klu, aor_klu, gen_p_klu, gen_q_klu, time_step = run_env(env_klu, max_ts, agent)
    cp.disable()
    print(f'Time for {nb_ts_klu} steps: {time_step} => {time_step / nb_ts_klu} s/step or {nb_ts_klu / time_step:.3e} step / s')
    nm_f, ext = os.path.splitext(__file__)
    nm_out = "{}_{}_{}.prof".format(nm_f, "lightsim" if use_ls else "pp", name)
    cp.dump_stats(nm_out)
    print("You can view profiling results with:\n\tsnakeviz {}".format(nm_out))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark pyKLU and Pandapower Backend for an agent that takes every '
                                                 'topological action possible')
    parser.add_argument('--name', default=ENV_NAME, type=str,
                        help='Environment name to be used for the benchmark.')
    parser.add_argument('--number', type=int, default=MAX_TS,
                        help='Maximum number of time steps for which the benchamark will be run.')
    parser.add_argument("--use_ls", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use the LightSim2Grid Backend.")
    parser.add_argument("--no_test", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Do not use a test environment for the profiling (default to False: meaning you use a test env)")

    args = parser.parse_args()

    max_ts = int(args.number)
    name = str(args.name)
    use_ls = args.use_ls
    test_env = not args.no_test
    main(max_ts, name, use_lightsim=use_ls, test_env=test_env)
