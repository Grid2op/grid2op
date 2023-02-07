# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
import warnings
from grid2op.Exceptions.EnvExceptions import EnvError

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Environment.BaseEnv import BaseEnv
from grid2op.Chronics import ChangeNothing
from grid2op.Rules import RulesChecker, BaseRules
from grid2op.Exceptions import Grid2OpException
from grid2op.operator_attention import LinearAttentionBudget


class _ObsCH(ChangeNothing):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This class is reserved to internal use. Do not attempt to do anything with it.
    """

    def forecasts(self):
        return []


class _ObsEnv(BaseEnv):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This class is an 'Emulator' of a :class:`grid2op.Environment.Environment` used to be able to 'simulate'
    forecasted grid states.
    It should not be used outside of an :class:`grid2op.Observation.BaseObservation` instance, or one of its derivative.

    It contains only the most basic element of an Environment. See :class:`grid2op.Environment.Environment` for more
    details.

    This class is reserved for internal use. Do not attempt to do anything with it.
    """

    def __init__(
        self,
        init_env_path,
        init_grid_path,
        backend_instanciated,
        parameters,
        reward_helper,
        obsClass,  # not initialized :-/
        action_helper,
        thermal_limit_a,
        legalActClass,
        helper_action_class,
        helper_action_env,
        epsilon_poly,
        tol_poly,
        max_episode_duration,
        delta_time_seconds,
        other_rewards={},
        has_attention_budget=False,
        attention_budget_cls=LinearAttentionBudget,
        kwargs_attention_budget={},
        logger=None,
        _complete_action_cls=None,
        _ptr_orig_obs_space=None,
    ):
        BaseEnv.__init__(
            self,
            init_env_path,
            init_grid_path,
            copy.deepcopy(parameters),
            thermal_limit_a,
            other_rewards=other_rewards,
            epsilon_poly=epsilon_poly,
            tol_poly=tol_poly,
            has_attention_budget=has_attention_budget,
            attention_budget_cls=attention_budget_cls,
            kwargs_attention_budget=kwargs_attention_budget,
            kwargs_observation=None,
            logger=logger,
        )
        self.__unusable = False  # unsuable if backend cannot be copied
        
        self._reward_helper = reward_helper
        self._helper_action_class = helper_action_class

        # initialize the observation space
        self._obsClass = None

        # self.gen_activeprod_t_init = np.zeros(self.n_gen, dtype=dt_float)
        # self.gen_activeprod_t_redisp_init = np.zeros(self.n_gen, dtype=dt_float)
        # self.times_before_line_status_actionable_init = np.zeros(
        #     self.n_line, dtype=dt_int
        # )
        # self.times_before_topology_actionable_init = np.zeros(self.n_sub, dtype=dt_int)
        # self.time_next_maintenance_init = np.zeros(self.n_line, dtype=dt_int)
        # self.duration_next_maintenance_init = np.zeros(self.n_line, dtype=dt_int)
        # self.target_dispatch_init = np.zeros(self.n_gen, dtype=dt_float)
        # self.actual_dispatch_init = np.zeros(self.n_gen, dtype=dt_float)
        # self._already_modified_gen_init = np.zeros(self.n_gen, dtype=dt_float)

        # line status (inherited from BaseEnv)
        self._line_status = np.full(self.n_line, dtype=dt_bool, fill_value=True)
        # line status (for this usage)
        self._line_status_me = np.ones(
            shape=self.n_line, dtype=dt_int
        )  # this is "line status" but encode in +1 / -1
        # self._line_status_orig = np.ones(shape=self.n_line, dtype=dt_int)

        if self._thermal_limit_a is None:
            self._thermal_limit_a = 1.0 * thermal_limit_a.astype(dt_float)
        else:
            self._thermal_limit_a[:] = thermal_limit_a
        
        self._init_backend(
            chronics_handler=_ObsCH(),
            backend=backend_instanciated,
            names_chronics_to_backend=None,
            actionClass=action_helper.actionClass,
            observationClass=obsClass,
            rewardClass=None,
            legalActClass=legalActClass,
        )

        ####
        # to be able to save and import (using env.generate_classes) correctly
        self._actionClass = action_helper.subtype
        self._observationClass = _complete_action_cls  # not used anyway
        self._complete_action_cls = _complete_action_cls
        self._action_space = (
            action_helper  # obs env and env share the same action space
        )
        self._observation_space = (
            action_helper  # not used here, so it's definitely a hack !
        )
        self._ptr_orig_obs_space = _ptr_orig_obs_space
        ####

        self.no_overflow_disconnection = parameters.NO_OVERFLOW_DISCONNECTION

        # self._load_p, self._load_q, self._load_v = None, None, None
        # self._prod_p, self._prod_q, self._prod_v = None, None, None
        self._topo_vect = np.zeros(type(backend_instanciated).dim_topo, dtype=dt_int)

        # other stuff
        self.is_init = False
        self._helper_action_env = helper_action_env
        self.env_modification = self._helper_action_env()
        self._do_nothing_act = self._helper_action_env()
        
        if self.__unusable:
            self._backend_action_set = None
        else:
            self._backend_action_set = self._backend_action_class()

        self.delta_time_seconds = delta_time_seconds
        
        # opponent
        # self.opp_space_state = None
        # self.opp_state = None

        # storage
        # self._storage_current_charge_init = None
        # self._storage_previous_charge_init = None
        # self._action_storage_init = None
        # self._amount_storage_init = None
        # self._amount_storage_prev_init = None
        # self._storage_power_init = None

        # # storage unit
        # if self.__unusable:
        #     self._storage_current_charge_init = np.zeros(0, dtype=dt_float)
        #     self._storage_previous_charge_init = np.zeros(0, dtype=dt_float)
        #     self._action_storage_init = np.zeros(0, dtype=dt_float)
        #     self._storage_power_init = np.zeros(0, dtype=dt_float)
        # else:
        #     self._storage_current_charge_init = np.zeros(self.n_storage, dtype=dt_float)
        #     self._storage_previous_charge_init = np.zeros(self.n_storage, dtype=dt_float)
        #     self._action_storage_init = np.zeros(self.n_storage, dtype=dt_float)
        #     self._storage_power_init = np.zeros(self.n_storage, dtype=dt_float)
            
        # self._amount_storage_init = 0.0
        # self._amount_storage_prev_init = 0.0
            

        # # curtailment
        # if self.__unusable:
        #     self._limit_curtailment_init = np.zeros(0, dtype=dt_float)
        #     self._gen_before_curtailment_init = np.zeros(0, dtype=dt_float)
        # else:
        #     self._limit_curtailment_init = np.zeros(self.n_gen, dtype=dt_float)
        #     self._gen_before_curtailment_init = np.zeros(self.n_gen, dtype=dt_float)
        # self._sum_curtailment_mw_init = 0.0
        # self._sum_curtailment_mw_prev_init = 0.0

        # # step count
        # self._nb_time_step_init = 0

        # # alarm / attention budget
        # self._attention_budget_state_init = None

        if self.__unusable:
            self._disc_lines = np.zeros(shape=0, dtype=dt_int) - 1
        else:
            self._disc_lines = np.zeros(shape=self.n_line, dtype=dt_int) - 1
            
        self._max_episode_duration = max_episode_duration

    def max_episode_duration(self):
        return self._max_episode_duration

    def _init_myclass(self):
        """this class has already all the powergrid information: it is initialized in the obs space !"""
        pass

    def _init_backend(
        self,
        chronics_handler,
        backend,
        names_chronics_to_backend,
        actionClass,
        observationClass,  # base grid2op type
        rewardClass,
        legalActClass,
    ):
        if backend is None:
            self.__unusable = True
            return
        
        self.__unusable = False
        self._env_dc = self.parameters.ENV_DC
        self.chronics_handler = chronics_handler
        self.backend = backend
        self._has_been_initialized()  # really important to include this piece of code! and just here after the

        if not issubclass(legalActClass, BaseRules):
            raise Grid2OpException(
                'Parameter "legalActClass" used to build the Environment should derived form the '
                'grid2op.BaseRules class, type provided is "{}"'.format(
                    type(legalActClass)
                )
            )
        self._game_rules = RulesChecker(legalActClass=legalActClass)
        self._legalActClass = legalActClass
        # self._action_space = self._do_nothing
        self.backend.set_thermal_limit(self._thermal_limit_a)

        # create the opponent
        self._create_opponent()

        # create the attention budget
        self._create_attention_budget()
        self._obsClass = observationClass.init_grid(type(self.backend))
        self._obsClass._INIT_GRID_CLS = observationClass
        self.current_obs_init = self._obsClass(obs_env=None, action_helper=None)
        self.current_obs = self.current_obs_init

        # backend has loaded everything
        self._hazard_duration = np.zeros(shape=self.n_line, dtype=dt_int)

    def _do_nothing(self, x):
        """
        this is should be only called within _Obsenv.step, and there, only return the "do nothing"
        action.

        This is why this function is used as the "obsenv action space"
        """
        return self._do_nothing_act

    def _update_actions(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Retrieve the actions to perform the update of the underlying powergrid represented by
         the :class:`grid2op.Backend`in the next time step.
        A call to this function will also read the next state of :attr:`chronics_handler`, so it must be called only
        once per time step.

        Returns
        --------
        res: :class:`grid2op.Action.Action`
            The action representing the modification of the powergrid induced by the Backend.
        """
        # TODO consider disconnecting maintenance forecasted :-)
        # This "environment" doesn't modify anything
        return self._do_nothing_act, None

    def copy(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Implement the deep copy of this instance.

        Returns
        -------
        res: :class:`ObsEnv`
            A deep copy of this instance.
        """
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        backend = self.backend
        self.backend = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            res = copy.deepcopy(self)
            res.backend = backend.copy()
        self.backend = backend
        return res

    def init(
        self,
        new_state_action,
        time_stamp,
        obs,
        time_step=1
    ):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Initialize a "forecasted grid state" based on the new injections, possibly new topological modifications etc.

        Parameters
        ----------
        new_state_action: :class:`grid2op.Action`
            The action that is performed on the powergrid to get the forecast at the current date. This "action" is
            NOT performed by the user, it's performed internally by the BaseObservation to have a "forecasted" powergrid
            with the forecasted values present in the chronics.

        time_stamp: ``datetime.datetime``
            The time stamp of the forecast, as a datetime.datetime object. NB this is not the time stamp at which the
            forecast is produced, but the time stamp of the powergrid forecasted.

        timestep_overflow: ``numpy.ndarray``
            The see :attr:`grid2op.Env.timestep_overflow` for a better description of this argument.

        Returns
        -------
        ``None``

        """
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        
        self._reset_to_orig_state(obs)
        self._topo_vect[:] = obs.topo_vect
        (
            still_in_maintenance,
            reconnected,
            first_ts_maintenance,
        ) = self._update_vector_with_timestep(time_step)
        if np.any(first_ts_maintenance):
            set_status = np.array(self._line_status_me, dtype=dt_int)
            set_status[first_ts_maintenance] = -1
            topo_vect = np.array(self._topo_vect, dtype=dt_int)
            topo_vect[self.line_or_pos_topo_vect[first_ts_maintenance]] = -1
            topo_vect[self.line_ex_pos_topo_vect[first_ts_maintenance]] = -1
        else:
            set_status = self._line_status_me
            topo_vect = self._topo_vect
        
        if np.any(still_in_maintenance):
            set_status[still_in_maintenance] = -1
            topo_vect = np.array(self._topo_vect, dtype=dt_int)
            topo_vect[self.line_or_pos_topo_vect[still_in_maintenance]] = -1
            topo_vect[self.line_ex_pos_topo_vect[still_in_maintenance]] = -1
            
        # TODO set the shunts here
        # update the action that set the grid to the real value
        self._backend_action_set += self._helper_action_env(
            {
                "set_line_status": set_status,
                "set_bus": topo_vect,
                "injection": {
                    "prod_p": obs.gen_p,
                    "prod_v": obs.gen_v,
                    "load_p": obs.load_p,
                    "load_q": obs.load_q,
                },
            }
        )
        self._backend_action_set += new_state_action
        # for storage unit
        self._backend_action_set.storage_power.values[:] = 0.0
        self._backend_action_set.all_changed()
        self._backend_action = copy.deepcopy(self._backend_action_set)
        
        # for curtailment
        if self._env_modification is not None:
            self._env_modification._dict_inj = {}
        
        self.is_init = True
        self.current_obs.reset()
        self.time_stamp = time_stamp
        
        # TODO here !
        self._timestep_overflow[:] = obs.timestep_overflow

    def _get_new_prod_setpoint(self, action):
        new_p = 1.0 * self._backend_action_set.prod_p.values
        if "prod_p" in action._dict_inj:
            tmp = action._dict_inj["prod_p"]
            indx_ok = np.isfinite(tmp)
            new_p[indx_ok] = tmp[indx_ok]

        # modification of the environment always override the modification of the agents (if any)
        # TODO have a flag there if this is the case.
        if "prod_p" in self._env_modification._dict_inj:
            # modification of the production setpoint value
            tmp = self._env_modification._dict_inj["prod_p"]
            indx_ok = np.isfinite(tmp)
            new_p[indx_ok] = tmp[indx_ok]
        return new_p

    def _update_vector_with_timestep(self, time_step):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        update the value of the "time dependant" attributes
        """
        cls = type(self)
        
        # update the cooldowns
        self._times_before_line_status_actionable[:] = np.maximum(
            self._times_before_line_status_actionable - (time_step - 1), 0
        )
        self._times_before_topology_actionable[:] = np.maximum(
            self._times_before_topology_actionable - (time_step - 1), 0
        )

        # update the maintenance
        tnm_orig = 1 * self._time_next_maintenance
        dnm_orig = 1 * self._duration_next_maintenance
        
        has_maint = self._time_next_maintenance != -1
        reconnected = np.full(cls.n_line, fill_value=False)
        maint_started = np.full(cls.n_line, fill_value=False)
        maint_over = np.full(cls.n_line, fill_value=False)
        
        maint_started[has_maint] = (tnm_orig[has_maint] <= time_step)
        maint_over[has_maint] = (tnm_orig[has_maint] + dnm_orig[has_maint] 
                                 <= time_step)
        
        reconnected[has_maint] = tnm_orig[has_maint] + dnm_orig[has_maint] == time_step
        first_ts_maintenance = tnm_orig == time_step
        still_in_maintenance = maint_started & (~maint_over) & (~first_ts_maintenance)
        
        if hasattr(self, "_DEBUG") and self._DEBUG: 
            import pdb
            pdb.set_trace()
            
        # count down time next maintenance
        self._time_next_maintenance[:] = np.maximum(
            self._time_next_maintenance - time_step, -1
        )
        
        # powerline that are still in maintenance at this time step
        self._time_next_maintenance[still_in_maintenance] = 0
        self._duration_next_maintenance[still_in_maintenance] -= (time_step - tnm_orig[still_in_maintenance])

        # powerline that will be in maintenance at this time step
        self._time_next_maintenance[first_ts_maintenance] = 0
        # self._duration_next_maintenance[first_ts_maintenance] -= time_step
        
        # powerline that will be in maintenance at this time step
        self._time_next_maintenance[reconnected | maint_over] = -1
        self._duration_next_maintenance[reconnected | maint_over] = 0

        return still_in_maintenance, reconnected, first_ts_maintenance

    def reset(self):
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        super().reset()
        self.current_obs = self.current_obs_init

    def _reset_to_orig_state(self, obs):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        reset this "environment" to the state it should be
        """
        self.reset()  # reset the "BaseEnv"
        self.backend.set_thermal_limit(obs._thermal_limit)
        self._backend_action_set.all_changed()
        self._backend_action = copy.deepcopy(self._backend_action_set)
        self._oppSpace._set_state(obs._env_internal_params["opp_space_state"], 
                                  obs._env_internal_params["opp_state"])
        # storage unit
        self._storage_current_charge[:] = obs.storage_charge
        self._storage_previous_charge[:] = obs._env_internal_params["_storage_previous_charge"]
        self._action_storage[:] = obs.storage_power_target
        self._storage_power[:] = obs.storage_power
        self._amount_storage = obs._env_internal_params["_amount_storage"]
        self._amount_storage_prev = obs._env_internal_params["_amount_storage_prev"]

        # curtailment
        self._limit_curtailment[:] = obs.curtailment_limit
        self._gen_before_curtailment[:] = obs.gen_p_before_curtail
        self._sum_curtailment_mw = obs._env_internal_params["_sum_curtailment_mw"]
        self._sum_curtailment_mw_prev = obs._env_internal_params["_sum_curtailment_mw_prev"]

        # line status
        self._line_status[:] = obs._env_internal_params["_line_status_env"] == 1
        self._line_status_me[:] = obs._env_internal_params["_line_status_env"]

        # attention budget
        if self._has_attention_budget:
            self._attention_budget.set_state(obs._env_internal_params["_attention_budget_state"])
            
        self._gen_activeprod_t[:] = obs._env_internal_params["_gen_activeprod_t"]
        self._gen_activeprod_t_redisp[:] = obs._env_internal_params["_gen_activeprod_t_redisp"]
        
        # cooldown
        self._times_before_line_status_actionable[
            :
        ] = obs.time_before_cooldown_line
        self._times_before_topology_actionable[
            :
        ] = obs.time_before_cooldown_sub
        
        # maintenance
        self._time_next_maintenance[:] = obs.time_next_maintenance
        self._duration_next_maintenance[:] = obs.duration_next_maintenance
        
        # redisp
        self._target_dispatch[:] = obs.target_dispatch
        self._actual_dispatch[:] = obs.actual_dispatch
        self._already_modified_gen[:] = obs._env_internal_params["_already_modified_gen"]
        
        # current step
        self.nb_time_step = obs.current_step
        self.delta_time_seconds = 60. * obs.delta_time

    def simulate(self, action):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Prefer using `obs.simulate(action)`

        This function is the core method of the :class:`ObsEnv`. It allows to perform a simulation of what would
        give and action if it were to be implemented on the "forecasted" powergrid.

        It has the same signature as :func:`grid2op.Environment.Environment.step`. One of the major difference is that
        it doesn't
        check whether the action is illegal or not (but an implementation could be provided for this method). The
        reason for this is that there is not one single and unique way to "forecast" how the thermal limit will behave,
        which lines will be available or not, which actions will be done or not between the time stamp at which
        "simulate" is called, and the time stamp that is simulated.

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action to test

        Returns
        -------
        observation: :class:`grid2op.Observation.Observation`
            agent's observation of the current environment

        reward: ``float``
            amount of reward returned after previous action

        done: ``bool``
            whether the episode has ended, in which case further step() calls will return undefined results

        info: ``dict``
            contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). It is a
            dictionary with keys:

                - "disc_lines": a numpy array (or ``None``) saying, for each powerline if it has been disconnected
                    due to overflow
                - "is_illegal" (``bool``) whether the action given as input was illegal
                - "is_ambiguous" (``bool``) whether the action given as input was ambiguous.

        """
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        self._ptr_orig_obs_space.simulate_called()
        maybe_exc = self._ptr_orig_obs_space.can_use_simulate()
        if maybe_exc is not None:
            raise maybe_exc
        
        obs, reward, done, info = self.step(action)
        return obs, reward, done, info

    def get_obs(self, _update_state=True):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Method to retrieve the "forecasted grid" as a valid observation object.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The observation available.
        """
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        if _update_state:
            self.current_obs.update(self, with_forecast=False)
        res = self.current_obs.copy()
        return res

    def update_grid(self, env):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Update this "emulated" environment with the real powergrid.

        # TODO it should be updated from the observation only, especially if the observation is partially
        # TODO observable. This would lead to data leakage here somehow.

        Parameters
        ----------
        env: :class:`grid2op.Environment.BaseEnv`
            A reference to the environment
        """
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        self.is_init = False
        
        # real_backend = env.backend

        # self._load_p, self._load_q, self._load_v = real_backend.loads_info()
        # self._prod_p, self._prod_q, self._prod_v = real_backend.generators_info()
        # self._topo_vect = real_backend.get_topo_vect()

        # convert line status to -1 / 1 instead of false / true
        # self._line_status_orig[:] = env.get_current_line_status().astype(
        #     dt_int
        # )  # false -> 0 true -> 1
        # self._line_status_orig *= 2  # false -> 0 true -> 2
        # self._line_status_orig -= 1  # false -> -1; true -> 1
        # self.is_init = False

        # Make a copy of env state for simulation
        # TODO this depends on the datetime simulated, so find a way to have it independant of that !!!
        # if self._thermal_limit_a is None:
        #     self._thermal_limit_a = 1.0 * env._thermal_limit_a.astype(dt_float)
        # else:
        #     self._thermal_limit_a[:] = env._thermal_limit_a.astype(dt_float)
        # self.gen_activeprod_t_init[:] = env._gen_activeprod_t
        # self.gen_activeprod_t_redisp_init[:] = env._gen_activeprod_t_redisp
        # self.times_before_line_status_actionable_init[
        #     :
        # ] = env._times_before_line_status_actionable
        # self.times_before_topology_actionable_init[
        #     :
        # ] = env._times_before_topology_actionable
        # self.time_next_maintenance_init[:] = env._time_next_maintenance
        # self.duration_next_maintenance_init[:] =  env._duration_next_maintenance
        # self.target_dispatch_init[:] = env._target_dispatch
        # self.actual_dispatch_init[:] = env._actual_dispatch
        # self._already_modified_gen_init[:] = env._already_modified_gen
        # self.opp_space_state, self.opp_state = env._oppSpace._get_state()

        # storage units
        # TODO this is not time independant... i set up the previous charge of the obs env to be
        # set current charge of the simulated env on purpose
        # self._storage_current_charge_init[:] = env._storage_current_charge
        # self._storage_previous_charge_init[:] = env._storage_previous_charge
        # self._action_storage_init[:] = env._action_storage
        # self._amount_storage_init = 1.0 * env._amount_storage
        # self._amount_storage_prev_init = 1.0 * env._amount_storage_prev
        # self._storage_power_init[:] = env._storage_power

        # curtailment
        # self._limit_curtailment_init[:] = env._limit_curtailment
        # self._gen_before_curtailment_init[:] = env._gen_before_curtailment
        # self._sum_curtailment_mw_init = 1.0 * env._sum_curtailment_mw
        # self._sum_curtailment_mw_prev_init = 1.0 * env._sum_curtailment_mw_prev

        # # time delta
        # self.delta_time_seconds = env.delta_time_seconds

        # # current time
        # self._nb_time_step_init = env.nb_time_step

        # # attention budget
        # if self._has_attention_budget:
        #     self._attention_budget_state_init = env._attention_budget.get_state()

    def get_current_line_status(self):
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        return self._line_status == 1

    def is_valid(self):
        """return whether or not the obs_env is valid, *eg* whether
        we could copy the backend of the environment."""
        return not self.__unusable
    
    def close(self):
        """close this environment, once and for all"""
        super().close()

        # clean all the attributes
        for attr_nm in [
            "_obsClass",
            # "gen_activeprod_t_init",
            # "gen_activeprod_t_redisp_init",
            # "times_before_line_status_actionable_init",
            # "times_before_topology_actionable_init",
            # "time_next_maintenance_init",
            # "duration_next_maintenance_init",
            # "target_dispatch_init",
            "_line_status",
            "_line_status_me",
            # "_line_status_orig",
            # "_load_p",
            # "_load_q",
            # "_load_v",
            # "_prod_p",
            # "_prod_q",
            # "_prod_v",
            # "_topo_vect",
            # "opp_space_state",
            # "opp_state",
            # "_storage_current_charge_init",
            # "_storage_previous_charge_init",
            # "_action_storage_init",
            # "_amount_storage_init",
            # "_amount_storage_prev_init",
            # "_storage_power_init",
            # "_storage_current_charge_init",
            # "_storage_previous_charge_init",
            # "_limit_curtailment_init",
            # "_gen_before_curtailment_init",
            # "_sum_curtailment_mw_init",
            # "_sum_curtailment_mw_prev_init",
            # "_nb_time_step_init",
            # "_attention_budget_state_init",
            "_max_episode_duration",
            "_ptr_orig_obs_space",
        ]:
            if hasattr(self, attr_nm):
                delattr(self, attr_nm)
            setattr(self, attr_nm, None)
