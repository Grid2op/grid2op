# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from abc import ABC, abstractmethod
from typing import Tuple, Union

from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from grid2op.typing_variables import STEP_INFO_TYPING, RESET_OPTIONS_TYPING


class EnvInterface(ABC):
    """
    This is an interface for Grid2op environments designed to ensure that all implementations (except for multi-environments,
    which have the same methods but slightly different signatures) define the minimum methods required to interact with
    an environment.
    """
    @abstractmethod
    def reset(self,
              *,
              seed: Union[int, None] = None,
              options: RESET_OPTIONS_TYPING = None) -> BaseObservation:
        pass

    @abstractmethod
    def step(self, action: BaseAction) -> Tuple[BaseObservation,
                                                float,
                                                bool,
                                                STEP_INFO_TYPING]:
        """
                Run one timestep of the environment's dynamics. When end of
                episode is reached, you are responsible for calling `reset()`
                to reset this environment's state.
                Accepts an action and returns a tuple (observation, reward, done, info).

                If the :class:`grid2op.BaseAction.BaseAction` is illegal or ambiguous, the step is performed, but the action is
                replaced with a "do nothing" action.

                Parameters
                ----------
                    action: :class:`grid2op.Action.Action`
                        an action provided by the agent that is applied on the underlying through the backend.

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
                              due to overflow (if not disconnected it will be -1, otherwise it will be a
                              positive integer: 0 meaning that is one of the cause of the cascading failure, 1 means
                              that it is disconnected just after, 2 that it's disconnected just after etc.)
                            - "is_illegal" (``bool``) whether the action given as input was illegal
                            - "is_ambiguous" (``bool``) whether the action given as input was ambiguous.
                            - "is_dispatching_illegal" (``bool``) was the action illegal due to redispatching
                            - "is_illegal_reco" (``bool``) was the action illegal due to a powerline reconnection
                            - "reason_alarm_illegal" (``None`` or ``Exception``) reason for which the alarm is illegal
                              (it's None if no alarm are raised or if the alarm feature is not used)
                            - "reason_alert_illegal" (``None`` or ``Exception``) reason for which the alert is illegal
                              (it's None if no alert are raised or if the alert feature is not used)
                            - "opponent_attack_line" (``np.ndarray``, ``bool``) for each powerline, say if the opponent
                              attacked it (``True``) or not (``False``).
                            - "opponent_attack_sub" (``np.ndarray``, ``bool``) for each substation, say if the opponent
                              attacked it (``True``) or not (``False``).
                            - "opponent_attack_duration" (``int``) the duration of the current attack (if any)
                            - "exception" (``list`` of :class:`Exceptions.Exceptions.Grid2OpException` if an exception was
                              raised  or ``[]`` if everything was fine.)
                            - "detailed_infos_for_cascading_failures" (optional, only if the backend has been create with
                              `detailed_infos_for_cascading_failures=True`) the list of the intermediate steps computed during
                              the simulation of the "cascading failures".
                            - "rewards": dictionary of all "other_rewards" provided when the env was built.
                            - "time_series_id": id of the time series used (if any, similar to a call to `env.chronics_handler.get_id()`)
                Examples
                ---------

                This is used like:

                .. code-block:: python

                    import grid2op
                    from grid2op.Agent import RandomAgent

                    # I create an environment
                    env = grid2op.make("l2rpn_case14_sandbox")

                    # define an agent here, this is an example
                    agent = RandomAgent(env.action_space)

                    # environment need to be "reset" before usage:
                    obs = env.reset()
                    reward = env.reward_range[0]
                    done = False

                    # now run through each steps like this
                    while not done:
                        action = agent.act(obs, reward, done)
                        obs, reward, done, info = env.step(action)

                Notes
                -----

                If the flag `done=True` is raised (*ie* this is the end of the episode) then the observation is NOT properly
                updated and should not be used at all.

                Actually, it will be in a "game over" state (see :class:`grid2op.Observation.BaseObservation.set_game_over`).

                """
        pass

    def render(self, mode="rgb_array"):
        """
        Render the state of the environment on the screen, using matplotlib
        Also returns the Matplotlib figure

        Examples
        --------
        Rendering need first to define a "renderer" which can be done with the following code:

        .. code-block:: python

            import grid2op

            # create the environment
            env = grid2op.make("l2rpn_case14_sandbox")

            # if you want to use the renderer
            env.attach_renderer()

            # and now you can "render" (plot) the state of the grid
            obs = env.reset()
            done = False
            reward = env.reward_range[0]
            while not done:
                env.render()  # this piece of code plot the grid
                action = agent.act(obs, reward, done)
                obs, reward, done, info = env.step(action)
        """
        pass

    def close(self):
        """close an environment: this will attempt to free as much memory as possible.
        Note that after an environment is closed, you will not be able to use anymore.

        Any attempt to use a closed environment might result in non deterministic behaviour.
        """
        pass

    def __enter__(self):
        """
        Support *with-statement* for the environment.

        Examples
        --------

        .. code-block:: python

            import grid2op
            import grid2op.BaseAgent
            with grid2op.make("l2rpn_case14_sandbox") as env:
                agent = grid2op.BaseAgent.DoNothingAgent(env.action_space)
                act = env.action_space()
                obs, r, done, info = env.step(act)
                act = agent.act(obs, r, info)
                obs, r, done, info = env.step(act)

        """
        return self

    def __exit__(self, *args):
        """
        Support *with-statement* for the environment.
        """
        self.close()
        # propagate exception
        return False
