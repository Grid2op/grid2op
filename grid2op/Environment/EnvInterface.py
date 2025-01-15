from abc import ABC, abstractmethod
from typing import Tuple, Union

from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from grid2op.typing_variables import STEP_INFO_TYPING, RESET_OPTIONS_TYPING


class EnvInterface(ABC):

    @abstractmethod
    def reset(self,
              *,
              seed: Union[int, None] = None,
              options: RESET_OPTIONS_TYPING = None) -> BaseObservation:
        """
        Reset the environment to a clean state.
        It will reload the next chronics if any. And reset the grid to a clean state.

        This triggers a full reloading of both the chronics (if they are stored as files) and of the powergrid,
        to ensure the episode is fully over.

        This method should be called only at the end of an episode.

        Parameters
        ----------
        seed: int
            The seed to used (new in version 1.9.8), see examples for more details. Ignored if not set (meaning no seeds will
            be used, experiments might not be reproducible)

        options: dict
            Some options to "customize" the reset call. For example (see detailed example bellow) :

            - "time serie id" (grid2op >= 1.9.8) to use a given time serie from the input data
            - "init state" that allows you to apply a given "action" when generating the
              initial observation (grid2op >= 1.10.2)
            - "init ts" (grid2op >= 1.10.3) to specify to which "steps" of the time series
              the episode will start
            - "max step" (grid2op >= 1.10.3) : maximum number of steps allowed for the episode
            - "thermal limit" (grid2op >= 1.11.0): which thermal limit to use for this episode
              (and the next ones, until they are changed)
            - "init datetime": which time stamp is used in the first observation of the episode.

            See examples for more information about this. Ignored if
            not set.

        Examples
        --------
        The standard "gym loop" can be done with the following code:

        .. code-block:: python

            import grid2op

            # create the environment
            env_name = "l2rpn_case14_sandbox"
            env = grid2op.make(env_name)

            # start a new episode
            obs = env.reset()
            done = False
            reward = env.reward_range[0]
            while not done:
                action = agent.act(obs, reward, done)
                obs, reward, done, info = env.step(action)

        .. versionadded:: 1.9.8
            It is now possible to set the seed and the time series you want to use at the new
            episode by calling `env.reset(seed=..., options={"time serie id": ...})`

        Before version 1.9.8, if you wanted to use a fixed seed, you would need to (see
        doc of :func:`grid2op.Environment.BaseEnv.seed` ):

        .. code-block:: python

            seed = ...
            env.seed(seed)
            obs = env.reset()
            ...

        Starting from version 1.9.8 you can do this in one call:

        .. code-block:: python

            seed = ...
            obs = env.reset(seed=seed)

        For the "time series id" it is the same concept. Before you would need to do (see
        doc of :func:`Environment.set_id` for more information ):

        .. code-block:: python

            time_serie_id = ...
            env.set_id(time_serie_id)
            obs = env.reset()
            ...

        And now (from version 1.9.8) you can more simply do:

        .. code-block:: python

            time_serie_id = ...
            obs = env.reset(options={"time serie id": time_serie_id})
            ...

        .. versionadded:: 1.10.2

        Another feature has been added in version 1.10.2, which is the possibility to set the
        grid to a given "topological" state at the first observation (before this version,
        you could only retrieve an observation with everything connected together).

        In grid2op 1.10.2, you can do that by using the keys `"init state"` in the "options" kwargs of
        the reset function. The value associated to this key should be dictionnary that can be
        converted to a non ambiguous grid2op action using an "action space".

        .. note::
            The "action space" used here is not the action space of the agent. It's an "action
            space" that uses a :func:`grid2op.Action.Action.BaseAction` class meaning you can do any
            type of action, on shunts, on topology, on line status etc. even if the agent is not
            allowed to.

            Likewise, nothing check if this action is legal or not.

        You can use it like this:

        .. code-block:: python

            # to start an episode with a line disconnected, you can do:
            init_state_dict = {"set_line_status": [(0, -1)]}
            obs = env.reset(options={"init state": init_state_dict})
            obs.line_status[0] is False

            # to start an episode with a different topolovy
            init_state_dict = {"set_bus": {"lines_or_id": [(0, 2)], "lines_ex_id": [(3, 2)]}}
            obs = env.reset(options={"init state": init_state_dict})

        .. note::
            Since grid2op version 1.10.2, there is also the possibility to set the "initial state"
            of the grid directly in the time series. The priority is always given to the
            argument passed in the "options" value.

            Concretely if, in the "time series" (formelly called "chronics") provides an action would change
            the topology of substation 1 and 2 (for example) and you provide an action that disable the
            line 6, then the initial state will see substation 1 and 2 changed (as in the time series)
            and line 6 disconnected.

            Another example in this case: if the action you provide would change topology of substation 2 and 4
            then the initial state (after `env.reset`) will give:

            - substation 1 as in the time serie
            - substation 2 as in "options"
            - substation 4 as in "options"

        .. note::
            Concerning the previously described behaviour, if you want to ignore the data in the
            time series, you can add : `"method": "ignore"` in the dictionary describing the action.
            In this case the action in the time series will be totally ignored and the initial
            state will be fully set by the action passed in the "options" dict.

            An example is:

            .. code-block:: python

                init_state_dict = {"set_line_status": [(0, -1)], "method": "force"}
                obs = env.reset(options={"init state": init_state_dict})
                obs.line_status[0] is False

        .. versionadded:: 1.10.3

        Another feature has been added in version 1.10.3, the possibility to skip the
        some steps of the time series and starts at some given steps.

        The time series often always start at a given day of the week (*eg* Monday)
        and at a given time (*eg* midnight). But for some reason you notice that your
        agent performs poorly on other day of the week or time of the day. This might be
        because it has seen much more data from Monday at midnight that from any other
        day and hour of the day.

        To alleviate this issue, you can now easily reset an episode and ask grid2op
        to start this episode after xxx steps have "passed".

        Concretely, you can do it with:

        .. code-block:: python

            import grid2op
            env_name = "l2rpn_case14_sandbox"
            env = grid2op.make(env_name)

            obs = env.reset(options={"init ts": 1})

        Doing that your agent will start its episode not at midnight (which
        is the case for this environment), but at 00:05

        If you do:

        .. code-block:: python

            obs = env.reset(options={"init ts": 12})

        In this case, you start the episode at 01:00 and not at midnight (you
        start at what would have been the 12th steps)

        If you want to start the "next day", you can do:

        .. code-block:: python

            obs = env.reset(options={"init ts": 288})

        etc.

        .. note::
            On this feature, if a powerline is on soft overflow (meaning its flow is above
            the limit but below the :attr:`grid2op.Parameters.Parameters.HARD_OVERFLOW_THRESHOLD` * `the limit`)
            then it is still connected (of course) and the counter
            :attr:`grid2op.Observation.BaseObservation.timestep_overflow` is at 0.

            If a powerline is on "hard overflow" (meaning its flow would be above
            :attr:`grid2op.Parameters.Parameters.HARD_OVERFLOW_THRESHOLD` * `the limit`), then, as it is
            the case for a "normal" (without options) reset, this line is disconnected, but can be reconnected
            directly (:attr:`grid2op.Observation.BaseObservation.time_before_cooldown_line` == 0)

        .. seealso::
            The function :func:`Environment.fast_forward_chronics` for an alternative usage (that will be
            deprecated at some point)

        Yet another feature has been added in grid2op version 1.10.3 in this `env.reset` function. It is
        the capacity to limit the duration of an episode.

        .. code-block:: python

            import grid2op
            env_name = "l2rpn_case14_sandbox"
            env = grid2op.make(env_name)

            obs = env.reset(options={"max step": 288})

        This will limit the duration to 288 steps (1 day), meaning your agent
        will have successfully managed the entire episode if it manages to keep
        the grid in a safe state for a whole day (depending on the environment you are
        using the default duration is either one week - roughly 2016 steps or 4 weeks)

        .. note::
            This option only affect the current episode. It will have no impact on the
            next episode (after reset)

        For example:

        .. code-block:: python

            obs = env.reset()
            obs.max_step == 8064  # default for this environment

            obs = env.reset(options={"max step": 288})
            obs.max_step == 288  # specified by the option

            obs = env.reset()
            obs.max_step == 8064  # retrieve the default behaviour

        .. seealso::
            The function :func:`Environment.set_max_iter` for an alternative usage with the different
            that `set_max_iter` is permenanent: it impacts all the future episodes and not only
            the next one.

        If you want your environment to start at a given time stamp you can do:

        .. code-block:: python

            import grid2op
            env_name = "l2rpn_case14_sandbox"

            env = grid2op.make(env_name)
            obs = env.reset(options={"init datetime": "2024-12-06 00:00"})
            obs.year == 2024
            obs.month == 12
            obs.day == 6

        .. seealso::
            If you specify "init datetime" then the observation resulting to the
            `env.reset` call will have this datetime. If you specify also `"skip ts"`
            option the behaviour does not change: the first observation will
            have the date time attributes you specified.

            In other words, the "init datetime" refers to the initial observation of the
            episode and NOT the initial time present in the time series.

        """
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
