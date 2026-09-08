# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Literal, TYPE_CHECKING

from grid2op.Exceptions import ChronicsError

from grid2op.Chronics.gridValue import GridValue
from grid2op.Chronics.fromOneEpisodeData import TYPE_EP_DATA_INGESTED, FromOneEpisodeData


if TYPE_CHECKING:
    from grid2op.Action.playableAction import PlayableAction

class FromMultiEpisodeData(GridValue): 
    """This class allows to redo some episode that have been previously run using a runner.
    
    It is an extension of the class :class:`FromOneEpisodeData` but with multiple episodes.
    
    .. seealso::
        :class:`grid2op.Chronics.FromOneEpisodeData` if you want to use only one episode
        
    .. warning::
        It has the same limitation as :class:`grid2op.Chronics.FromOneEpisodeData`, including:
        
        - forecasts are not saved so cannot be retrieved with this class. You can however
          use `obs.simulate` and in this case it will lead perfect forecasts.
        - to make sure you are running the exact same episode, you need to create the environment
          with the :class:`grid2op.Opponent.FromEpisodeDataOpponent` opponent

    .. versionchanged:: 1.11.0
        Before versin 1.11.0 this class would load all the data in memory at the creation of the environment,
        which could take lots of time and memory but once done a call to `env.reset` would be really fast.
        
        From grid2op >= 1.11.0 a kwargs `caching` has been added (default value is ``FALSE``) which
        does not load everything in memory which makes it more memory efficient and (maybe) more time saving
        (if some data happened to be loaded but never used). The default behaviour has then
        changed.
        
        You can still benefit from previous behaviour by loading with `caching=True`
        
    Examples
    ---------
    You can use this class this way:
    
    First, you generate some data by running an episode with do nothing or reco powerline agent,
    preferably episode that go until the end of your time series
    
    .. code-block:: python
    
        import grid2op
        from grid2op.Runner import Runner
        from grid2op.Agent import RecoPowerlineAgent
        
        path_agent = ....
        nb_episode = ...
        env_name = "l2rpn_case14_sandbox"  # or any other name
        env = grid2op.make(env_name, etc.)
        
        # optional (change the parameters to allow the )
        param = env.parameters
        param.NO_OVERFLOW_DISCONNECTION = True
        env.change_parameters(param)
        env.reset()
        # end optional
        
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=RecoPowerlineAgent)
        runner.run(nb_episode=nb_episode,
                   path_save=path_agent)
    
    And then you can load it back and run the exact same environment with the same
    time series, the same attacks etc. with:
    
    .. code-block:: python
    
        import grid2op
        from grid2op.Chronics import FromMultiEpisodeData
        from grid2op.Opponent import FromEpisodeDataOpponent
        from grid2op.Episode import EpisodeData
        
        path_agent = ....  # same as above
        env_name = .... # same as above
        
        # path_agent is the path where data coming from a grid2op runner are stored
        # NB it should come from a do nothing agent, or at least
        # an agent that does not modify the injections (no redispatching, curtailment, storage)
        li_episode = EpisodeData.list_episode(path_agent)
        
        env = grid2op.make(env_name,
                           chronics_class=FromMultiEpisodeData,
                           data_feeding_kwargs={"li_ep_data": li_episode},
                           opponent_class=FromEpisodeDataOpponent,
                           opponent_attack_cooldown=1,
                      )
        # li_ep_data in this case is a list of anything that is accepted by `FromOneEpisodeData`

        obs = env.reset()
    
        # and now you can use "env" as any grid2op environment.
        
    """
    MULTI_CHRONICS = True
    def __init__(self,
                 path,  # can be None !
                 li_ep_data: List[TYPE_EP_DATA_INGESTED],
                 time_interval=timedelta(minutes=5),
                 sep=";",  # here for compatibility with grid2op, but not used
                 max_iter=-1,
                 start_datetime=datetime(year=2019, month=1, day=1),
                 chunk_size=None,
                 list_perfect_forecasts=None,
                 caching : bool=False,
                 **kwargs,  # unused
                 ):
        super().__init__(time_interval, max_iter, start_datetime, chunk_size)
        self._caching : bool = bool(caching)
        self._path = path
        self._chunk_size = chunk_size
        self._list_perfect_forecasts = list_perfect_forecasts
        self._input_li_ep_data = li_ep_data
        if self._caching:
            self.li_ep_data = [FromOneEpisodeData(path,
                                                  ep_data=el,
                                                  time_interval=time_interval,
                                                  max_iter=max_iter,
                                                  chunk_size=chunk_size,
                                                  list_perfect_forecasts=list_perfect_forecasts,
                                                  start_datetime=start_datetime)
                               for el in li_ep_data
                              ]
        else:
            self.li_ep_data = [None for _ in li_ep_data]
        self._prev_cache_id = len(self.li_ep_data) - 1
        self.data = self.li_ep_data[self._prev_cache_id]
        if self.data is None:
            self.data = FromOneEpisodeData(self._path,
                                           ep_data=self._input_li_ep_data[self._prev_cache_id],
                                           time_interval=self.time_interval,
                                           max_iter=self.max_iter,
                                           chunk_size=self._chunk_size,
                                           list_perfect_forecasts=self._list_perfect_forecasts,
                                           start_datetime=self.start_datetime)
        self._episode_data = self.data._episode_data  # used by the fromEpisodeDataOpponent
        
    def next_chronics(self):
        self._prev_cache_id += 1
        # TODO implement the shuffling indeed.
        # if self._prev_cache_id >= len(self._order):
        #     self.space_prng.shuffle(self._order)
        self._prev_cache_id %= len(self.li_ep_data)

    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):

        self.data = self.li_ep_data[self._prev_cache_id]
        if self.data is None:
            # data was not in cache:
            self.data = FromOneEpisodeData(self._path,
                                           ep_data=self._input_li_ep_data[self._prev_cache_id],
                                           time_interval=self.time_interval,
                                           max_iter=self.max_iter,
                                           chunk_size=self._chunk_size,
                                           list_perfect_forecasts=self._list_perfect_forecasts,
                                           start_datetime=self.start_datetime)
        self.data.initialize(
            order_backend_loads,
            order_backend_prods,
            order_backend_lines,
            order_backend_subs,
            names_chronics_to_backend=names_chronics_to_backend,
        )
        self._episode_data = self.data._episode_data 
        if self.action_space is not None:
            if self.data.action_space is None:
                self.data.action_space = self.action_space
        
    def done(self):
        return self.data.done()

    def load_next(self):
        return self.data.load_next()
    
    def check_validity(self, backend):
        return self.data.check_validity(backend)

    def forecasts(self):
        return self.data.forecasts()
    
    def tell_id(self, id_num: str, previous=False):
        try:
            id_num = int(id_num)
            path_ = None
        except ValueError:
            path_, id_num = id_num.split("@")
            id_num = int(id_num)
            
        if path_ is not None and path_ != self._path:
            raise ChronicsError("FromMultiEpisodeData can only be used with `tell_id` being an integer "
                                "or if tell_id has the same path as the original file. "
                                "Feel free to write a feature request if you want more.")
        
        self._prev_cache_id = id_num
        self._prev_cache_id %= len(self.li_ep_data)

        if previous:
            self._prev_cache_id -= 1
            self._prev_cache_id %= len(self.li_ep_data)
    
    def get_id(self) -> str:
        return f'{self._path}@{self._prev_cache_id}'
    
    def max_timestep(self):
        return self.data.max_timestep()
    
    def fast_forward(self, nb_timestep):
        self.data.fast_forward(nb_timestep)
        
    def get_init_action(self, names_chronics_to_backend: Optional[Dict[Literal["loads", "prods", "lines"], Dict[str, str]]]=None) -> Union["PlayableAction", None]:
        return self.data.get_init_action(names_chronics_to_backend)
    
    def cleanup_action_space(self):
        super().cleanup_action_space()
        self.data.cleanup_action_space()
