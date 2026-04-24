# Copyright (c) 2019-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import copy
import warnings

from grid2op.Environment import Environment
from grid2op.Exceptions import EnvError
from grid2op.Space import GRID2OP_CLASSES_ENV_FOLDER
from grid2op.Space import DEFAULT_N_BUSBAR_PER_SUB, DEFAULT_ALLOW_DETACHMENT

from grid2op.MakeEnv.PathUtils import _aux_fix_backend_internal_classes

from .get_default_env_kwargs import (
    get_default_env_kwargs,
    _check_path
)


def make_from_dataset_path(
    dataset_path="/",
    logger=None,
    experimental_read_from_local_dir=False,
    n_busbar=DEFAULT_N_BUSBAR_PER_SUB,
    allow_detachment=DEFAULT_ALLOW_DETACHMENT,
    _add_cls_nm_bk=True,
    _add_to_name="",
    _compat_glop_version=None,
    _overload_name_multimix=None,
    **kwargs,
) -> Environment:
    """
    INTERNAL USE ONLY

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Prefer using the :func:`grid2op.make` function.


    .. danger::
        The :func:`grid2op.make` function can execute arbitrary code. Do not attempt
        to "make" an environment for which you don't trust (or even know) the authors.
        

    This function is a shortcut to rapidly create environments within the grid2op Framework. We don't
    recommend using directly this function. Prefer using the :func:`make` function.

    It mimic the ``gym.make`` function.

    .. _Parameters-make-from-path:

    Parameters
    ----------

    dataset_path: ``str``
        Path to the dataset folder

    logger:
        Something to pass to grid2op environment to be used as logger.

    param: ``grid2op.Parameters.Parameters``, optional
        Type of parameters used for the Environment. Parameters defines how the powergrid problem is cast into an
        markov decision process, and some internal

    backend: ``grid2op.Backend.Backend``, optional
        The backend to use for the computation. If provided, it must be an instance of :class:`grid2op.Backend.Backend`.

    n_busbar: ``int``
        Number of independant busbars allowed per substations. By default it's 2.

    allow_detachment; ``bool``
        Whether to allow loads/generators to be detached without a game over. By default False.
        
    action_class: ``type``, optional
        Type of BaseAction the BaseAgent will be able to perform.
        If provided, it must be a subclass of :class:`grid2op.BaseAction.BaseAction`

    observation_class: ``type``, optional
        Type of BaseObservation the BaseAgent will receive.
        If provided, It must be a subclass of :class:`grid2op.BaseAction.BaseObservation`

    reward_class: ``type``, optional
        Type of reward signal the BaseAgent will receive.
        If provided, It must be a subclass of :class:`grid2op.BaseReward.BaseReward`

    other_rewards: ``dict``, optional
        Used to additional information than the "info" returned value after a call to env.step.

    gamerules_class: ``type``, optional
        Type of "Rules" the BaseAgent need to comply with. Rules are here to model some operational constraints.
        If provided, It must be a subclass of :class:`grid2op.RulesChecker.BaseRules`

    data_feeding_kwargs: ``dict``, optional
        Dictionnary that is used to build the `data_feeding` (chronics) objects.

    chronics_class: ``type``, optional
        The type of chronics that represents the dynamics of the Environment created. Usually they come from different
        folders.

    data_feeding: ``type``, optional
        The type of chronics handler you want to use.

    volagecontroler_class: ``type``, optional
        The type of :class:`grid2op.VoltageControler.VoltageControler` to use, it defaults to

    chronics_path: ``str``
        Path where to look for the chronics dataset (optional)

    grid_path: ``str``, optional
        The path where the powergrid is located.
        If provided it must be a string, and point to a valid file present on the hard drive.

    difficulty: ``str``, optional
        the difficulty level. If present it starts from "0" the "easiest" but least realistic mode. In the case of the
        dataset being used in the l2rpn competition, the level used for the competition is "competition" ("hardest" and
        most realistic mode). If multiple difficulty levels are available, the most realistic one
        (the "hardest") is the default choice.

    opponent_space_type: ``type``, optional
        The type of opponent space to use. If provided, it must be a subclass of `OpponentSpace`.
        
    opponent_action_class: ``type``, optional
        The action class used for the opponent. The opponent will not be able to use action that are invalid with
        the given action class provided. It defaults to :class:`grid2op.Action.DontAct` which forbid any type
        of action possible.

    opponent_class: ``type``, optional
        The opponent class to use. The default class is :class:`grid2op.Opponent.BaseOpponent` which is a type
        of opponents that does nothing.

    opponent_init_budget: ``float``, optional
        The initial budget of the opponent. It defaults to 0.0 which means the opponent cannot perform any action
        if this is not modified.

    opponent_attack_duration: ``int``, optional
        The number of time steps an attack from the opponent lasts.

    opponent_attack_cooldown: ``int``, optional
        The number of time steps the opponent as to wait for an attack.

    opponent_budget_per_ts: ``float``, optional
        The increase of the opponent budget per time step. Each time step the opponent see its budget increase. It
        defaults to 0.0.

    opponent_budget_class: ``type``, optional
        defaults: :class:`grid2op.Opponent.UnlimitedBudget`

    kwargs_observation: ``dict``
        Key words used to initialize the observation. For example, in case of NoisyObservation, 
        it might be the standar error for each underlying distribution. It might
        be more complicated for other type of custom observations but should be
        deep copiable.

        Each observation will be initialized (by the observation_space) with:

        .. code-block:: python
        
            obs = observation_class(obs_env=self.obs_env,
                                    action_helper=self.action_helper_env,
                                    random_prng=self.space_prng,
                                    **kwargs_observation  # <- this kwargs is used here
                                   )

    observation_backend_class:
        The class used to build the observation backend (used for Simulator 
        obs.simulate and obs.get_forecasted_env). If provided, this should 
        be a type / class and not an instance of this class. (by default it's None)
        
    observation_backend_kwargs:
        The key-word arguments to build the observation backend (used for Simulator, 
        obs.simulate and obs.get_forecasted_env). This should be a dictionnary. 
        (by default it's None)
    
    _add_to_name:
        Internal, used for test only. Do not attempt to modify under any circumstances.

    _compat_glop_version:
        Internal, used for test only. Do not attempt to modify under any circumstances.

    # TODO update doc with attention budget

    Returns
    -------
    env: :class:`grid2op.Environment.Environment`
        The created environment with the given properties.

    """    
    # Compute and find root folder
    _check_path(dataset_path, "Dataset root directory")
 
    res_default_kwargs = get_default_env_kwargs(
        dataset_path=dataset_path,
        logger=logger,
        n_busbar=n_busbar,
        allow_detachment=allow_detachment,
        _add_cls_nm_bk=_add_cls_nm_bk,
        _add_to_name=_add_to_name,
        _compat_glop_version=_compat_glop_version,
        _overload_name_multimix=_overload_name_multimix,
        **kwargs
    )
    
    (default_kwargs,
     use_class_in_files,
     grid_path_abs,
     data_feeding,
     graph_layout,
     backend,
     thermal_limits,
     allow_loaded_backend,
     classes_path,
     init_env,
     this_local_dir,
     do_not_erase_cls) = res_default_kwargs
    
    if use_class_in_files:
        # new behaviour
        if _overload_name_multimix is None:
            sys_path_cls = os.path.join(os.path.split(grid_path_abs)[0], GRID2OP_CLASSES_ENV_FOLDER)
        else:
            sys_path_cls = os.path.join(_overload_name_multimix[1], GRID2OP_CLASSES_ENV_FOLDER)
        if not os.path.exists(sys_path_cls):
            try:
                os.mkdir(sys_path_cls)
            except FileExistsError:
                # if another process created it, no problem
                pass
            
        init_nm = os.path.join(sys_path_cls, "__init__.py")
        if not os.path.exists(init_nm):
            try:
                with open(init_nm, "w", encoding="utf-8") as f:
                    f.write("This file has been created by grid2op in a `env.make(...)` call. Do not modify it or remove it")
            except FileExistsError:
                pass
            
        import tempfile
        if _overload_name_multimix is None or _overload_name_multimix[0] is None:
            this_local_dir = tempfile.TemporaryDirectory(dir=sys_path_cls)
            this_local_dir_name = this_local_dir.name
        else:
            this_local_dir_name = _overload_name_multimix[0]
            this_local_dir = None
            do_not_erase_cls = True
            
        if experimental_read_from_local_dir:
            warnings.warn("With the automatic class generation, we removed the possibility to "
                          "set `experimental_read_from_local_dir` to True.")
            experimental_read_from_local_dir = False
        # TODO: check the hash thingy is working in baseEnv._aux_gen_classes (currently a pdb)
        
        # TODO check that it works if the backend changes, if shunt / no_shunt if name of env changes etc.
        
        # TODO: what if it cannot write on disk => fallback to previous behaviour
        data_feeding_fake = copy.deepcopy(data_feeding)
        data_feeding_fake.cleanup_action_space()
        
        # Set graph layout if not None and not an empty dict
        if graph_layout is not None and graph_layout:
            type(backend).attach_layout(graph_layout)
            
        if not os.path.exists(this_local_dir_name):
            raise EnvError(f"Path {this_local_dir_name} has not been created by the tempfile package")
        init_env = Environment(**default_kwargs,
                               chronics_handler=data_feeding_fake,
                               _read_from_local_dir=None,  # first environment to generate the classes and save them
                               _local_dir_cls=None,
                               )   
        if not os.path.exists(this_local_dir.name):
            raise EnvError(f"Path {this_local_dir.name} has not been created by the tempfile package")
        init_env.generate_classes(local_dir_id=this_local_dir.name)
        # fix `my_bk_act_class` and `_complete_action_class`
        _aux_fix_backend_internal_classes(type(backend), this_local_dir)
        init_env.backend = None  # to avoid to close the backend when init_env is deleted
        init_env._local_dir_cls = None
        classes_path = this_local_dir_name
        allow_loaded_backend = True
    else:
        # legacy behaviour (<= 1.10.1 behaviour)
        classes_path = None if not experimental_read_from_local_dir else experimental_read_from_local_dir
        if experimental_read_from_local_dir:
            if _overload_name_multimix is not None:
                # I am in a multimix
                sys_path = os.path.join(_overload_name_multimix.path_env, GRID2OP_CLASSES_ENV_FOLDER)
            else:
                # I am not in a multimix
                sys_path = os.path.join(os.path.split(grid_path_abs)[0], GRID2OP_CLASSES_ENV_FOLDER)
            if not os.path.exists(sys_path):
                raise RuntimeError(
                    "Attempting to load the grid classes from the env path. Yet the directory "
                    "where they should be placed does not exists. Did you call `env.generate_classes()` "
                    "BEFORE creating an environment with `experimental_read_from_local_dir=True` ?"
                )
            if not os.path.isdir(sys_path) or not os.path.exists(
                os.path.join(sys_path, "__init__.py")
            ):
                raise RuntimeError(
                    f"Impossible to load the classes from the env path. There is something that is "
                    f"not a directory and that is called `_grid2op_classes`. "
                    f'Please remove "{sys_path}" and call `env.generate_classes()` where env is an '
                    f"environment created with `experimental_read_from_local_dir=False` (default)"
                )
            import sys
            sys.path.append(os.path.split(os.path.abspath(sys_path))[0])
            classes_path = sys_path

    # new in 1.11.0
    if _overload_name_multimix is not None:
        # case of multimix
        if  _overload_name_multimix.mix_id >= 1 and _overload_name_multimix.local_dir_tmpfolder is not None:  
            # this is not the first mix
            # for the other mix I need to read the data from files and NOT
            # create the classes
            this_local_dir = _overload_name_multimix.local_dir_tmpfolder
            classes_path = this_local_dir.name
        
    # Finally instantiate env from config & overrides
    # including (if activated the new grid2op behaviour)
    env = Environment(
        **default_kwargs,
         chronics_handler=data_feeding,
        _allow_loaded_backend=allow_loaded_backend,
        _read_from_local_dir=classes_path,
        _local_dir_cls=this_local_dir,
    )   
    if do_not_erase_cls is not None:
        env._do_not_erase_local_dir_cls = do_not_erase_cls
    # Update the thermal limit if any
    if thermal_limits is not None:
        env.set_thermal_limit(thermal_limits)
        
    # Set graph layout if not None and not an empty dict
    if graph_layout is not None and graph_layout:
        try:
            env.attach_layout(graph_layout)
        except EnvError as exc_:
            warnings.warn(f"Error {exc_} while setting the environment layout.")
    return env
