How to add a new attribute to the observation
==============================================

Work in progress !

The best things to do if you want to add attributes to the observation is to
create a class, derived from `CompleteObservation` and to add attributes in:

- the `attr_list_vect` class attribute
- the `attr_vect_cpy` / `attr_simple_cpy` class attribute
  (depending of if the attributes are np.array or more simple attribute
  like `float` or `bool`)

And then you create an environment with your action class.

An example is given in the https://github.com/Grid2op/grid2op/tree/master/examples/backend_dependant_code
example.

.. danger::
    These examples works for grid2op >= 1.12. Upgrade grid2op if you 
    need this feature.

.. warning::
    In all cases, we recommend to do some extensive tests before using your new class 
    in an experiment using grid2op (that could last very long times)

    And once validated and everything works perfectly in a consistent manner, use your
    feature as much as you want.

    
Most simple setting
-----------------------

For example, suppose you want to add something computed by the environment but not part of the observation.

For the sake of this example let's assume the environment computes something calls "hidden_stuff" (an integer), so `env.hidden_stuff`
is an existing variable, but `obs.hidden_stuff` does not exists (yet).

But for some reason, you decide that an agent should have access to this `env.hidden_stuff`.

You can extend grid2op quite easily in this case.

First you create the Observation class that you want:

.. code-block:: python

    from grid2op.Environment import BaseEnv
    from grid2op.Observation import CompleteObservation

    class ObsWithHiddenStuff(CompleteObservation):
        # attributes that will be saved when action is
        # serialized as a numpy vector
        attr_list_vect = copy.deepcopy(CompleteObservation.attr_list_vect)
        attr_list_vect.append("hidden_stuff")

        # attributes that will also be used when action is
        # serialized as json
        attr_list_json = copy.deepcopy(CompleteObservation.attr_list_json)

        # attributes that will be copied
        # when observation is copied
        attr_simple_cpy  = copy.deepcopy(CompleteObservation.attr_simple_cpy)
        attr_simple_cpy.append("hidden_stuff")

        def __init__(self,
                     obs_env=None,
                     action_helper=None,
                     random_prng=None,
                     kwargs_env=None,
        )
            super.__init__(
                obs_env=obs_env,
                action_helper=action_helper,
                random_prng=random_prng,
                kwargs_env=kwargs_env)
            self.hidden_stuff = None

        def update(self, env: BaseEnv, with_forecast=True):
            # update standard attribute
            super().update(env, with_forecast=with_forecast)
            self.hidden_stuff = copy.copy(env.hidden_stuff)

And that is it.

Now, you need to tell grid2op to use this class, and for that you can do the 
following:

.. code-block:: python

    import grid2op
    
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name, 
                       observation_class=ObsWithHiddenStuff
                       )

    obs = env.reset()

    print(f"New things: {obs.hidden_stuff}")

.. warning::
    This is a fictious example, there is no attribute "hidden_stuff" in the environment,
    so this will NOT work. 

    This is, again, just an example.


Adding a vector attribute
---------------------------

A step a bit more complicated would be to add "vector" attribute to the observation.

It is quite similar to the previous things, but instead of setting `attr_simple_cpy` 
class attribute, you modify `attr_vect_cpy`

This gives something like:

.. code-block:: python

    from grid2op.Environment import BaseEnv
    from grid2op.Observation import CompleteObservation

    class ObsWithHiddenStuffVector(CompleteObservation):
        # attributes that will be saved when action is
        # serialized as a numpy vector
        attr_list_vect = copy.deepcopy(CompleteObservation.attr_list_vect)
        attr_list_vect.append("hidden_stuff_vect")

        # attributes that will also be used when action is
        # serialized as json
        attr_list_json = copy.deepcopy(CompleteObservation.attr_list_json)

        # attributes that will be copied
        # when observation is copied
        attr_vect_cpy  = copy.deepcopy(CompleteObservation.attr_vect_cpy)
        attr_vect_cpy.append("hidden_stuff_vect")

        def __init__(self,
                     obs_env=None,
                     action_helper=None,
                     random_prng=None,
                     kwargs_env=None,
        )
            super.__init__(
                obs_env=obs_env,
                action_helper=action_helper,
                random_prng=random_prng,
                kwargs_env=kwargs_env)
            
            # in this case it is required to know the size 
            # when initializing an observation.
            # if you want attribute with dynamic size, you can
            # 1) write a github issue explaining your usecase so 
            #    that a good solution is found
            # 2) use "0-padding" strategy and set the vector
            #    to the maximum size
            # 3) mark your vector as "simple" and not worry about the
            #    size at all (so setting attr_simple_cpy instead of attr_vect_cpy)
            self.hidden_stuff_vect = np.zeros(42, dtype=...)

        def update(self, env: BaseEnv, with_forecast=True):
            # update standard attribute
            super().update(env, with_forecast=with_forecast)
            self.hidden_stuff_vect[:] = copy.copy(env.hidden_stuff_vect)

And that is it.

.. note::
    You can also set `attr_simple_cpy` in this case. The difference is that
    it will be less efficient, especially when observation are copied.

    Observation can be copied multiple times in a single step

Adding attributes non trivial attribute
----------------------------------------

In the previous section, we explained how to add attributes that were already available in the environment.

But this feature can be used to retrieve more information, for example to compute other (or retrieve)
other type of attributes.


Generic Observation attributes
++++++++++++++++++++++++++++++++

You can dive a bit deeper and use this feature to compute values or extract values
from the environment's backend.

This is what is actually done in the example at https://github.com/Grid2op/grid2op/tree/master/examples/backend_dependant_code
were the backend computes some flow results based on n-1 computation.

Feel free to consult the documentation there for more information (duplicating code is the best way to add
typo, so best keep the example in a single place :-) )

The way it is coded is "generic" in the sens that it relies only on grid2op features. So nothing 
dramatic should happen if you use it correctly.

Backend Dependent Observation attribtues
+++++++++++++++++++++++++++++++++++++++++

You can also use this feature to add to the observation physical properties of grid equipment, for 
example the physical properties of powerlines.

In this later case, the observation will be "backend dependant" (you will not be able to use it with other grid2op backend,
but that is totally fine, as long as you are fine with the consequences). A code to get the "tap position" of
the transformer when using the PandaPowerBackend would look something like:


.. code-block:: python

    from grid2op.Environment import BaseEnv
    from grid2op.Observation import CompleteObservation

    class ObsWithTrafoTapPos(CompleteObservation):
        # attributes that will be saved when action is
        # serialized as a numpy vector
        attr_list_vect = copy.deepcopy(CompleteObservation.attr_list_vect)
        attr_list_vect.append("pp_tap_position")

        # attributes that will also be used when action is
        # serialized as json
        attr_list_json = copy.deepcopy(CompleteObservation.attr_list_json)

        # attributes that will be copied
        # when observation is copied
        attr_vect_cpy  = copy.deepcopy(CompleteObservation.attr_vect_cpy)
        attr_vect_cpy.append("pp_tap_position")

        def __init__(self,
                     obs_env=None,
                     action_helper=None,
                     random_prng=None,
                     kwargs_env=None,
        )
            super.__init__(
                obs_env=obs_env,
                action_helper=action_helper,
                random_prng=random_prng,
                kwargs_env=kwargs_env)
            
            # in this case it is required to know the size 
            # when initializing an observation.
            # if you want attribute with dynamic size, you can
            # 1) write a github issue explaining your usecase so 
            #    that a good solution is found
            # 2) use "0-padding" strategy and set the vector
            #    to the maximum size
            # 3) mark your vector as "simple" and not worry about the
            #    size at all (so setting attr_simple_cpy instead of attr_vect_cpy)
            self.pp_tap_position = np.zeros(type(self).n_line, dtype=int)

        def update(self, env: BaseEnv, with_forecast=True):
            # update standard attribute
            super().update(env, with_forecast=with_forecast)
            pp_grid = env.backend._grid  # of course this works only with pandapower !
            n_powerline = pp_grid.line.shape[0] 
            self.pp_tap_position[n_powerline:] = copy.copy(pp_grid.trafo["tap_pos"].values)
            # I know this will work because we know that PandaPowerBackend orders
            # first the powerlines (where we put a tap of 0 by default here)
            # and then the trafos. So we only need to update the last elements for the trafos

If you did not follow exactly what is happening here it's perfectly fine. It probably means 
that either pandapower or the PandaPowerBackend still have secrets for you. The idea is 
simply to tell you that, in the "update" definition you can use `env.backend._grid` to 
perform whatever you want.

Side effects and limitations
+++++++++++++++++++++++++++++

If you use backend dependant code, your observation class will only be usable with a given backend. It is also 
preferable that you are really familiar with the backend if you want to do correct things.

We also recommend to copy things from the backend to the observation (see the use of copy.copy AND of [:] in the example above).
This is to prevent any "unwanted modifications" when the agent reads back the observation. For example, we would not want that 
`obs.pp_tap_position[10] = 3` somewhere in the code modifies the backend internal state.

And, as for the "act.backend_dependant_callback", "with a great power comes great responsibility". You have total control
on the internal state of the backend here. So you can modify it without grid2op knowing it. This means that if you 
perform some "weird" things you can break almost everything on grid2op, and you can do so without grid2op raising 
any error or exception.

You might even do some modification that will be kept for an entire episode and even after a "env.reset".

.. danger::
    If you want to avoid any side effect, please consider that `env.backend` in general and
    `env.backend._grid` in particular are "read only" attributes. It's fine to copy them, 
    look at them etc.
    
    But we would not recommend to modify them directly, let alone in the "observation" class.
    If you want a direct control on them, you can use the :attr:`grid2op.Action.BaseAction.backend_dependant_callback`
    instead.

Observation class parametrized at their creation
-------------------------------------------------

You can also have your observation class depends on some parameters.

This is for example the case with the :class:`grid2op.Observation.NoisyObservation`

If you want to use such a class, you need to :

- the `attr_list_vect` class attribute
- the `attr_vect_cpy` / `attr_simple_cpy` class attribute
  (depending of if the attributes are np.array or more simple attribute
  like `float` or `bool`)
- pass the extra key-word arguments (kwargs) to the init of the BaseObservation class.

This looks something like this 
(code taken from https://github.com/Grid2op/grid2op/tree/master/examples/backend_dependant_code):


.. code-block:: python

    class ObsWithN1(CompleteObservation):
        # attributes that will be saved when action is
        # serialized as a numpy vector
        attr_list_vect = copy.deepcopy(CompleteObservation.attr_list_vect)
        attr_list_vect.append("n1_vals")
        
        # attributes that will also be used when action is
        # serialized as json
        attr_list_json = copy.deepcopy(CompleteObservation.attr_list_json)
        
        # attributes that will be copied
        # when observation is copied
        attr_vect_cpy  = copy.deepcopy(CompleteObservation.attr_vect_cpy)
        attr_vect_cpy.append("n1_vals")
        
        def __init__(self,
                    obs_env=None,
                    action_helper=None,
                    random_prng=None,
                    kwargs_env=None,
                    n1_li=None,  # kwargs not present in CompleteObservation
                    reduce_n1: Literal["max", "count", "sum"]="max", # kwargs not present in CompleteObservation
                    compute_algo: Literal["ac", "dc"]="ac"): # kwargs not present in CompleteObservation
            super().__init__(obs_env,
                            action_helper,
                            random_prng,
                            kwargs_env,
                            n1_li=n1_li,  # add the extra kwargs in the init here
                            reduce_n1=reduce_n1,  # add the extra kwargs in the init here
                            compute_algo=compute_algo  # add the extra kwargs in the init here
                            )
            ...
        
        def update(...):
            ...

You need to add the constructor in the init otherwise they will be "lost" when the observations
are copied for example. So basically they will never be used.

.. include:: final.rst
