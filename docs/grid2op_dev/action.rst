How to add a new type of action
===================================

Work in progress !

Gory implementation detail
----------------------------

The Action class has been, in grid2Op 1.12.1, refactored to be faster. From this version onwards,
if the agent does "nothing", then no numpy arrays are created. 

From our benchmark, the creation of all of these numpy arrays was taking (relatively) long time. This
was then decided to increase the code complexity to avoid creating these arrays when they are not needed, 
while keeping compatibility with oldest code.

Powergrid modifications are represented with vectors (numpy array), for example `self._set_topo_vect` is a "vector"
representing the changes made by the agent to the topology of the grid.

From grid2op 1.12.1, this "vector" is now a property:

.. code-block:: python

    @property
    def _set_topo_vect(self) -> np.ndarray:
        # see later for this part of the code
        return self._private_set_topo_vect

The underlying `self._private_set_topo_vect` is either a numpy array (if the agent already modified it) or
`None`. It is initialized at `None` by default.

The `_private_set_topo_vect` is automatically created (and initialized with a vector corresponding to `do nothing`)
if the property `_set_topo_vect` is called, this done in :

.. code-block:: python

    @property
    def _set_topo_vect(self) -> np.ndarray:
        if self._private_set_topo_vect is None:
            cls = type(self)
            self._private_set_topo_vect = cls._build_attr("_set_topo_vect")
        return self._private_set_topo_vect

The function `cls._build_attr("_set_topo_vect")` will take care of the creation of such element.
This is a class method that defines the class attribute `DICT_ATTR_` which is simply a mapping with
the key being the property name (*eg* `"_set_topo_vect"`) and the value being the default value of 
this property.

Agent public API
-----------------------------------------

First, you need to come up with a name for it, say "_my_awesome_way_to_modif".

Create the glue code with the property
***************************************
Then you need to create the "glue code" of the "property" and the underlying element.

We recommend that you create, in the `__init__` of the base action:

.. code-block:: python

    # in the __init__
    self._private_my_awesome_way_to_modif = None

Then you create the property (just like a regular method):

.. code-block:: python

    @property
    def _my_awesome_way_to_modif(self) -> np.ndarray:
        if self._private_my_awesome_way_to_modif is None:
            cls = type(self)
            self._private_my_awesome_way_to_modif = cls._build_attr("_my_awesome_way_to_modif")
        return self._private_my_awesome_way_to_modif

And finally, you need to create the "default" / "do nothing" value for your "new action" and update the 
class method `_build_dict_attr_if_needed` of the BaseAction.

For example, say your attribute works like a "set" attribute, the default value (correspoding to 
"I don't want to modify it") will then be all 0, and your attribute modifies the loads (it has 
`n_load` components) then you can add, in the `_build_dict_attr_if_needed` method, at the end:

.. code-block:: python

    @classmethod
    def _build_dict_attr_if_needed(cls):
        # lots of code you don't have to modify
        # ...
        # ...
        # and after everything: 
        cls.DICT_ATTR_["_my_awesome_way_to_modif"] = np.full(cls.n_load, dtype=dt_int, fill_value=0)

Add the possibility for the agent to modifies it
*************************************************

TODO see the `digest_XXX` code and 
the `cls.authorized_keys` and
function `def supports_type` of SerializableActionSpace and
`cls.mapping_vect_auth_keys` (BaseAction) 

Checklist for SerializableActionSpace:
    Add a new ID number for your action to `SerializeableActionSpace`'s class attributes. Add this to the sample() 
    and "_get_possible_action_types()" methods. 
    Add a _sample_YOURACTION() method to `SerializableActionSpace`
    Add a _aux_get_back_to_ref_state_YOURACTION() method.
    Add your action to get_back_to_ref_state() method.
    Optional: If your new action is continuous, add a get_all_unitary_YOURACTION() method to
    `SerializableActionSpace` to help users discretize it.

Create a new YourAction.py file in `Grid2op/Action/*` and add this to the `Grid2Op/Action/__init__.py`.


Add the vectorization
***********************

TODO add the `cls.attr_list_vect` to baseAction:
* Add to baseAction.authorized_keys
* Add to baseAction.attr_list_vect
* Add to baseAction.mapping_vect_auth_keys
* Make sure your `self._private_YOURACTION = None` attribute is in baseAction's `__init__`
* Add a self._modif_YOURACTION to baseAction's `__init__`
* Double check you added a @property for your action in BaseAction
* Update the BaseAction._aux_copy() method to include your action (ideally in an if-statement)
* Modify the end of `process_grid2op_compat()`
* Add your _modif_YOURACTION to `reset_modified_flags()` and `can_affect_something()`
* Add your action to `_post_process_from_vect()` as appropriate
* Add a new method called `_aux_iadd_YOURACTION()` and reference this in `__iadd__()`
* Add your action to `__call__(self)` by referencing self._private_YOURACTION
* Add a `_digest_YOURACTION()` method
* Add an example to the docstring of `update(self)`
* Add modification check to `_check_for_corret_modif_flags()`
* Add ambiguity checks for your action in `_check_for_ambiguity()`, for instance by adding and calling a new `_is_YOURACTION_ambigious()` method
* Add a string formatting for your action to `__str__()`
* Add a check to `impact_on_objects()` for your action
* Add a boolean for your action in `get_types()`
* Modify `_aux_effect_on_XXX()` as appropriate
* If you action does NOT modify topology, add it to the `_dont_affect_topology()` method

Worry about the backward compatibility
****************************************

TODO see `cls.process_grid2op_compat`: best practice:

Add a class version id, like `cls.MIN_VERSION_DETACH` 
And then write things like it has been done for the detachment feature.
This could look like this:

.. code-block:: python

    @classmethod
    def process_grid2op_compat(cls):
        # lots of things irrelevant here
        # ...
        if glop_ver < cls.MIN_VERSION_AWESOME_WAY:
            # this feature did not exist before.
            if "awesome_way" in cls.authorized_keys:
                cls.authorized_keys = copy.deepcopy(cls.authorized_keys)
                cls.authorized_keys.remove("awesome_way")
            if "_my_awesome_way_to_modif" in cls.attr_list_vect:
                cls.attr_list_vect = copy.deepcopy(cls.attr_list_vect)
                cls.attr_list_vect.remove("_my_awesome_way_to_modif")
        # other irrelevant things for the example
        # ...


Implement the actual API
*******************************

This is located in the `digest_XXX` function (*eg* `_digest_set_status`) which are in turned used in the `self.update` action.

You might also want to update the `_modif_XXX` flags and the `self.can_affect_something()`, `self._dont_affect_topology()` etc.

.. danger::
    When Implementing the function `digest_XXX` either you rely on setting the properties and the `_aux_affect_object_bool`,
    `_aux_affect_object_float` or `_aux_affect_object_int` and you're good.

    Or you want a specific API and in this case do not forget to set the flag `self._cached_is_not_ambiguous` to False
    if your action modifies something.

Other action usage
--------------------

Implement the "other" action API, for example `as_dict`, `as_serializable_dict`, `__eq__`, `get_topological_impact`, 
`self.decompose_as_unary_action()` etc.

Add gym compatibility in the gym_compat module
***********************************************

TODO 

Backend "private" API
-----------------------

Then you need to modify the `__iadd__` method of the `BackendAction` class to handle the modification
you performed and pass it to the backend.
* Add a `set_YOURACTION()`  if your action modifies setpoints / injections.


Add tests
-----------

TODO
* Add a test for creating the action type


Add documentation
-------------------

TODO 


.. include:: final.rst