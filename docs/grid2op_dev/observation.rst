How to add a new type of observation attribute
===================================

Work in progress !

GridObjects
===================================
* Add your attribute(s) to the end of `_clear_grid_dependant_class_attributes()`, assume they are None / False by default. 
For tidiness you could define a new DEFAULT_XXXX in default_var, but this is not required.
* Add your attributes to the appropriate `_li_attr_COMPONENT`, or define a new one in the class attributes. Do the same for `_type_attr_XXXX`.
* Ensure your new class attributes are also defined in `_clear_class_attributes(cls)`
* Add your attribute(s) to `_check_convert_to_numpy_array()` method, under the appropriate data type(s)
* Add your attribute flag(s), if any, to `assert_grid_correct_cls()`. You could for instance use a flag for checking if your attributes should exist. 
If the logic gets complicated define a new `_check_validity_XXXX()` method.
* Add your attribute(s) to `_make_cls_dict()` and `_make_cls_dict_extended()` methods
* Add your attribute(s) to `from_dict()` method, with appropriate defaults. 
* Add your attribute(s) to the `__str__()` method, note that this can break some unit tests - so would recommend doing this last (after you are passing all tests).


BaseObservation
===================================
* Add your attribute(s) to the class attribute: `attr_list_vect`
* Add empty array(s) for your attribute(s) in the `__init__()` method.
* Add your attribute(s) to the list of Literals in `state_of()` signature and add it/them to the appropriate res dictionary inside the method.
* Include your attribute(s) in the `reset()` method
* Append your attribute(s) to the appropriate `_aux_add_XXX()` method
* Add your attribute(s) to `_update_obs_complete()` method

CompleteObservation
===================================
* Append your attribute(s) to the class attribute: `attr_list_vect`. 

.. include:: final.rst

