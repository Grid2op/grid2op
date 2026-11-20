# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Optional, Dict, Literal
from grid2op.Action.playableAction import PlayableAction


class ShuntAction(PlayableAction):
    """
    This type of :class:`PlayableAction` only implements the modifications of the shunts,
    through the "shunt" keyword or the :attr:`BaseAction.shunt_p`, :attr:`BaseAction.shunt_q`
    and :attr:`BaseAction.shunt_bus` properties.

    Nothing else is supported (no topology, no redispatching, no curtailment, etc.) and any attempt
    to use something else is refused. This makes it suited to study voltage control with the shunts only.

    The shunts are only available if the backend supports them (`shunts_data_available`). If it does
    not, this action type has no attribute at all and it behaves like a "do nothing" action.

    Examples
    --------

    .. code-block:: python

        import grid2op
        from grid2op.Action import ShuntAction
        env = grid2op.make("l2rpn_case14_sandbox", action_class=ShuntAction)

        act = env.action_space()
        act.shunt_q = [(0, -25.)]  # shunt 0 now produces 25 MVAr
        obs, reward, done, info = env.step(act)

    """

    # the shunt attributes and the "shunt" key are added by `_add_shunt_data`
    # only if the backend supports the shunts
    authorized_keys = set()
    attr_list_vect = []
    attr_list_set = set()
    shunt_added = False

    def __init__(self, _names_chronics_to_backend: Optional[Dict[Literal["loads", "prods", "lines"], Dict[str, str]]]=None):
        super().__init__(_names_chronics_to_backend)
        self.authorized_keys_to_digest = {"shunt": self._digest_shunt}
