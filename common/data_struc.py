"""
Copyright (C) 2024  Olivier DEBAUCHE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Dict
import numpy as np


class LimitedDict(dict):
    def __init__(self, keys):
        self._keys = keys
        self.clear()

    def __setitem__(self, key, value):
        if key not in self._keys:
            raise KeyError
        dict.__setitem__(self, key, value)

    def clear(self):
        for key in self._keys:
            self[key] = list()


def convert_LDict_to_Dict(ld: LimitedDict) -> Dict[str, np.array]:
    d = dict()
    for key, val in ld.items():
        d[key] = val[0]
    return d
