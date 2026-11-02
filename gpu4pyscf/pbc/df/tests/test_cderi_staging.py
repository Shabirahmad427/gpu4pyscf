# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy
import numpy

from gpu4pyscf.pbc.df.df import _stage_mapped_host_array


def test_stage_mapped_host_array_copies_pinned_numpy_views():
    pinned = cupy.cuda.alloc_pinned_memory(8 * 6 * 4)
    owner = numpy.ndarray((6, 4), dtype=numpy.float64, buffer=pinned)
    owner[:] = numpy.arange(owner.size).reshape(owner.shape)
    array = owner[::2]

    staged = _stage_mapped_host_array(array)

    assert staged is not array
    assert staged.flags.c_contiguous
    assert not numpy.shares_memory(staged, array)
    numpy.testing.assert_array_equal(staged, array)


def test_stage_mapped_host_array_keeps_ordinary_numpy_arrays():
    array = numpy.arange(12, dtype=numpy.float64).reshape(3, 4)

    assert _stage_mapped_host_array(array) is array
