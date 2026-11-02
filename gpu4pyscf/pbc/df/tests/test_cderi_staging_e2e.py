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
from pyscf.pbc import gto
from pyscf.pbc.df import df as df_cpu

from gpu4pyscf.pbc.df.df import GDF


def test_gdf_get_jk_stages_builder_pinned_cderi():
    cell = gto.Cell()
    cell.atom = 'He 0 0 0'
    cell.a = numpy.eye(3) * 4.0
    cell.basis = 'sto-3g'
    cell.mesh = [9, 9, 9]
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()

    gpu_df = GDF(cell).build()
    cderi = gpu_df._cderi[0]
    assert isinstance(cderi, numpy.ndarray)
    owner = cderi
    while isinstance(owner, numpy.ndarray):
        owner = owner.base
    assert isinstance(owner, cupy.cuda.PinnedMemoryPointer)

    dm = numpy.eye(cell.nao)
    vj, vk = gpu_df.get_jk(dm, hermi=1)
    vj_ref, vk_ref = df_cpu.GDF(cell).get_jk(dm, hermi=1)

    cupy.testing.assert_allclose(vj.get(), vj_ref, rtol=1e-8, atol=1e-8)
    cupy.testing.assert_allclose(vk.get(), vk_ref, rtol=1e-8, atol=1e-8)
