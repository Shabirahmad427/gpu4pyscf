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


import pyscf
import cupy as cp
import numpy as np

import pyscf.df.addons

from pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.mp import dfmp2_addons, dfmp2_drivers
from gpu4pyscf.mp.dfmp2 import DFMP2 as GPUDFMP2


def kernel(
    mp,
    mo_energy=None,
    mo_coeff=None,
    mo_occ=None,
    frozen_mask=None,
    with_t2=None,
    j2c_decomp_alg=None,
    j3c_backend=None,
    fp_type=None,
    verbose=None,
):
    mol = mp.mol
    aux = mp.auxmol
    log = logger.new_logger(mol, verbose)

    # handle default values for parameters
    frozen_mask = frozen_mask if frozen_mask is not None else mp.get_frozen_mask()
    with_t2 = with_t2 if with_t2 is not None else mp.with_t2
    j2c_decomp_alg = j2c_decomp_alg if j2c_decomp_alg is not None else mp.j2c_decomp_alg
    j3c_backend = j3c_backend if j3c_backend is not None else mp.j3c_backend
    fp_type = fp_type if fp_type is not None else mp.fp_type

    assert fp_type in ['FP64', 'FP32']
    dtype_cderi = np.float64 if fp_type == 'FP64' else np.float32

    # obtain necessary arguments
    [_, occ_coeff, vir_coeff, _] = dfmp2_addons.split_mo_coeff_unrestricted(mp, mo_coeff=mo_coeff, frozen_mask=frozen_mask, mo_occ=mo_occ)
    [_, occ_energy, vir_energy, _] = dfmp2_addons.split_mo_energy_unrestricted(mp, mo_energy=mo_energy, frozen_mask=frozen_mask, mo_occ=mo_occ)

    # ROHF-derived orbitals are not UHF-canonical: rotate to the semicanonical
    # basis and add the first-order singles energy, as in PySCF UMP2.
    e_corr_singles = 0.
    mo_coeff = mp.mo_coeff if mo_coeff is None else mo_coeff
    mo_occ = mp.mo_occ if mo_occ is None else mo_occ
    if _is_spin_restricted_open_shell(mo_coeff, mo_occ):
        if with_t2:
            raise NotImplementedError(
                'with_t2=True is not supported for ROHF references: the amplitudes '
                'would be in the semicanonical basis, not the input orbital basis.')
        log.info('Spin-restricted open-shell reference: using semicanonical orbitals '
                 'and first-order singles')
        occ_coeff, vir_coeff, occ_energy, vir_energy, e_corr_singles = \
            _semicanonicalize(mp, mo_coeff, mo_occ, occ_coeff, vir_coeff)

    # allocate t2 if needed
    if with_t2 is True:
        if mp.t2 is None:
            spins = [0, 1]
            nocc = [occ_coeff[spin].shape[1] for spin in spins]
            nmo = occ_coeff[0].shape[0]
            nvir = [nmo - nocc[spin] for spin in spins]
            mp.t2 = [
                cp.empty((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=dtype_cderi),
                cp.empty((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=dtype_cderi),
                cp.empty((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=dtype_cderi),
            ]
    else:
        mp.t2 = None

    # run driver
    result = dfmp2_drivers.dfump2_kernel_one_gpu(
        mol,
        aux,
        occ_coeff,
        vir_coeff,
        occ_energy,
        vir_energy,
        j3c_backend=j3c_backend,
        j2c_decomp_alg=j2c_decomp_alg,
        t2=mp.t2,
        dtype_cderi=dtype_cderi,
        log=log,
    )

    # handle results
    e_corr_os = result['e_corr_os']
    e_corr_ss = result['e_corr_ss']
    e_corr = e_corr_os + e_corr_ss + e_corr_singles

    mp.e_corr_singles = e_corr_singles
    mp.e_corr_os = e_corr_os
    mp.e_corr_ss = e_corr_ss
    mp.e_corr = e_corr
    return mp.e_corr


def _is_spin_restricted_open_shell(mo_coeff, mo_occ):
    """True for alpha and beta sharing orbitals with different occupations (ROHF)."""
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    return bool(not cp.array_equal(mo_occ[0], mo_occ[1])
                and cp.allclose(mo_coeff[0], mo_coeff[1]))


def _semicanonicalize(mp, mo_coeff, mo_occ, occ_coeff, vir_coeff):
    """Diagonalize the active occupied and virtual blocks of the UHF Fock matrix.

    Returns the rotated coefficients, the semicanonical orbital energies and the
    first-order singles energy sum_ia |f_ia|^2 / (e_i - e_a).
    """
    mf = mp._scf
    fock_ao = mf.get_fock(dm=mf.make_rdm1(cp.asarray(mo_coeff), cp.asarray(mo_occ)))
    occ_out, vir_out, eo_out, ev_out = [], [], [], []
    e_singles = 0.
    for spin in (0, 1):
        f = fock_ao[spin]
        eo, uo = cp.linalg.eigh(occ_coeff[spin].T @ f @ occ_coeff[spin])
        ev, uv = cp.linalg.eigh(vir_coeff[spin].T @ f @ vir_coeff[spin])
        co = occ_coeff[spin] @ uo
        cv = vir_coeff[spin] @ uv
        fov = co.T @ f @ cv
        e_singles += float((fov**2 / (eo[:, None] - ev[None, :])).sum())
        occ_out.append(co); vir_out.append(cv)
        eo_out.append(eo); ev_out.append(ev)
    return occ_out, vir_out, eo_out, ev_out, e_singles


class DFUMP2(GPUDFMP2):
    """Density-fitted UMP2. ROHF references are converted to UHF and treated in
    the semicanonical basis with first-order singles (matches PySCF UMP2)."""
    e_corr_singles = None
    _supports_rohf = True
    _keys = GPUDFMP2._keys | {'e_corr_singles'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, auxbasis=None):
        if mf.istype('ROHF') and not mf.istype('UHF'):
            mf = mf.to_uhf()
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ, auxbasis=auxbasis)

    get_nocc = pyscf.mp.ump2.get_nocc
    get_nmo = pyscf.mp.ump2.get_nmo
    get_frozen_mask = pyscf.mp.ump2.get_frozen_mask
    _kernel_impl = kernel
