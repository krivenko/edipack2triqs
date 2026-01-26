#!/bin/env python

#
# DMFT study of superconductivity in the attractive Hubbard model
#

from itertools import product
import numpy as np

# TRIQS modules
from triqs.gf import (Gf, BlockGf, MeshImFreq, MeshProduct,
                      iOmega_n, conjugate, inverse)
from triqs.gf.tools import dyson
from triqs.operators import c, c_dag, n, dagger
from triqs.lattice.tight_binding import TBLattice
from h5 import HDFArchive

# edipack2triqs modules
from edipack2triqs.solver import EDIpackSolver, LanczosParams
from edipack2triqs.fit import BathFittingParams


# Parameters
Nspin = 1
Nloop = 100
Nsuccess = 1
threshold = 1e-6
wmixing = 0.5
Norb = 1
Nbath = 4
n_k = 20
U = -3.0      # Local intra-orbital interactions U
Ust = 1.2     # Local inter-orbital interaction U'
Jh = 0.2      # Hund's coupling
Jx = 0.15     # Spin-exchange coupling constant
Jp = 0.1      # Pair-hopping coupling constant
beta = 100.0  # Inverse temperature
n_iw = 1024   # Number of Matsubara frequencies for impurity GF calculations
t = 0.5
# Energy window for real-frequency GF calculations
energy_window = (-2.0 * t, 2.0 * t)
n_w = 4000    # Number of real-frequency points for impurity GF calculations
broadening = 0.005  # Broadening on the real axis for impurity GF calculations

spins = ('up', 'dn')
orbs = range(Norb)

# Fundamental sets for impurity degrees of freedom
fops_imp_up = [('up', o) for o in orbs]
fops_imp_dn = [('dn', o) for o in orbs]

# Fundamental sets for bath degrees of freedom
fops_bath_up = [('B_up', i) for i in range(Norb * Nbath)]
fops_bath_dn = [('B_dn', i) for i in range(Norb * Nbath)]

# Non-interacting part of the impurity Hamiltonian
xmu = U / 2 + (Norb - 1) * Ust / 2 + (Norb - 1) * (Ust - Jh) / 2

h_loc = -xmu * np.eye(Norb)
H = sum(h_loc[o1, o2] * c_dag(spin, o1) * c(spin, o2)
        for spin, o1, o2 in product(spins, orbs, orbs))

# Interaction part
H += U * sum(n('up', o) * n('dn', o) for o in orbs)
H += Ust * sum(int(o1 != o2) * n('up', o1) * n('dn', o2)
               for o1, o2 in product(orbs, orbs))
H += (Ust - Jh) * sum(int(o1 < o2) * n(s, o1) * n(s, o2)
                      for s, o1, o2 in product(spins, orbs, orbs))
H -= Jx * sum(int(o1 != o2)
              * c_dag('up', o1) * c('dn', o1) * c_dag('dn', o2) * c('up', o2)
              for o1, o2 in product(orbs, orbs))
H += Jp * sum(int(o1 != o2)
              * c_dag('up', o1) * c_dag('dn', o1) * c('dn', o2) * c('up', o2)
              for o1, o2 in product(orbs, orbs))

# Lattice Hamiltonian
units = [(1, 0, 0), (0, 1, 0)]
orbital_positions = [(0.0, 0.0, 0.0) for i in range(Norb)]
hoppings = {(1, 0): t * np.eye(Norb),
            (-1, 0): t * np.eye(Norb),
            (0, 1): t * np.eye(Norb),
            (0, -1): t * np.eye(Norb)}

TBL = TBLattice(units=units,
                hoppings=hoppings,
                orbital_positions=orbital_positions)
kmesh = TBL.get_kmesh(n_k=(n_k, n_k, 1))
enk = TBL.fourier(kmesh)

nambu_shape = (2 * Norb, 2 * Norb)
h0_nambu_k = Gf(mesh=kmesh, target_shape=nambu_shape)
for k in kmesh:
    h0_nambu_k[k][:Norb, :Norb] = enk(k)
    h0_nambu_k[k][Norb:, Norb:] = -enk(-k)

# Matrix dimensions of eps and V: 3 orbitals x 2 bath states
eps = np.array([[-1.0, -0.5, 0.5, 1.0] for i in range(Norb)])
V = 0.5 * np.ones((Norb, Nbath))
D = -0.2 * np.eye(Norb * Nbath)

# Bath
H += sum(eps[o, nu]
         * c_dag("B_" + s, o * Nbath + nu) * c("B_" + s, o * Nbath + nu)
         for s, o, nu in product(spins, orbs, range(Nbath)))

H += sum(V[o, nu] * (c_dag(s, o) * c("B_" + s, o * Nbath + nu)
                     + c_dag("B_" + s, o * Nbath + nu) * c(s, o))
         for s, o, nu in product(spins, orbs, range(Nbath)))

# Anomalous bath
H += sum(D[o, q] * (c('B_up', o) * c('B_dn', q))
         + dagger(D[o, q] * (c('B_up', o) * c('B_dn', q)))
         for o, q in product(range(Norb * Nbath), range(Norb * Nbath)))

# Create solver object
fit_params = BathFittingParams(method="minimize", grad="numeric")
solver = EDIpackSolver(H,
                       fops_imp_up, fops_imp_dn, fops_bath_up, fops_bath_dn,
                       lanczos_params=LanczosParams(dim_threshold=1024),
                       verbose=1,
                       bath_fitting_params=fit_params)


# Compute local GF from bare lattice Hamiltonian and self-energy
def get_gloc(s, s_an):
    z = Gf(mesh=s.mesh, target_shape=nambu_shape)
    if isinstance(s.mesh, MeshImFreq):
        z[:Norb, :Norb] << iOmega_n + xmu - s
        z[:Norb, Norb:] << -s_an
        z[Norb:, :Norb] << -s_an
        z[Norb:, Norb:] << iOmega_n - xmu + conjugate(s)
    else:
        z[:Norb, Norb:] << -s_an
        z[Norb:, :Norb] << -s_an
        for w in z.mesh:
            z[w][:Norb, :Norb] = \
                (w + 1j * broadening + xmu) * np.eye(Norb) - s[w]
            z[w][Norb:, Norb:] = \
                (w + 1j * broadening - xmu) * np.eye(Norb) + conjugate(s(-w))

    g_k = Gf(mesh=MeshProduct(kmesh, z.mesh), target_shape=nambu_shape)
    for k in kmesh:
        g_k[k, :] << inverse(z - h0_nambu_k[k])

    g_loc_nambu = sum(g_k[k, :] for k in kmesh) / len(kmesh)

    g_loc = s.copy()
    g_loc_an = s_an.copy()
    g_loc[:] = g_loc_nambu[:Norb, :Norb]
    g_loc_an[:] = g_loc_nambu[:Norb, Norb:]
    return g_loc, g_loc_an


# Compute Weiss field from local GF and self-energy
def dmft_weiss_field(g_iw, g_an_iw, s_iw, s_an_iw):
    g_nambu_iw = Gf(mesh=g_iw.mesh, target_shape=nambu_shape)
    s_nambu_iw = Gf(mesh=s_iw.mesh, target_shape=nambu_shape)

    g_nambu_iw[:Norb, :Norb] = g_iw
    g_nambu_iw[:Norb, Norb:] = g_an_iw
    g_nambu_iw[Norb:, :Norb] = g_an_iw
    g_nambu_iw[Norb:, Norb:] = -conjugate(g_iw)

    s_nambu_iw[:Norb, :Norb] = s_iw
    s_nambu_iw[:Norb, Norb:] = s_an_iw
    s_nambu_iw[Norb:, :Norb] = s_an_iw
    s_nambu_iw[Norb:, Norb:] = -conjugate(s_iw)

    g0_nambu_iw = dyson(G_iw=g_nambu_iw, Sigma_iw=s_nambu_iw)

    g0_iw = g_iw.copy()
    g0_an_iw = g_an_iw.copy()
    g0_iw[:] = g0_nambu_iw[:Norb, :Norb]
    g0_an_iw[:] = g0_nambu_iw[:Norb, Norb:]
    return g0_iw, g0_an_iw


#
# DMFT loop
#

gooditer = 0
g0_prev = np.zeros((2, 2 * n_iw, Norb, Norb), dtype=complex)
for iloop in range(Nloop):
    print(f"\nLoop {iloop + 1} of {Nloop}")

    # Solve the effective impurity problem
    solver.solve(beta=beta,
                 n_iw=n_iw,
                 energy_window=energy_window,
                 n_w=n_w,
                 broadening=broadening)

    # Normal and anomalous components of computed self-energy
    s_iw = solver.Sigma_iw["up"]
    s_an_iw = solver.Sigma_an_iw["up_dn"]

    # Compute local Green's function
    g_iw, g_an_iw = get_gloc(s_iw, s_an_iw)
    # Compute Weiss field
    g0_iw, g0_an_iw = dmft_weiss_field(g_iw, g_an_iw, s_iw, s_an_iw)

    # Bath fitting and mixing
    G0_iw = BlockGf(name_list=spins, block_list=[g0_iw, g0_iw])
    G0_an_iw = BlockGf(name_list=["up_dn"], block_list=[g0_an_iw])

    bath_new = solver.chi2_fit_bath(G0_iw, G0_an_iw)[0]
    solver.bath = wmixing * bath_new + (1 - wmixing) * solver.bath

    # Check convergence of the Weiss field

    g0 = np.asarray([g0_iw.data, g0_an_iw.data])
    errvec = np.real(np.sum(abs(g0 - g0_prev), axis=1)
                     / np.sum(abs(g0), axis=1))
    # First iteration
    if iloop == 0:
        errvec = np.ones_like(errvec)
    errmin, err, errmax = np.min(errvec), np.average(errvec), np.max(errvec)

    g0_prev = np.copy(g0)

    if err < threshold:
        gooditer += 1  # Increase good iterations count
    else:
        gooditer = 0  # Reset good iterations count
    iloop += 1
    conv_bool = ((err < threshold) and (gooditer > Nsuccess)
                 and (iloop < Nloop)) or (iloop >= Nloop)

    # Print convergence message
    if iloop < Nloop:
        if errvec.size > 1:
            print(f"max error={errmax:.6e}")
        print("    " * (errvec.size > 1) + f"error={err:.6e}")
        if errvec.size > 1:
            print(f"min error={errmin:.6e}")
    else:
        if errvec.size > 1:
            print(f"max error={errmax:.6e}")
        print("    " * (errvec.size > 1) + f"error={err:.6e}")
        if errvec.size > 1:
            print(f"min error={errmin:.6e}")
        print(f"Not converged after {Nloop} iterations.")

    if conv_bool:
        break


# Calculate local Green's function on the real axis
s_w = solver.Sigma_w["up"]
s_an_w = solver.Sigma_an_w["up_dn"]
g_w, g_an_w = get_gloc(s_w, s_an_w)

# Save calculation results
with HDFArchive('ahm.h5', 'w') as ar:
    ar["s_iw"] = s_iw
    ar["s_an_iw"] = s_an_iw
    ar["g_iw"] = g_iw
    ar["g_an_iw"] = g_an_iw
    ar["g_w"] = g_w
    ar["g_an_w"] = g_an_w

print("Done...")
