#!/bin/env python

#
# 3 orbital impurity with Hubbard-Kanamori interaction
# 2 bath states per orbital
#

from itertools import product
import numpy as np
from mpi4py import MPI

# TRIQS many-body operator objects
from triqs.operators import c, c_dag, n

# Compatibility layer between EDIpack and TRIQS
from edipack2triqs.solver import EDIpackSolver

# Fundamental sets for impurity degrees of freedom
fops_imp_up = [('up', 0), ('up', 1), ('up', 2)]
fops_imp_dn = [('dn', 0), ('dn', 1), ('dn', 2)]

# Fundamental sets for bath degrees of freedom
fops_bath_up = [('B_up', i) for i in range(3 * 2)]
fops_bath_dn = [('B_dn', i) for i in range(3 * 2)]

# Define the Hamiltonian
orbs = range(3)

## Non-interacting part of the impurity Hamiltonian
h_loc = np.diag([-0.7, -0.5, -0.7])
H = sum(h_loc[o1, o2] * c_dag(spin, o1) * c(spin, o2)
        for spin, o1, o2 in product(('up', 'dn'), orbs, orbs))

## Interaction part
U = 3.0     # Local intra-orbital interactions U
Ust = 1.2   # Local inter-orbital interaction U'
Jh = 0.2    # Hund's coupling
Jx = 0.15   # Spin-exchange coupling constant
Jp = 0.1    # Pair-hopping coupling constant

H += U * sum(n('up', o) * n('dn', o) for o in orbs)
H += Ust * sum(int(o1 != o2) * n('up', o1) * n('dn', o2)
               for o1, o2 in product(orbs, orbs))
H += (Ust - Jh) * sum(int(o1 < o2) * n(s, o1) * n(s, o2)
                      for s, o1, o2 in product(('up', 'dn'), orbs, orbs))
H -= Jx * sum(int(o1 != o2)
              * c_dag('up', o1) * c('dn', o1) * c_dag('dn', o2) * c('up', o2)
              for o1, o2 in product(orbs, orbs))
H += Jp * sum(int(o1 != o2)
              * c_dag('up', o1) * c_dag('dn', o1) * c('dn', o2) * c('up', o2)
              for o1, o2 in product(orbs, orbs))

## Bath part

# Matrix dimensions of eps and V: 3 orbitals x 2 bath states
eps = np.array([[-0.1, 0.1],
                [-0.2, 0.2],
                [-0.3, 0.3]])
V = np.array([[0.1, 0.2],
              [0.1, 0.2],
              [0.1, 0.2]])

H += sum(eps[o, nu] * c_dag("B_" + s, nu * 3 + o) * c("B_" + s, nu * 3 + o)
         for s, o, nu in product(('up', 'dn'), orbs, range(2)))
H += sum(V[o, nu] * (c_dag(s, o) * c("B_" + s, nu * 3 + o)
                     + c_dag("B_" + s, nu * 3 + o) * c(s, o))
         for s, o, nu in product(('up', 'dn'), orbs, range(2)))

# Create a solver object that wraps EDIpack functionality
# See help(EDIpackSolver) for a complete list of parameters
solver = EDIpackSolver(H,
                       fops_imp_up, fops_imp_dn, fops_bath_up, fops_bath_dn,
                       lanc_dim_threshold=16)

# Solve the impurity model
beta = 100.0  # Inverse temperature
n_iw = 1024   # Number of Matsubara frequencies for impurity GF calculations
energy_window = (-2.0, 2.0)  # Energy window for real-frequency GF calculations
n_w = 4000    # Number of real-frequency points for impurity GF calculations
broadening = 0.005  # Broadening on the real axis for impurity GF calculations

solver.solve(beta=beta,
             n_iw=n_iw,
             energy_window=energy_window,
             n_w=n_w,
             broadening=broadening)

# On master node, output results of calculation
if MPI.COMM_WORLD.Get_rank() == 0:
    print("Using SciFortran", solver.scifor_version)
    print("Using EDIpack", solver.edipack_version)

    print("Densities (per orbital):", solver.densities())
    print("Double occupancy (per orbital):", solver.double_occ())
    print("Magnetization (per orbital):", solver.magnetization())

    from triqs.plot.mpl_interface import plt, oplot

    # Extract the Matsubara GF from the solver and plot it
    g_iw = solver.g_iw()

    plt.figure()
    for spin in ('up', 'dn'):
        for orb in range(3):
            label = r"g_{%s,%i%i}(i\omega_n)$" % (spin, orb, orb)
            oplot(g_iw[spin][orb, orb].real, label=r"$\Re " + label)
            oplot(g_iw[spin][orb, orb].imag, label=r"$\Im " + label)
    plt.legend(loc='lower center', ncols=3)
    plt.savefig("g_iw.pdf")

    # Extract the Matsubara self-energy from the solver and plot it
    Sigma_iw = solver.Sigma_iw()

    plt.figure()
    for spin in ('up', 'dn'):
        for orb in range(3):
            label = r"\Sigma_{%s,%i%i}(i\omega_n)$" % (spin, orb, orb)
            oplot(Sigma_iw[spin][orb, orb].real, label=r"$\Re " + label)
            oplot(Sigma_iw[spin][orb, orb].imag, label=r"$\Im " + label)
    plt.legend(loc='lower center', ncols=3)
    plt.savefig("Sigma_iw.pdf")

    # Extract the real-frequency GF from the solver and plot it
    g_w = solver.g_w()

    plt.figure()
    for spin in ('up', 'dn'):
        for orb in range(3):
            label = r"g_{%s,%i%i}(\omega)$" % (spin, orb, orb)
            oplot(g_w[spin][orb, orb].real, label=r"$\Re " + label)
            oplot(g_w[spin][orb, orb].imag, label=r"$\Im " + label)
    plt.legend(loc='lower center', ncols=3)
    plt.savefig("g_w.pdf")

    # Extract the real-frequency self-energy from the solver and plot it
    Sigma_w = solver.Sigma_w()

    plt.figure()
    for spin in ('up', 'dn'):
        for orb in range(3):
            label = r"\Sigma_{%s,%i%i}(\omega)$" % (spin, orb, orb)
            oplot(Sigma_w[spin][orb, orb].real, label=r"$\Re " + label)
            oplot(Sigma_w[spin][orb, orb].imag, label=r"$\Im " + label)
    plt.legend(loc='lower center', ncols=3)
    plt.savefig("Sigma_w.pdf")

# It is possible to change some parameters of the Hamiltonian and to diagonalize
# it again.

# Change a matrix element of h_loc
# Since the non-interacting part of the original Hamiltonian is spin-symmetric,
# solver.hloc() returns an array of shape (1, 1, 3, 3). In presence of a spin
# energy splitting the shape would be (2, 2, 3, 3).
solver.hloc()[0, 0, 1, 1] = -0.6  # spin1 = spin2 = up and down, orb1 = orb2 = 1

# Change Jp
solver.update_int_params(Jp=0.15)

# Update some bath parameters
solver.bath().eps[0, 2, 1] = -0.4  # spin = up and down, orb1 = 2, orb2 = 1
solver.bath().V[0, 0, 1] = 0.3     # spin = up and down, orb1 = 0, orb2 = 1

solver.solve(beta=beta)
