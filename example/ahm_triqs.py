import numpy as np
import scipy as sp
import mpi4py
from mpi4py import MPI
import os,sys
import matplotlib.pyplot as plt
from aux_funx import *
from itertools import product

# TRIQS many-body operator objects
from triqs.gf import *
from triqs.operators import c, c_dag, n, dagger
from triqs.lattice.tight_binding import TBLattice
from triqs.sumk import *
from triqs_tprf.lattice import lattice_dyson_g_w
from edipack2triqs.solver import EDIpackSolver

#INIT MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("I am process",rank,"of",comm.Get_size())
master = (rank==0)

#Parameters
Nspin = 1
Nloop = 100
Nsuccess = 1
error = 1e-6
wmixing = 0.5
Norb  = 1
Nbath = 4
Nk = 20
U = -3.0     # Local intra-orbital interactions U
Ust = 1.2   # Local inter-orbital interaction U'
Jh = 0.2    # Hund's coupling
Jx = 0.15   # Spin-exchange coupling constant
Jp = 0.1    # Pair-hopping coupling constant
beta = 100.0  # Inverse temperature
n_iw = 1024   # Number of Matsubara frequencies for impurity GF calculations
t = 0.5
energy_window = (-2.0*t, 2.0*t)  # Energy window for real-frequency GF calculations
n_w = 4000    # Number of real-frequency points for impurity GF calculations
broadening = 0.005  # Broadening on the real axis for impurity GF calculations
use_ph = False

if use_ph:
    Nspin = 2
    U = -U              #U becomes minus U
    Ust = -Ust          #Uprime becomes minus Uprime
    Jx, Jp = Jp, Jx     #Jp becomes Jx and vice versa
    Jh = -2.0*Ust + Jh  #For the U" term, Edipack uses Uprime-Jh.   This is transformed to itself. But Ust has changed to -Ust. So, Jh has to change to -2 Ust + Jh

# Fundamental sets for impurity degrees of freedom
fops_imp_up = [('up', i) for i in range(Norb)]
fops_imp_dn = [('dn', i) for i in range(Norb)]

# Fundamental sets for bath degrees of freedom
fops_bath_up = [('B_up', i) for i in range(Norb*Nbath)]
fops_bath_dn = [('B_dn', i) for i in range(Norb*Nbath)]

## Non-interacting part of the impurity Hamiltonian
orbs = range(Norb)
xmu = - (U/2.0 + (Norb-1)*Ust/2.0 + (Norb-1)*(Ust-Jh)/2.0)

h_loc = np.diag([xmu for i in range(Norb)])
H = sum(h_loc[o1, o2] * c_dag(spin, o1) * c(spin, o2)
        for spin, o1, o2 in product(('up', 'dn'), orbs, orbs))
 
       
## Interaction part
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
              
#Lattice Hamiltonian

units = [(1,0,0),(0,1,0)]

hoppings = {( 1,0) : t*np.eye(Norb),
            (-1,0) : t*np.eye(Norb),
            ( 0,1) : t*np.eye(Norb),
            (0,-1) : t*np.eye(Norb)}

orbital_positions = [(0.0, 0.0, 0.0) for i in range(Norb)]

TBL = TBLattice(units=units, hoppings=hoppings, orbital_positions=orbital_positions)
kmesh = TBL.get_kmesh(n_k = (Nk,Nk,1))
enk = TBL.fourier(kmesh)

h0_k = Gf(mesh=kmesh, target_shape=[2*Norb,2*Norb])
h0_k.data[:] = 0.0

#Up block
h0_k.data[:,0:Norb,0:Norb] = enk.data[:,0:Norb,0:Norb]

#Dn block
for k in kmesh:
    ki = k.data_index
    h0_k.data[ki,Norb:2*Norb, Norb:2*Norb] = -enk(-k.value)[0:Norb,0:Norb]
    
    
    

#edipack convention    
hk_nambu=np.zeros((2,np.shape(h0_k.data)[0],Norb,Norb),dtype=complex)
hk_nambu[0,:,0:Norb,0:Norb] = h0_k.data[:,0:Norb,0:Norb]
hk_nambu[1,:,0:Norb,0:Norb] = h0_k.data[:,Norb:2*Norb, Norb:2*Norb]


#frequencies
mesh = MeshImFreq(beta=100, S="Fermion", n_iw=n_iw)
z_vals = np.array([complex(z) for z in mesh])

             
# Matrix dimensions of eps and V: 3 orbitals x 2 bath states
eps = np.array([[-1.0, -0.5, 0.5, 1.0] for i in range(Norb)])
V = np.array([[0.5 for i in range(Nbath)] for j in range(Norb)])
D = np.diag([-0.2 for i in range(Norb*Nbath)])

#Up bath
H += sum(eps[o, nu] * c_dag("B_up", o * Nbath + nu) * c("B_up", o * Nbath + nu)
         for o, nu in product(orbs, range(Nbath)))
         
H += sum(V[o, nu] * (c_dag("up", o) * c("B_up", o * Nbath + nu)
                     + c_dag("B_up", o * Nbath + nu) * c("up", o))
         for o, nu in product( orbs, range(Nbath)))

#Down bath    
if use_ph:
    H += sum(eps[o, nu] * c("B_dn", o * Nbath + nu) * c_dag("B_dn", o * Nbath + nu)
             for o, nu in product(orbs, range(Nbath)))
             
    H += sum(V[o, nu] * (c("dn", o) * c_dag("B_dn", o * Nbath + nu)
                         + c("B_dn", o * Nbath + nu) * c_dag("dn", o))
             for o, nu in product(orbs, range(Nbath)))
else:
    H += sum(eps[o, nu] * c_dag("B_dn", o * Nbath + nu) * c("B_dn", o * Nbath + nu)
             for o, nu in product(orbs, range(Nbath)))
             
    H += sum(V[o, nu] * (c_dag("dn", o) * c("B_dn", o * Nbath + nu)
                         + c_dag("B_dn", o * Nbath + nu) * c("dn", o))
             for o, nu in product(orbs, range(Nbath)))


#Anomalous bath   
if use_ph:
    H += sum(D[o, q] * (c('B_up', o) * c_dag('B_dn', q))
                         + dagger(D[o, q] * (c('B_up', o) * c_dag('B_dn', q)))
             for o, q in product(range(Norb*Nbath), range(Norb*Nbath)))
else:
    H += sum(D[o, q] * (c('B_up', o) * c('B_dn', q))
                         + dagger(D[o, q] * (c('B_up', o) * c('B_dn', q)))
             for o, q in product(range(Norb*Nbath), range(Norb*Nbath)))

         
# Create solver object
solver = EDIpackSolver(H,
                       fops_imp_up, fops_imp_dn, fops_bath_up, fops_bath_dn,
                       lanc_dim_threshold=1024,verbose=1,cg_method=1,cg_grad=1)

converged=False;iloop=0
while (not converged and iloop<Nloop ):
    iloop=iloop+1
    print(" ")
    print("Loop "+str(iloop)+" of "+str(Nloop))

    #Solve the EFFECTIVE IMPURITY PROBLEM (first w/ a guess for the bath)
    solver.solve(beta=beta,
                 n_iw=n_iw,
                 energy_window=energy_window,
                 n_w=n_w,
                 broadening=broadening,
                 )
    if use_ph:
        S_iw = solver.Sigma_iw
        Gloc   = lattice_dyson_g_w(mu=0, e_k=h0_k, sigma_w=S_iw['up'])
        Weiss  = inverse(inverse(Gloc) + S_iw['up']) 
        fit_weiss_normal = solver.fit_weiss(Weiss,0.5)
    else:
        #edipack convention
        S_iwed = [solver.Sigma_iw,solver.Sigma_an_iw]       
        S_up = np.rollaxis(S_iwed[0]['up'].data,0,S_iwed[0]['up'].data.ndim)
        F_up = np.rollaxis(S_iwed[1]['up_dn'].data,0,S_iwed[1]['up_dn'].data.ndim)
        Smats = np.array([S_up[...,],F_up])             
        Gmats = get_gloc(z_vals,0.5*U,hk_nambu,Smats,axis="m")
        Weiss_edipack = dmft_weiss_field(Gmats,Smats)
        
        
        #triqs convention        
        list_names=[list(solver.Sigma_iw.indices)[0],list(solver.Sigma_an_iw.indices)[0]]
        S_iw = BlockGf(name_list=list_names,
                           block_list=[solver.Sigma_iw[list_names[0]],solver.Sigma_an_iw[list_names[1]]],
                           make_copies=False)
        Weiss = S_iw.copy()
        Weiss['up'].data[:,:,:] = np.transpose(Weiss_edipack[0,:,:,:],(2,0,1))
        Weiss['up_dn'].data[:,:,:] = np.transpose(Weiss_edipack[1,:,:,:],(2,0,1))

   
        np.savetxt('testsigma.dat', np.transpose([np.imag(z_vals),S_up[0,0,:].imag,S_up[0,0,:].real]))
        np.savetxt('testsanol.dat', np.transpose([np.imag(z_vals),F_up[0,0,:].imag,F_up[0,0,:].real]))
        np.savetxt('testgmats.dat', np.transpose([np.imag(z_vals),Gmats[0,0,0,:].imag,Gmats[0,0,0,:].real]))
        np.savetxt('testfmats.dat', np.transpose([np.imag(z_vals),Gmats[1,0,0,:].imag,Gmats[1,0,0,:].real]))
        np.savetxt('testweiss.dat', np.transpose([np.imag(z_vals),Weiss_edipack[0,0,0,:].imag,Weiss_edipack[0,0,0,:].real]))
        

        FittedWeiss = solver.fit_gf(Weiss,0.5)
        outfitted = np.transpose(FittedWeiss['up'].data,(1,2,0))
        np.savetxt('fittedweis.dat', np.transpose([np.imag(z_vals),outfitted[0,0,:].imag,outfitted[0,0,:].real]))
        
        err,converged = check_convergence(Weiss_edipack,error,Nsuccess,Nloop)


print("Done...")

