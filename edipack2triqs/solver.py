from pathlib import Path
from tempfile import TemporaryDirectory
from warnings import warn
import re

import numpy as np
from mpi4py import MPI

import triqs.operators as op
from triqs.gf import BlockGf, Gf, MeshImFreq, MeshReFreq

from edipy2 import global_env as ed

from .util import IndicesType, validate_fops_up_dn, write_config, chdircontext
from .hamiltonian import parse_hamiltonian, BathNormal, BathHybrid, BathGeneral


class EDIpackSolver:

    # EDIpack maintains the state of a simulation as a set of global variables.
    # Therefore, this state must be controlled by at most one EDIpackSolver
    # instance at any time.
    instance_count = [0]

    # Default configuration
    default_config = {
        # DMFT
        "NLOOP": 0,
        "DMFT_ERROR": 0.0,
        "NSUCCESS": 0,
        # Hartree-Fock
        "HFMODE": False,
        "XMU": 0.0,
        # Phonons
        "PH_TYPE": 1,
        "NPH": 0,
        # Fixed density calculations
        "NREAD": 0.0,
        "NERR": 0.0,
        "NDELTA": 0.0,
        "NCOEFF": 0.0,
        # ED
        "ED_PRINT_SIGMA": False,
        "ED_PRINT_G": False,
        "ED_PRINT_G0": False,
        "ED_PRINT_SIGMA": False,
        "ED_FINITE_TEMP": True,
        "ED_TWIN": False,
        "ED_SECTORS": False,
        "ED_SECTORS_SHIFT": 0,
        "ED_SOLVE_OFFDIAG_GF": False,   # TODO
        "ED_ALL_G": True,
        "ED_OFFSET_BATH": 0.0,
        # TODO: Susceptibilities
        "LTAU": 1000,               # TODO: To be set in solve()
        "CHISPIN_FLAG": False,      # TODO: To be set in __init__()
        "CHIDENS_FLAG": False,      # TODO: To be set in __init__()
        "CHIPAIR_FLAG": False,      # TODO: To be set in __init__()
        "CHIEXCT_FLAG": False       # TODO: To be set in __init__()
    }

    def __init__(self,
                 hamiltonian: op.Operator,
                 fops_imp_up: list[IndicesType],
                 fops_imp_dn: list[IndicesType],
                 fops_bath_up: list[IndicesType],
                 fops_bath_dn: list[IndicesType],
                 **kwargs
                 ):
        """
        Initialize internal state of the underlying EDIpack solver.

        Parameters
        ----------
        hamiltonian: triqs.operators.Operator
            Many-body Hamiltonian to diagonalize
        fops_imp_up: list of tuples of strings and ints
            List of all spin-up impurity annihilation / creation operator
            flavors (indices)
        fops_imp_dn: list of tuples of strings and ints
            List of all spin-down impurity annihilation / creation operator
            flavors (indices)
        fops_bath_up: list of tuples of strings and ints
            List of all spin-up bath annihilation / creation operator
            flavors (indices)
        fops_bath_dn: list of tuples of strings and ints
            List of all spin-down bath annihilation / creation operator
            flavors (indices)
        input_file: str
            Path to an input file to be used by EDIpack. Mutually exclusive with
            all other keyword arguments.
        verbose: int, default 3
            Verbosity level: 0=almost nothing --> 5:all
        cutoff: float, default 1e-9
            Spectrum cutoff, used to determine the number states to be retained
        gs_threshold: float, default 1e-9
            Energy threshold for ground state degeneracy loop up
        ed_sparse_H: bool, default True
            Flag to select storage of sparse matrix H (True), or direct
            on-the-fly product H*v (False).
        lanc_method: str, default "arpack"
            Select the Lanczos method to be used in the determination of the
            spectrum: "arpack", "dvdson" (no MPI only).
        lanc_nstates_sector: int, default 2
            Initial number of states per sector to be determined
        lanc_nstates_total: int, default 2
            Initial total number of states to be determined
        lanc_nstates_step: int, default 2
            Number of states added to the spectrum at each step
        lanc_ncv_factor: int, default 10
            Set the size of the block used in Lanczos-ARPACK by multiplying
            the required Neigen (NCV = lanc_ncv_factor * Neigen + lanc_ncv_add)
        lanc_ncv_add: int, default 0
            Adds up to the size of the block to prevent it from becoming too
            small (NCV = lanc_ncv_factor * Neigen + lanc_ncv_add)
        lanc_niter: int, default 512
            Number of Lanczos iterations in spectrum determination
        lanc_ngfiter: int, default 200
            Number of Lanczos iterations in GF determination (number of momenta)
        lanc_tolerance: float, default 1e-18
            Tolerance for the Lanczos iterations as used in ARPACK
        lanc_dim_threshold: int, default 1024
            Min dimension threshold to use Lanczos determination of the spectrum
            rather than LAPACK based exact diagonalization
        """

        assert self.instance_count[0] < 1, \
            "Only one instance of EDIpackSolver can exist at any time"

        validate_fops_up_dn(fops_imp_up, fops_imp_dn,
                            "fops_imp_up", "fops_imp_dn")
        validate_fops_up_dn(fops_bath_up, fops_bath_dn,
                            "fops_bath_up", "fops_bath_dn")

        self.norb = len(fops_imp_up)
        assert self.norb <= 5, \
            f"At most 5 orbitals are allowed, got {self.norb}"

        self.fops_imp_up = fops_imp_up
        self.fops_imp_dn = fops_imp_dn
        self.fops_bath_up = fops_bath_up
        self.fops_bath_dn = fops_bath_dn

        # Detect GF block names
        block_names_up = [ind[0] for ind in fops_imp_up]
        block_names_dn = [ind[0] for ind in fops_imp_dn]
        if any(bn != block_names_up[0] for bn in block_names_up):
            warn(f"Inconsistent block names in {block_names_up}")
        if any(bn != block_names_dn[0] for bn in block_names_dn):
            warn(f"Inconsistent block names in {block_names_dn}")

        self.h_params = parse_hamiltonian(
            hamiltonian,
            self.fops_imp_up, self.fops_imp_dn,
            self.fops_bath_up, self.fops_bath_dn
        )
        self.nspin = self.h_params.Hloc.shape[0]

        if "input_file" in kwargs:
            self.input_file = Path(kwargs.pop("input_file"))
            self.input_file = self.input_file.resolve(strict=True)

            if kwargs:
                raise ValueError(
                    "'input_file' is mutually exclusive with other parameters"
                )
        else:
            self.input_file = None

            c = self.default_config.copy()
            c["ED_VERBOSE"] = kwargs.get("verbose", 3)
            c["CUTOFF"] = kwargs.get("cutoff", 1e-9)
            c["GS_THRESHOLD"] = kwargs.get("gs_threshold", 1e-9)
            c["ED_SPARSE_H"] = kwargs.get("ed_sparse_H", True)
            c["LANC_METHOD"] = kwargs.get("lanc_method", "arpack")
            c["LANC_NSTATES_SECTOR"] = kwargs.get("lanc_nstates_sector", 2)
            c["LANC_NSTATES_TOTAL"] = kwargs.get("lanc_nstates_total", 2)
            c["LANC_NSTATES_STEP"] = kwargs.get("lanc_nstates_step", 2)
            c["LANC_NCV_FACTOR"] = kwargs.get("lanc_ncv_factor", 10)
            c["LANC_NCV_ADD"] = kwargs.get("lanc_ncv_add", 0)
            c["LANC_NITER"] = kwargs.get("lanc_niter", 512)
            c["LANC_NGFITER"] = kwargs.get("lanc_ngfiter", 200)
            c["LANC_TOLERANCE"] = kwargs.get("lanc_tolerance", 1e-18)
            c["LANC_DIM_THRESHOLD"] = kwargs.get("lanc_dim_threshold", 1024)

            # Impurity structure
            c["ED_MODE"] = self.h_params.ed_mode
            c["NSPIN"] = self.nspin
            c["NORB"] = self.norb

            # Bath geometry
            c["BATH_TYPE"] = self.h_params.bath.name
            c["NBATH"] = self.h_params.bath.nbath

            # Interaction parameters
            c["ULOC"] = self.h_params.Uloc
            c["UST"] = self.h_params.Ust
            c["JH"] = self.h_params.Jh
            c["JX"] = self.h_params.Jx
            c["JP"] = self.h_params.Jp

            # ed_total_ud
            if isinstance(self.h_params.bath, BathNormal):
                c["ED_TOTAL_UD"] = not (self.h_params.Jx == 0
                                        and self.h_params.Jp == 0)
            elif isinstance(self.h_params.bath, (BathHybrid, BathGeneral)):
                c["ED_TOTAL_UD"] = True
            else:
                raise RuntimeError("Unrecognized bath type")

            self.config = c

        self.workdir = TemporaryDirectory()

        with chdircontext(self.workdir.name):
            if self.input_file is None:
                self.input_file = Path('input.conf').resolve()
                with open(self.input_file, 'w') as config_file:
                    write_config(config_file, self.config)

            else:
                Path('input.conf').symlink_to(self.input_file)

            ed.read_input('input.conf')

            if MPI.COMM_WORLD.Get_rank() == 0:
                self.scifor_version = re.match(
                    r"^SCIFOR VERSION \(GIT\): (.*)",
                    open("scifor_version.inc", 'r').readline())[1]

                self.edipack_version = re.match(
                    r"^code VERSION: (.*)",
                    open("code_version.inc", 'r').readline())[1]
            else:
                self.scifor_version = ""
                self.edipack_version = ""

            self.scifor_version = MPI.COMM_WORLD.bcast(self.scifor_version)
            self.edipack_version = MPI.COMM_WORLD.bcast(self.edipack_version)

        if isinstance(self.h_params.bath, BathGeneral):
            ed.set_hgeneral(self.h_params.bath.hvec,
                            self.h_params.bath.lambdavec)
        else:
            assert self.h_params.bath.data.size == ed.get_bath_dimension()

        # Initialize EDIpack
        ed.init_solver(bath=np.zeros(self.h_params.bath.data.size, dtype=float))

        # GF block names
        if ed.get_ed_mode() in (1, 2):  # normal or superc
            self.gf_block_names = (block_names_up[0], block_names_dn[0])
        else:
            self.gf_block_names = (block_names_up[0],)

        if ed.get_ed_mode() == 2:  # superc
            self.gf_an_block_names = (block_names_up[0]
                                      + "_" + block_names_dn[0],)

        self.instance_count[0] += 1

    def __del__(self):
        try:
            ed.finalize_solver()
            self.instance_count[0] -= 1
        # ed.finalize_solver() can fail if this __del__() method is called as
        # part of interpreter destruction procedure.
        except TypeError:
            pass

    def update_int_params(self, **kwargs):
        """
        Update interaction parameters.

        Parameters
        ----------
        Uloc: numpy.ndarray
            Values of the local interaction per orbital (max 5)
        Ust: float
            Value of the inter-orbital interaction term
        Jh: float
            Hund's coupling
        Jx: float
            Spin-exchange coupling
        Jp: float
            Pair-hopping coupling
        """

        if 'Uloc' in kwargs:
            Uloc = kwargs.pop("Uloc")
            assert len(Uloc) == self.norb, \
                "Required exactly {self.norb} values in Uloc"
            Uloc_ = np.zeros(5, dtype=float)
            Uloc_[:self.norb] = Uloc
            ed.Uloc = Uloc_

        ed.Ust = kwargs.pop("Ust", ed.Ust)
        ed.Jh = kwargs.pop("Jh", ed.Jh)
        ed.Jx = kwargs.pop("Jx", ed.Jx)
        ed.Jp = kwargs.pop("Jp", ed.Jp)

        if not ed.ed_total_ud:
            if ed.Jx != 0:
                raise RuntimeError("Cannot set Jx to a non-zero value")
            if ed.Jp != 0:
                raise RuntimeError("Cannot set Jp to a non-zero value")

        if len(kwargs) > 0:
            raise RuntimeError("Unrecognized interaction parameters: "
                               + ', '.join(map(str, kwargs.keys()))
                               )

    @property
    def hloc(self):
        "Access the local impurity Hamiltonian H_loc"
        return self.h_params.Hloc

    @property
    def bath(self):
        "Access the bath object"
        return self.h_params.bath

    def solve(self,
              beta: float,
              *,
              n_iw: int = 4096,
              energy_window: tuple[float, float] = (-5.0, 5.0),
              n_w: int = 5000,
              broadening: float = 0.01):
        """
        Solve the impurity problem.

        Parameters
        ----------
        beta: float
            Inverse temperature for observable and GF calculations
        n_iw: int, default 4096
            Number of Matsubara frequencies for impurity GF calculations
        energy_window: (float, float), default (-5.0, 5.0)
            Energy window for real-frequency impurity GF calculations
        n_w: int, default 5000
            Number of real-frequency points for impurity GF calculations
        broadening: float, default 0.01
            Broadening (eps) on the real axis
        TODO: Ltau
        """

        # Pass parameters to EDIpack
        ed.beta = beta
        ed.Lmats = n_iw
        ed.wini, ed.wfin = energy_window
        ed.Lreal = n_w
        ed.eps = broadening

        # Solve!
        with chdircontext(self.workdir.name):
            ed.set_hloc(hloc=self.h_params.Hloc)
            ed.solve(self.h_params.bath.data)

    @property
    def energies(self):
        "Returns the impurity local energies components"
        return ed.get_eimp()

    @property
    def densities(self):
        "Returns the impurity occupations, one element per orbital"
        return ed.get_dens()

    @property
    def double_occ(self):
        "Returns the impurity double occupancy, one element per orbital"
        return ed.get_docc()

    def magnetization(self, comp: str = 'z'):
        """
        Returns a component of the impurity magnetization vector, one element
        per orbital. The component selector 'comp' can be one of 'x', 'y'
        and 'z'.
        """
        return ed.get_mag(icomp=comp)

    def _make_gf(self, ed_func, real_freq, anomalous):
        if anomalous:
            if ed.get_ed_mode() != 2:  # superc
                raise RuntimeError("anomalous = True is only supported for "
                                   "superconducting bath")

        if real_freq:
            mesh = MeshReFreq(window=(ed.wini, ed.wfin), n_w=ed.Lreal)
            z_vals = [complex(z) + ed.eps * 1j for z in mesh]
        else:
            mesh = MeshImFreq(beta=ed.beta, S="Fermion", n_iw=ed.Lmats)
            z_vals = [complex(z) for z in mesh]

        with chdircontext(self.workdir.name):
            data = ed_func(z_vals, typ='a' if anomalous else 'n')

        if anomalous:
            F = Gf(mesh=mesh, target_shape=(self.norb, self.norb))
            F.data[:] = np.rollaxis(data[0, 0, :, :, :], 2)
            return BlockGf(name_list=self.gf_an_block_names,
                           block_list=[F],
                           make_copies=False)

        if ed.get_ed_mode() in (1, 2):  # 2 spin blocks
            blocks = [
                Gf(mesh=mesh, target_shape=(self.norb, self.norb))
                for _ in range(2)
            ]
            # Block up
            blocks[0].data[:] = np.rollaxis(data[0, 0, :, :, :], 2)
            # Block down
            if self.nspin == 1:
                blocks[1].data[:] = blocks[0].data
            else:
                blocks[1].data[:] = np.rollaxis(data[1, 1, :, :, :], 2)
        else:  # One block
            blocks = [
                Gf(mesh=mesh, target_shape=(2 * self.norb, 2 * self.norb))
            ]
            blocks[0].data[:] = np.transpose(data, (4, 0, 2, 1, 3)).reshape(
                len(mesh), 2 * self.norb, 2 * self.norb
            )

        return BlockGf(name_list=self.gf_block_names,
                       block_list=blocks,
                       make_copies=False)

    #
    # GF and self-energy properties
    #

    g_iw = property(lambda s: s._make_gf(ed.build_gimp, False, False),
                    doc="Matsubara impurity Green's function")
    g_an_iw = property(lambda s: s._make_gf(ed.build_gimp, False, True),
                       doc="Anomalous Matsubara impurity Green's function")
    Sigma_iw = property(lambda s: s._make_gf(ed.build_sigma, False, False),
                        doc="Matsubara impurity self-energy")
    Sigma_an_iw = property(lambda s: s._make_gf(ed.build_sigma, False, True),
                           doc="Anomalous Matsubara impurity self-energy")

    g_w = property(lambda s: s._make_gf(ed.build_gimp, True, False),
                   doc="Real-frequency impurity Green's function")
    g_an_w = property(lambda s: s._make_gf(ed.build_gimp, True, True),
                      doc="Anomalous real-frequency impurity Green's function")
    Sigma_w = property(lambda s: s._make_gf(ed.build_sigma, True, False),
                       doc="Real-frequency impurity self-energy")
    Sigma_an_w = property(lambda s: s._make_gf(ed.build_sigma, True, True),
                          doc="Anomalous real-frequency impurity self-energy")
