"""
TRIQS interface to **EDIpack** exact diagonalization solver.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import NoneType
from typing import Union
from warnings import warn
import os
import re
import sys

import numpy as np
from mpi4py import MPI

import triqs.operators as op
from triqs.gf import BlockGf, Gf, MeshImFreq, MeshReFreq, MeshImTime

from edipack2py import global_env as ed

from . import EDMode
from .util import (IndicesType,
                   validate_fops_up_dn,
                   is_spin_diagonal,
                   is_spin_degenerate,
                   write_config,
                   chdircontext)
from .bath import Bath, BathNormal, BathHybrid, BathGeneral
from .hamiltonian import (parse_hamiltonian,
                          extract_quadratic,
                          _is_density,
                          _is_density_density)
from .fit import BathFittingParams, _chi2_fit_bath


class EDIpackSolver:
    """
    This class represents the **EDIpack** exact diagonalization library and
    incapsulates its internal state. Its methods and attributes allow to perform
    ED calculations and to access their results.

    .. note::

        At most one instance of this type can exists at any time. An attempt to
        create more such objects will raise an :py:class:`AssertionError`.
    """

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
        "ED_USE_KANAMORI": False,
        "ED_READ_UMATRIX": False,
        "PRINT_SECTOR_EIGENVALUES": False,
        "ED_PRINT_SIGMA": False,
        "ED_PRINT_G": False,
        "ED_PRINT_G0": False,
        "RDM_FLAG": False,
        "ED_TWIN": False,
        "ED_SECTORS": False,
        "ED_SECTORS_SHIFT": 0,
        "ED_SOLVE_OFFDIAG_GF": False,   # TODO
        "ED_ALL_G": True,
        "ED_OFFSET_BATH": 0.0,
        "LTAU": 1000,
        "CHISPIN_FLAG": False,
        "CHIDENS_FLAG": False,
        "CHIPAIR_FLAG": False,
        "CHIEXCT_FLAG": False
    }

    def __init__(self,  # noqa: C901
                 hamiltonian: op.Operator,
                 fops_imp_up: list[IndicesType],
                 fops_imp_dn: list[IndicesType],
                 fops_bath_up: list[IndicesType] = [],
                 fops_bath_dn: list[IndicesType] = [],
                 **kwargs
                 ):
        r"""
        Initialize internal state of the underlying **EDIpack** solver.

        The fundamental operator sets (**fops** for short) define sizes of
        impurity and bath Hilbert spaces. The expected **fops** objects are
        lists of tuples (pairs). Each element ``(b, i)`` of such a list
        corresponds to a single fermionic degree of freedom created by
        the many-body operator ``c_dag(b, i)``. ``b`` and ``i`` are a (string or
        integer) block index and an index within a block respectively, which
        are used to construct output :py:class:`Green's function
        containers <triqs.gf.block_gf.BlockGf>`.

        :param hamiltonian: Many-body electronic Hamiltonian to diagonalize.
            Symmetries of this Hamiltonian are automatically analyzed and
            dictate the choice of **EDIpacks**'s
            :f:var:`ED mode <f/ed_input_vars/ed_mode>` and
            :f:mod:`bath geometry <f/ed_bath>`. This choice
            remains unchangeable throughout object's lifetime.
        :type hamiltonian: triqs.operators.operators.Operator

        :param fops_imp_up: Fundamental operator set for spin-up impurity
            degrees of freedom.
        :type fops_imp_up: list[tuple[int | str, int | str]]

        :param fops_imp_dn: Fundamental operator set for spin-down impurity
            degrees of freedom.
        :type fops_imp_dn: list[tuple[int | str, int | str]]

        :param fops_bath_up: Fundamental operator set for spin-up bath
            degrees of freedom. Must be empty for calculations without bath.
        :type fops_bath_up: list[tuple[int | str, int | str]], optional,
            default=[]

        :param fops_bath_dn: Fundamental operator set for spin-down bath
            degrees of freedom. Must be empty for calculations without bath.
        :type fops_bath_dn: list[tuple[int | str, int | str]], optional,
            default=[]

        :param input_file: Path to a custom input file compatible with
            **EDIpack**'s :f:func:`f/ed_input_vars/ed_read_input`.
            When specified, it is used to initialize **EDIpack**.
            This option is mutually exclusive with all other keyword arguments.
        :type input_file: str, optional

        :param verbose: Verbosity level of the solver: 0 for almost no output,
            5 for the full output from **EDIpack**.
        :type verbose: int, default=3

        :param print_input_vars: Flag to toggle the printing of all the input
            variables and their values on the console.
        :type print_input_vars: bool, default=False

        :param keep_dir: Do not remove **EDIpack**'s temporary directory upon
                         destruction of this :py:class:`EDIpackSolver` object.
        :type keep_dir: bool, default=False

        :param ed_mode: Force a specific EDIpack exact diagonalization mode.
                        The requested mode still has to be compatible with the
                        provided Hamiltonian. Using this parameter
                        makes sense when :py:attr:`EDMode.NORMAL` is
                        automatically selected, but :py:attr:`EDMode.SUPERC`
                        or :py:attr:`EDMode.NONSU2` is desired instead.
        :type ed_mode: EDMode, optional

        :param zerotemp: Enable zero temperature calculations.
        :type zerotemp: bool, default=False

        :param cutoff: Spectrum cutoff used to determine the number of states
                       to be retained.
        :type cutoff: float, default=1e-9

        :param gs_threshold: Energy threshold for ground state degeneracy loop.
        :type gs_threshold: float, default=1e-9

        :param ed_sparse_h: Switch between storing the sparse matrix
            :math:`\hat H` (*True*) and direct on-the-fly evaluation of
            the product :math:`\hat H |v\rangle` (*False*).
        :type ed_sparse_h: bool, default=True

        :param ed_total_ud: Force use of the total spin-up and spin-down
            occupancies as quantum numbers instead of the orbital-resolved
            occupancies.
        :type ed_total_ud: bool, default=False

        :param lanc_method: Select the method to be used in
            the determination of the spectrum, one of *"arpack"*, *"lanczos"*
            (simple Lanczos method, only works at zero temperature, can be
            useful in some rare cases, when ARPACK fails) and *"dvdson"*
            (no-MPI mode only).
        :type lanc_method: str, default="arpack"

        :param lanc_nstates_sector: Initial number of states per sector
            to be determined.
        :type lanc_nstates_sector: int, default=2

        :param lanc_nstates_total: Initial total number of states
            to be determined.
        :type lanc_nstates_total: int, default=2

        :param lanc_nstates_step: Number of states added to the spectrum
            at each step.
        :type lanc_nstates_step: int, default=2

        :param lanc_ncv_factor: Set the size of the block used in
            Lanczos-ARPACK by multiplying the required Neigen
            (``NCV = lanc_ncv_factor * Neigen + lanc_ncv_add``).
        :type lanc_ncv_factor: int, default=10

        :param lanc_ncv_add: Adds up to the size of the block to prevent it
            from becoming too small
            (``NCV = lanc_ncv_factor * Neigen + lanc_ncv_add``).
        :type lanc_ncv_add: int, default=0

        :param lanc_niter: Number of Lanczos iterations in spectrum
            determination.
        :type lanc_niter: int, default=512

        :param lanc_ngfiter: Number of Lanczos iterations in GF determination
            (number of momenta).
        :type lanc_ngfiter: int, default=200

        :param lanc_tolerance: Tolerance for the Lanczos iterations as used
            in ARPACK.
        :type lanc_tolerance: float, default=1e-18

        :param lanc_dim_threshold: Minimal dimension threshold to use Lanczos
            determination of the spectrum rather than LAPACK-based
            exact diagonalization.
        :type lanc_dim_threshold: int, default=1024

        :param bath_fitting_params: Parameters used to perform bath fitting.
        :type bath_fitting_params: BathFittingParams, optional
        """

        assert self.instance_count[0] < 1, \
            "Only one instance of EDIpackSolver can exist at any time"

        assert len(fops_imp_up) > 0, "fops_imp_up must not be empty"
        assert len(fops_imp_dn) > 0, "fops_imp_dn must not be empty"
        validate_fops_up_dn(fops_imp_up, fops_imp_dn,
                            "fops_imp_up", "fops_imp_dn")
        validate_fops_up_dn(fops_bath_up, fops_bath_dn,
                            "fops_bath_up", "fops_bath_dn")

        self.norb = len(fops_imp_up)

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
            self.fops_bath_up, self.fops_bath_dn,
            kwargs.get("ed_mode", None)
        )
        self.nspin = self.h_params.Hloc.shape[0]

        # Keep temporary directory?
        keep_dir = kwargs.get("keep_dir", False)
        if keep_dir and sys.version_info < (3, 12, 0):
            keep_dir = False
            warn("keep_dir=True is ignored on Python versions prior to 3.12")

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
            c["PRINT_INPUT_VARS"] = kwargs.get("print_input_vars", False)
            c["CUTOFF"] = kwargs.get("cutoff", 1e-9)
            c["GS_THRESHOLD"] = kwargs.get("gs_threshold", 1e-9)
            c["ED_SPARSE_H"] = kwargs.get("ed_sparse_h", True)
            c["LANC_METHOD"] = kwargs.get("lanc_method", "arpack")
            c["LANC_NSTATES_STEP"] = kwargs.get("lanc_nstates_step", 2)
            c["LANC_NCV_FACTOR"] = kwargs.get("lanc_ncv_factor", 10)
            c["LANC_NCV_ADD"] = kwargs.get("lanc_ncv_add", 0)
            c["LANC_NITER"] = kwargs.get("lanc_niter", 512)
            c["LANC_NGFITER"] = kwargs.get("lanc_ngfiter", 200)
            c["LANC_TOLERANCE"] = kwargs.get("lanc_tolerance", 1e-18)
            c["LANC_DIM_THRESHOLD"] = kwargs.get("lanc_dim_threshold", 1024)

            # Impurity structure
            c["ED_MODE"] = self.h_params.ed_mode.value
            c["NSPIN"] = self.nspin
            c["NORB"] = self.norb

            # Bath geometry
            if self.h_params.bath is not None:
                c["BATH_TYPE"] = self.h_params.bath.name
                c["NBATH"] = self.h_params.bath.nbath
            else:
                c["NBATH"] = 0
                c["BATH_TYPE"] = "hybrid"

            # ed_total_ud
            ed_total_ud = kwargs.get("ed_total_ud", False)
            self.denden_int = _is_density_density(self.h_params.U)
            if not ((not ed_total_ud)
                    and self.h_params.ed_mode == EDMode.NORMAL
                    and isinstance(self.h_params.bath, (NoneType, BathNormal))
                    and _is_density(self.h_params.Hloc)
                    and self.denden_int):
                ed_total_ud = True
            c["ED_TOTAL_UD"] = ed_total_ud

            # Zero temperature calculations
            self.zerotemp = kwargs.get("zerotemp", False)
            c["ED_FINITE_TEMP"] = not self.zerotemp
            if self.zerotemp:
                c["LANC_NSTATES_TOTAL"] = 1
            else:
                c["LANC_NSTATES_TOTAL"] = kwargs.get("lanc_nstates_total", 2)
            c["LANC_NSTATES_SECTOR"] = kwargs.get("lanc_nstates_sector", 2)

            # Bath fitting
            bfp = kwargs.get("bath_fitting_params", BathFittingParams())
            c.update(bfp.__dict__())

            # Save G, G0 and \Sigma if the temporary directory is to be kept
            if keep_dir:
                c["PRINT_SECTOR_EIGENVALUES"] = True
                c["ED_PRINT_G"] = True
                c["ED_PRINT_G0"] = True
                c["ED_PRINT_SIGMA"] = True

            self.config = c

        self.comm = MPI.COMM_WORLD

        # A temporary directory for EDIpack is created and managed by the MPI
        # process with rank 0. The directory is assumed to be accessible to all
        # other MPI processes under the same name via a common file system.
        if self.comm.Get_rank() == 0:
            tdargs = {"prefix": "edipack-",
                      "suffix": ".tmp",
                      "dir": os.getcwd()}
            if keep_dir:
                tdargs["delete"] = False

            self.workdir = TemporaryDirectory(**tdargs)
            self.wdname = self.workdir.name
        else:
            self.wdname = None
        self.wdname = self.comm.bcast(self.wdname)

        with chdircontext(self.wdname):
            if self.comm.Get_rank() == 0:
                if self.input_file is None:
                    self.input_file = Path('input.conf').resolve()
                    with open(self.input_file, 'w') as config_file:
                        write_config(config_file, self.config)

                else:
                    Path('input.conf').symlink_to(self.input_file)

            self.comm.barrier()
            ed.read_input('input.conf')

            if self.comm.Get_rank() == 0:
                self.scifor_version = re.match(
                    r"^(.*): (.*)",
                    open("scifor_version.inc", 'r').readline())[2]

                self.edipack_version = re.match(
                    r"^(.*): (.*)",
                    open("EDIpack_version.inc", 'r').readline())[2]
            else:
                self.scifor_version = ""
                self.edipack_version = ""

            self.scifor_version = self.comm.bcast(self.scifor_version)
            self.edipack_version = self.comm.bcast(self.edipack_version)

        if isinstance(self.h_params.bath, BathGeneral):
            ed.set_hgeneral(self.h_params.bath.hvec,
                            self.h_params.bath.lambdavec)
        elif isinstance(self.h_params.bath, (BathNormal, BathHybrid)):
            assert self.h_params.bath.data.size == ed.get_bath_dimension()

        # Initialize EDIpack
        if self.h_params.bath is not None:
            ed.init_solver(
                bath=np.zeros(self.h_params.bath.data.size, dtype=float)
            )
        else:
            ed.init_solver(bath=np.array((), dtype=float))

        # GF block names
        if ed.get_ed_mode() in (int(EDMode.NORMAL), int(EDMode.SUPERC)):
            self.gf_block_names = (block_names_up[0], block_names_dn[0])
        else:
            self.gf_block_names = (block_names_up[0],)

        if ed.get_ed_mode() == int(EDMode.SUPERC):
            self.gf_an_block_names = (block_names_up[0]
                                      + "_" + block_names_dn[0],)

        self.instance_count[0] += 1
        self.comm.barrier()

    def __del__(self):
        self.comm.barrier()
        try:
            ed.finalize_solver()
            self.instance_count[0] -= 1
        # ed.finalize_solver() can fail if this __del__() method is called as
        # part of interpreter destruction procedure.
        except (AttributeError, TypeError):
            pass

    @property
    def hloc_mat(self) -> np.ndarray:
        r"""
        Access to the matrix of the local impurity Hamiltonian
        :math:`\hat H_\text{loc}`.
        """
        return self.h_params.Hloc

    @property
    def hloc_an_mat(self) -> np.ndarray:
        r"""
        Access to the matrix of the anomalous local impurity Hamiltonian
        :math:`\hat H_\text{loc, an}`.
        """
        return self.h_params.Hloc_an

    @property
    def hloc(self) -> op.Operator:
        r"""
        Access to the local impurity Hamiltonian :math:`\hat H_\text{loc}`
        in the form of a TRIQS many-body operator. The operator object contains
        both normal and anomalous contributions.
        """
        if self.h_params.ed_mode == EDMode.SUPERC:
            return self.h_params.Hloc_op(self.fops_imp_up, self.fops_imp_dn) + \
                self.h_params.Hloc_an_op(self.fops_imp_up, self.fops_imp_dn)
        else:
            return self.h_params.Hloc_op(self.fops_imp_up, self.fops_imp_dn)

    @hloc.setter
    def hloc(self, h: op.Operator):
        """
        Set the local impurity Hamiltonian given in the form of a TRIQS
        many-body operator containing both normal and anomalous contributions.
        """
        new_hloc, new_hloc_an = extract_quadratic(h,
                                                  self.fops_imp_up,
                                                  self.fops_imp_dn,
                                                  ignore_unexpected=False)
        if self.nspin == 1:
            if not is_spin_degenerate(new_hloc):
                raise RuntimeError(
                    "Local Hamiltonian must remain spin-degenerate"
                )
            self.h_params.Hloc[:] = new_hloc[:1, :1, :, :]
        else:
            self.h_params.Hloc[:] = new_hloc
        self.h_params.Hloc_an[0, 0, :, :] = new_hloc_an

    @property
    def U(self) -> np.ndarray:
        r"""
        Access to the two-particle interaction tensor
        :math:`U_{o_1 \sigma_1 o_2 \sigma_2 o_3 \sigma_3 o_4 \sigma_4}`.
        Contributions to the impurity interaction Hamiltonian are defined
        according to

        .. math::

            \hat H_{int} = \frac{1}{2}
                \sum_{\sigma_1 \sigma_2 \sigma_3 \sigma_4}\sum_{o_1 o_2 o_3 o_4}
                    U_{o_1 \sigma_1 o_2 \sigma_2 o_3 \sigma_3 o_4 \sigma_4}
                    c^\dagger_{\sigma_1 o_1} c^\dagger_{\sigma_2 o_2}
                    c_{\sigma_4 o_4} c_{\sigma_3 o_3}.

        """
        return self.h_params.U

    @property
    def bath(self) -> Union[Bath, NoneType]:
        r"""
        Access to the current :py:class:`bath <edipack2triqs.bath.Bath>` object
        stored in the solver. It is possible to assign a new bath object as long
        as it has the matching type and describes a bath of the same geometry.
        In the no-bath mode this attribute is set to :py:data:`None`.
        """
        return self.h_params.bath

    @bath.setter
    def bath(self, new_bath: Union[Bath, NoneType]):
        "Set the bath object"
        if self.h_params.bath is None:
            assert new_bath is None, \
                "Cannot set a new bath object in a no-bath calculation"
        else:
            self.h_params.bath.assert_compatible(new_bath)
            self.h_params.bath = new_bath

    def solve(self,
              beta: float = 1000,
              *,
              n_iw: int = 4096,
              energy_window: tuple[float, float] = (-5.0, 5.0),
              n_w: int = 5000,
              broadening: float = 0.01,
              n_tau: int = 1024,
              chi_spin: bool = False,
              chi_dens: bool = False,
              chi_exct: bool = False,
              chi_pair: bool = False):
        r"""
        Solve the impurity problem and calculate the observables, Green's
        function and self-energy.

        :param beta:
            Inverse temperature :math:`\beta = 1 / (k_B T)` for observable and
            Green's function calculations. In the zero temperature mode, this
            parameter determines spacing between fictitious Matsubara
            frequencies used as a mesh for the Green's functions.
        :type beta: float, default=1000

        :param n_iw: Number of Matsubara frequencies for impurity GF and
            response function calculations.
        :type n_iw: int, default=4096

        :param energy_window: Energy window for real-frequency impurity GF and
            response function calculations.
        :type energy_window: tuple[float, float], default=(-5.0, 5.0)

        :param n_w: Number of real-frequency points for impurity GF and
            response function calculations.
        :type n_w: int, default=5000

        :param broadening: Broadening (imaginary shift away from the
            real-frequency axis) for real-frequency impurity GF and
            response function calculations.
        :type broadening: float, default=0.01

        :param n_tau: Number of imaginary time points for calculations of
                      impurity response functions.
        :type n_tau: int, default=1024

        :param chi_spin: Compute spin response functions.
        :type chi_spin: bool, default=False

        :param chi_dens: Compute density response functions.
        :type chi_dens: bool, default=False

        :param chi_pair: Compute pairing response functions.
        :type chi_pair: bool, default=False

        :param chi_exct: Compute exciton response functions.
        :type chi_exct:  bool, default=False
        """

        ed.beta = beta
        ed.Lmats = n_iw
        ed.wini, ed.wfin = energy_window
        ed.Lreal = n_w
        ed.eps = broadening
        ed.Ltau = n_tau

        if (self.nspin == 2) and (self.h_params.ed_mode == EDMode.NORMAL) and \
                (not is_spin_diagonal(self.h_params.Hloc)):
            raise RuntimeError("Local Hamiltonian must remain spin-diagonal")

        # The interactions must remain of the density-density type, if this is
        # how they were at the construction time.
        if self.denden_int and (not _is_density_density(self.h_params.U)):
            raise RuntimeError(
                "Cannot add non-density-density terms to the interaction"
            )

        if (True in (chi_spin, chi_dens, chi_exct, chi_pair)) and \
                ed.get_ed_mode() != int(EDMode.NORMAL):
            raise RuntimeError(
                "Response functions are only available in the normal ED mode"
            )
        ed.chispin_flag = chi_spin
        ed.chidens_flag = chi_dens
        ed.chipair_flag = chi_pair
        ed.chiexct_flag = chi_exct

        self.comm.barrier()
        with chdircontext(self.wdname):
            # Set H_{loc} and H_{loc, an}
            if self.h_params.ed_mode == EDMode.SUPERC:
                ed.set_hloc(hloc=self.h_params.Hloc,
                            hloc_anomalous=self.h_params.Hloc_an)
            else:
                ed.set_hloc(hloc=self.h_params.Hloc)

            # Add interaction terms
            ed.reset_umatrix()
            for ind in np.ndindex(self.h_params.U.shape):
                val = self.h_params.U[ind]
                if val == 0:
                    continue
                o1, o2, o3, o4 = ind[0:8:2]
                s1, s2, s3, s4 = (('u' if s == 0 else 'd') for s in ind[1:8:2])
                ed.add_twobody_operator(o1, s1, o2, s2, o3, s3, o4, s4, val)

            # Solve!
            if self.h_params.bath is not None:
                ed.solve(self.h_params.bath.data)
            else:
                ed.solve(np.array((), dtype=float))
        self.comm.barrier()

    @property
    def e_pot(self) -> float:
        "Potential energy from interaction."
        return ed.get_eimp(ikind=0)

    @property
    def e_kin(self) -> float:
        "Kinetic energy."
        return ed.get_eimp(ikind=3)

    @property
    def densities(self) -> np.ndarray:
        "Impurity occupations, one element per orbital."
        return ed.get_dens()

    @property
    def double_occ(self) -> np.ndarray:
        "Impurity double occupancy, one element per orbital."
        return ed.get_docc()

    @property
    def superconductive_phi(self) -> np.ndarray:
        r"""
        Modulus of the impurity superconductive order parameter
        :math:`\phi = \langle c_{o_1,\uparrow} c_{o_2,\downarrow} \rangle`
        (matrix in the orbital space).
        """
        return ed.get_phi(component='mod')

    @property
    def superconductive_phi_arg(self) -> np.ndarray:
        r"""
        Complex argument of the impurity superconductive order parameter
        :math:`\phi = \langle c_{o_1,\uparrow} c_{o_2,\downarrow} \rangle`
        (matrix in the orbital space).
        """
        return ed.get_phi(component='arg')

    @property
    def magnetization(self) -> np.ndarray:
        """
        Cartesian components of impurity magnetization vectors,
        one row per orbital.
        """
        return ed.get_mag().T

    def _make_gf(self, ed_func, real_freq, anomalous) -> BlockGf:
        if anomalous:
            if ed.get_ed_mode() != int(EDMode.SUPERC):
                raise RuntimeError("anomalous = True is only supported for "
                                   "superconducting bath")

        if real_freq:
            mesh = MeshReFreq(window=(ed.wini, ed.wfin), n_w=ed.Lreal)
            z_vals = np.asarray([complex(z) + ed.eps * 1j for z in mesh])
        else:
            mesh = MeshImFreq(beta=ed.beta, S="Fermion", n_iw=ed.Lmats)
            z_vals = np.asarray([complex(z) for z in mesh])

        with chdircontext(self.wdname):
            data = ed_func(z_vals, typ='a' if anomalous else 'n')

        if anomalous:
            F = Gf(mesh=mesh, target_shape=(self.norb, self.norb))
            F.data[:] = np.rollaxis(data[0, 0, :, :, :], 2)
            return BlockGf(name_list=self.gf_an_block_names,
                           block_list=[F],
                           make_copies=False)

        if ed.get_ed_mode() in (int(EDMode.NORMAL), int(EDMode.SUPERC)):
            # 2 spin blocks
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
    # Green's functions
    #

    @property
    def g_iw(self) -> BlockGf:
        "Matsubara impurity Green's function."
        return self._make_gf(ed.build_gimp, False, False)

    @property
    def g_an_iw(self) -> BlockGf:
        "Anomalous Matsubara impurity Green's function."
        return self._make_gf(ed.build_gimp, False, True)

    @property
    def g_w(self) -> BlockGf:
        "Real-frequency impurity Green's function."
        return self._make_gf(ed.build_gimp, True, False)

    @property
    def g_an_w(self) -> BlockGf:
        "Anomalous real-frequency impurity Green's function."
        return self._make_gf(ed.build_gimp, True, True)

    #
    # Non-interacting Green's functions
    #

    @property
    def g0_iw(self) -> BlockGf:
        "Matsubara non-interacting impurity Green's function"
        def ed_func(z_vals, typ):
            return ed.get_g0and(z_vals, self.bath.data, typ=typ)
        return self._make_gf(ed_func, False, False)

    @property
    def g0_an_iw(self) -> BlockGf:
        "Anomalous Matsubara non-interacting impurity Green's function"
        def ed_func(z_vals, typ):
            return ed.get_g0and(z_vals, self.bath.data, typ=typ)
        return self._make_gf(ed_func, False, True)

    @property
    def g0_w(self) -> BlockGf:
        "Real-frequency non-interacting impurity Green's function"
        def ed_func(z_vals, typ):
            return ed.get_g0and(z_vals, self.bath.data, typ=typ)
        return self._make_gf(ed_func, True, False)

    @property
    def g0_an_w(self) -> BlockGf:
        "Anomalous real-frequency non-interacting impurity Green's function"
        def ed_func(z_vals, typ):
            return ed.get_g0and(z_vals, self.bath.data, typ=typ)
        return self._make_gf(ed_func, True, True)

    #
    # Self-energies
    #

    @property
    def Sigma_iw(self) -> BlockGf:
        "Matsubara impurity self-energy."
        return self._make_gf(ed.build_sigma, False, False)

    @property
    def Sigma_an_iw(self) -> BlockGf:
        "Anomalous Matsubara impurity self-energy."
        return self._make_gf(ed.build_sigma, False, True)

    @property
    def Sigma_w(self) -> BlockGf:
        "Real-frequency impurity self-energy."
        return self._make_gf(ed.build_sigma, True, False)

    @property
    def Sigma_an_w(self) -> BlockGf:
        "Anomalous real-frequency impurity self-energy."
        return self._make_gf(ed.build_sigma, True, True)

    #
    # Hybridization function
    #

    @property
    def Delta_iw(self) -> BlockGf:
        "Matsubara hybridization function"
        def ed_func(z_vals, typ):
            return ed.get_delta(z_vals, self.bath.data, typ=typ)
        return self._make_gf(ed_func, False, False)

    @property
    def Delta_an_iw(self) -> BlockGf:
        "Anomalous Matsubara hybridization function"
        def ed_func(z_vals, typ):
            return ed.get_delta(z_vals, self.bath.data, typ=typ)
        return self._make_gf(ed_func, False, True)

    @property
    def Delta_w(self) -> BlockGf:
        "Real-frequency hybridization function"
        def ed_func(z_vals, typ):
            return ed.get_delta(z_vals, self.bath.data, typ=typ)
        return self._make_gf(ed_func, True, False)

    @property
    def Delta_an_w(self) -> BlockGf:
        "Anomalous real-frequency hybridization function"
        def ed_func(z_vals, typ):
            return ed.get_delta(z_vals, self.bath.data, typ=typ)
        return self._make_gf(ed_func, True, True)

    #
    # Response functions
    #

    def _make_chi(self, chan, axis) -> Gf:
        if ed.get_ed_mode() != int(EDMode.NORMAL):
            raise RuntimeError(
                "Response functions are only available in the normal ED mode"
            )

        match axis:
            case "m":
                mesh = MeshImFreq(beta=ed.beta, S="Boson", n_iw=ed.Lmats)
                z_vals = np.asarray([complex(z) for z in mesh])
            case "r":
                mesh = MeshReFreq(window=(ed.wini, ed.wfin), n_w=ed.Lreal)
                z_vals = np.asarray([complex(z) + ed.eps * 1j for z in mesh])
            case "t":
                mesh = MeshImTime(beta=ed.beta, S="Boson", n_tau=ed.Ltau)
                z_vals = np.asarray([complex(z) for z in mesh])
            case _:
                raise AssertionError("Unexpected axis")

        target_shape = (3, self.norb, self.norb) if chan == "exct" else \
                       (self.norb, self.norb)

        with chdircontext(self.wdname):
            data = ed.get_chi(chan=chan, zeta=z_vals, axis=axis)

        chi = Gf(mesh=mesh, target_shape=target_shape)
        chi.data[:] = np.rollaxis(data, -1)

        return chi

    #
    # Bath fitting
    #

    chi2_fit_bath = _chi2_fit_bath


# Add response function properties to EDIpackSolver
def _make_chi_property(chan, chan_name, axis, axis_name):
    def getter(self) -> Gf:
        return self._make_chi(chan, axis)
    docstring = f"{axis_name} {chan_name} response function"
    return property(fget=getter, doc=docstring)


for chan, chan_name in [("spin", "spin"),
                        ("dens", "density"),
                        ("pair", "pairing"),
                        ("exct", "exciton")]:
    for axis, axis_name, axis_suffix in [("m", "Matsubara", "iw"),
                                         ("r", "Real-frequency", "w"),
                                         ("t", "Imaginary time", "tau")]:
        setattr(EDIpackSolver, f"chi_{chan}_{axis_suffix}",
                _make_chi_property(chan, chan_name, axis, axis_name))
