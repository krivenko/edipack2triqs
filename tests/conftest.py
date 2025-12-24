import pytest

from .test_solver import TestSolver


# Add a pytest flag that forces reference data to be recomputed
# using pomerol2triqs and written into HDF5 archives.
def pytest_addoption(parser):
    parser.addoption(
        "--generate-ref-data",
        action="store_true",
        help="Generate reference data for unit tests using pomerol2triqs "
             "and write it into HDF5 archives.",
    )


@pytest.fixture(scope="session", autouse=True)
def generate_ref_data(request):
    TestSolver.generate_ref_data = request.config.getoption(
        "generate_ref_data", default=False
    )
