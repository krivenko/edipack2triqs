from enum import Enum


class EDMode(Enum):
    "EDIpack's :f:var:`exact diagonalization mode <f/ed_input_vars/ed_mode>`."
    NORMAL = "normal"
    """
    Normal
    """
    SUPERC = "superc"
    """
    s-wave superconductive
    """
    NONSU2 = "nonsu2"
    """
    Broken SU(2) symmetry
    """
