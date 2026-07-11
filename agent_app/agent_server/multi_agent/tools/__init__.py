"""
Tools for the multi-agent system.

This package contains UC function registration and utilities.
"""

from .tabular_ltm import (
    get_tabular_prediction_tool,
    run_tabular_prediction,
    warmup_tabular_ltm,
)


def register_uc_functions(*args, **kwargs):
    from .uc_functions import register_uc_functions as _register_uc_functions

    return _register_uc_functions(*args, **kwargs)


def check_uc_functions_exist(*args, **kwargs):
    from .uc_functions import check_uc_functions_exist as _check_uc_functions_exist

    return _check_uc_functions_exist(*args, **kwargs)

__all__ = [
    "register_uc_functions",
    "check_uc_functions_exist",
    "get_tabular_prediction_tool",
    "run_tabular_prediction",
    "warmup_tabular_ltm",
]
