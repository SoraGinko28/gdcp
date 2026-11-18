"""Torch-free signature and unsupported-option contract (acceptance 12/14)."""

import inspect
import subprocess
import sys

import pytest

from gdcp import FSDP2Checkpointer
from tests.support.gdcp_api import (
    LOAD_PARAMETERS,
    LOAD_REJECTED,
    REQUIRED,
    SAVE_PARAMETERS,
    SAVE_REJECTED,
)


@pytest.mark.parametrize(
    "method,expected", [("save", SAVE_PARAMETERS), ("load", LOAD_PARAMETERS)]
)
def test_reference_signature_parity(method, expected):
    signature = inspect.signature(getattr(FSDP2Checkpointer, method))
    actual = [
        (
            name,
            REQUIRED
            if parameter.default is inspect.Parameter.empty
            else parameter.default,
        )
        for name, parameter in signature.parameters.items()
    ]
    assert actual == expected
    assert signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name, parameter in signature.parameters.items()
        if name != "kwargs"
    )


@pytest.mark.parametrize("name,value", SAVE_REJECTED.items())
def test_save_rejects_unsupported_parameter_by_name(tmp_path, name, value):
    with pytest.raises(NotImplementedError, match=name):
        FSDP2Checkpointer.save(tmp_path, {}, **{name: value})


@pytest.mark.parametrize("name,value", LOAD_REJECTED.items())
def test_load_rejects_unsupported_parameter_by_name(tmp_path, name, value):
    with pytest.raises(NotImplementedError, match=name):
        FSDP2Checkpointer.load(tmp_path, {}, **{name: value})


@pytest.mark.parametrize("method", ["save", "load"])
def test_unknown_keywords_are_rejected_by_name(tmp_path, method):
    with pytest.raises(NotImplementedError, match="unknown_option"):
        getattr(FSDP2Checkpointer, method)(tmp_path, {}, unknown_option=True)


def test_importing_gdcp_neutral_layers_does_not_import_torch():
    code = (
        "import sys; import gdcp, gdcp.backend, gdcp.core, gdcp.desc, gdcp.policy, gdcp.state; "
        "assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
