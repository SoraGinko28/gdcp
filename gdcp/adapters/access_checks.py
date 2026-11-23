"""Hash owned bytes through the same public accessor used for checkpoint I/O."""

import ctypes
from hashlib import sha256

from gdtensor.builder import Err, Ok
from gdtensor.internal.ranges import RangeFromComputeView

from .torch_access import TorchAccessor


def owned_digest(component, key):
    digest = sha256()

    def consume(pointer, size):
        digest.update((ctypes.c_ubyte * size).from_address(pointer))
        return Ok(None)

    accessor = component.source_accessor or component.accessor
    assert component.data_request is not None
    for start, end in component.data_request.ownership[key]:
        result = accessor.read_to_buffer(
            RangeFromComputeView(key, start, end - start), consume
        )
        if type(result) is Err:
            raise ValueError(f"{key}: {result.value.msg}")
    # Checks must not leave the default accessor's consecutive-range staging
    # session open before the executor begins reading the component again.
    if isinstance(accessor, TorchAccessor):
        accessor.finish()
    return digest.digest()
