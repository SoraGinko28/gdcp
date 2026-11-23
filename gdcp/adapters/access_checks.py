"""Hash owned bytes through the same public accessor used for checkpoint I/O."""

import ctypes
from hashlib import sha256
from types import SimpleNamespace

from gdtensor.builder import Err, Ok


def owned_digest(component, key):
    digest = sha256()

    def consume(pointer, size):
        digest.update((ctypes.c_ubyte * size).from_address(pointer))
        return Ok(None)

    accessor = component.source_accessor or component.accessor
    assert component.data_request is not None
    for start, end in component.data_request.ownership[key]:
        result = accessor.read_to_buffer(
            SimpleNamespace(
                global_tensor_id=key, global_tensor_offset=start, size=end - start
            ),
            consume,
        )
        if type(result) is Err:
            raise ValueError(f"{key}: {result.value.msg}")
    # Checks must not leave the default accessor's consecutive-range staging
    # session open before the executor begins reading the component again.
    if component.source_accessor is not None:
        component.source_accessor.finish()
    return digest.digest()
