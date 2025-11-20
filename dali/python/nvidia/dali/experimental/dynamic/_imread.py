# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
imread - A convenience function for reading and decoding images from file paths.
"""

import numpy as np
from typing import Union, List
from ._tensor import Tensor, tensor
from ._batch import Batch, batch


def _imread_impl(
    filepaths: Union[str, List[str], Tensor, Batch],
    device: str = "cpu",
    **kwargs
):
    """
    Reads and decodes an image (or batch of images) from file path(s).

    Parameters
    ----------
    filepaths : str, list of str, Tensor, or Batch
        Path(s) to image file(s) to read. Can be:
        
        - A single string filepath
        - A list of string filepaths (returns a Batch)
        - A Tensor containing encoded filepath bytes
        - A Batch containing encoded filepath bytes
    
    device : str, optional
        Device to decode the image on. Can be "cpu", or "mixed".
        Default is "cpu".
    
    <DECODERS_IMAGE_KWARGS_PLACEHOLDER>

    Returns
    -------
    Tensor or Batch
        Decoded image(s). Returns a Tensor for single images, or a Batch
        for multiple images.

    Note
    ----
    This function is currently implemented by combining :meth:`io.file.read` and
    :meth:`decoders.image`, providing a simple interface for loading images
    from disk. This may change in the future to provide a more efficient implementation.

    Examples
    --------
    Read a single image:

    >>> import nvidia.dali.experimental.dynamic as ndd
    >>> img = ndd.imread("/path/to/image.jpg")
    >>> img = img.evaluate()

    Read multiple images as a batch:

    >>> imgs = ndd.imread(["/path/to/img1.jpg", "/path/to/img2.jpg"])
    >>> imgs = imgs.evaluate()

    Decode on GPU with mixed device:

    >>> img = ndd.imread("/path/to/image.jpg", device="mixed")
    >>> img = img.evaluate()
    
    Decode to grayscale:

    >>> from nvidia.dali import types
    >>> img = ndd.imread("/path/to/image.jpg", output_type=types.GRAY)
    >>> img = img.evaluate()

    Note
    ----
    The filepath encoding is handled automatically when passing strings.
    If you already have encoded filepaths as Tensors (from :meth:`io.file.read`
    documentation format), you can pass them directly.
    """
    from . import io, decoders

    if device not in ["cpu", "mixed"]:
        raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'mixed'.")

    # Handle different input types
    if isinstance(filepaths, str):
        # Single filepath string
        filepath_bytes = np.frombuffer(filepaths.encode("utf-8"), dtype=np.uint8).copy()
        filepath_tensor = tensor(filepath_bytes)
    elif isinstance(filepaths, list):
        # List of filepath strings
        filepath_bytes_list = [
            np.frombuffer(fp.encode("utf-8"), dtype=np.uint8).copy() for fp in filepaths
        ]
        filepath_tensor = batch(filepath_bytes_list)
    elif isinstance(filepaths, (Tensor, Batch)):
        # Already a Tensor or Batch (assume it's encoded filepath bytes)
        filepath_tensor = filepaths
    else:
        raise TypeError(
            f"filepaths must be str, list of str, Tensor, or Batch, got {type(filepaths)}"
        )

    # Read file contents
    file_data = io.file.read(filepath_tensor)

    # Decode image
    decoded = decoders.image(file_data, device=device, **kwargs)

    return decoded


def _build_imread_with_expanded_signature():
    """
    Build an imread function with expanded signature including all kwargs from decoders.image.
    This improves IDE autocomplete support and makes the API more discoverable.
    """
    import nvidia.dali.backend as _b
    from nvidia.dali.ops import _docs
    import makefun

    schema = _b.GetSchema('decoders__Image')
    skip = {'bytes_per_sample_hint', 'preserve'}
    
    # Build the signature with all decoder kwargs
    signature_args = ['filepaths', '/', '*', 'device="cpu"']
    
    for arg in schema.GetArgumentNames():
        if arg in skip or schema.IsTensorArgument(arg):
            continue
        # All decoder kwargs are optional
        signature_args.append(f'{arg}=None')
    
    signature = f"imread({', '.join(signature_args)})"
    
    # Create wrapper function that calls the implementation
    def imread_wrapper(filepaths, /, *, device="cpu", **kwargs):
        return _imread_impl(filepaths, device=device, **kwargs)
    
    # Generate kwargs documentation
    args = [a for a in schema.GetArgumentNames() if a not in skip and not schema.IsTensorArgument(a)]
    kwargs_doc = _docs._get_kwargs(schema, api='dynamic', args=args)
    
    # Update docstring with kwargs documentation
    doc = _imread_impl.__doc__
    if kwargs_doc and doc:
        indented_kwargs_doc = kwargs_doc.replace('\n', '\n    ')
        doc = doc.replace(
            "<DECODERS_IMAGE_KWARGS_PLACEHOLDER>",
            indented_kwargs_doc
        )
    
    # Create the function with expanded signature
    imread_expanded = makefun.create_function(signature, imread_wrapper, doc=doc)
    imread_expanded.__module__ = __name__
    
    return imread_expanded


# Replace imread with the version that has expanded signature
try:
    imread = _build_imread_with_expanded_signature()
except Exception as e:
    print(f"WARNING: Failed to build expanded imread signature: {e}")
    # Fallback to the simple version
    imread = _imread_impl

