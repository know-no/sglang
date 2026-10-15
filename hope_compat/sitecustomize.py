"""Narrow compatibility shims for running current SGLang on the Hope image."""

from importlib import metadata

try:
    from transformers import configuration_utils

    if not hasattr(configuration_utils, "PreTrainedConfig"):
        configuration_utils.PreTrainedConfig = configuration_utils.PretrainedConfig
except Exception:
    # Keep Python startup usable; the normal import will surface the real error.
    pass


# H20 enables Programmatic Dependent Launch in current SGLang. Triton 3.4's
# language builtin cannot accept the newer compiler-injected ``_semantic``
# argument, so disable only this optional optimization on that exact old ABI.
try:
    if metadata.version("triton").startswith("3.4."):
        import sgl_kernel.utils as sgl_kernel_utils

        sgl_kernel_utils.is_arch_support_pdl = lambda *args, **kwargs: False
except Exception:
    # Keep Python startup usable; the normal import will surface the real error.
    pass
