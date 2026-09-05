"""Strict loading, merging, rendering, and safe publication for ``audio export``."""

from .core import export_documents
from .errors import (
    ExportError,
    IncompatibleResultsError,
    InvalidResultError,
    OutputExistsError,
    OutputWriteError,
    TimingRequiredError,
    UnsafeOutputError,
)
from .loading import load_result_document
from .merge import merge_documents
from .models import EXPORT_FORMATS, ExportProduct, LoadedResult, MergedTranscript
from .refusals import (
    export_input_invalid,
    export_inputs_incompatible,
    output_exists,
    output_is_canonical_input,
    output_path_invalid,
    output_required_for_force,
    timing_required_for_format,
)

__all__ = [
    "EXPORT_FORMATS",
    "ExportError",
    "ExportProduct",
    "IncompatibleResultsError",
    "InvalidResultError",
    "LoadedResult",
    "MergedTranscript",
    "OutputExistsError",
    "OutputWriteError",
    "TimingRequiredError",
    "UnsafeOutputError",
    "export_documents",
    "export_input_invalid",
    "export_inputs_incompatible",
    "load_result_document",
    "merge_documents",
    "output_exists",
    "output_is_canonical_input",
    "output_path_invalid",
    "output_required_for_force",
    "timing_required_for_format",
]
