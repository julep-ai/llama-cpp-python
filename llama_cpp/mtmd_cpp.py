from __future__ import annotations

import os
import ctypes
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_int32,
    c_uint8,
    c_uint32,
    c_float,
    c_size_t,
    c_void_p,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
)
import pathlib
from typing import (
    List,
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
)

import llama_cpp.llama_cpp as llama_cpp

from llama_cpp._ctypes_extensions import (
    load_shared_library,
    ctypes_function_for_shared_library,
)

if TYPE_CHECKING:
    from llama_cpp.llama_types import (
        llama_token,
        llama_pos,
    )
    from llama_cpp._ctypes_extensions import (
        CtypesArray,
        CtypesPointer,
    )

# Define input text structure
class mtmd_input_text(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("add_special", c_bool),
        ("parse_special", c_bool),
    ]

# Define context parameters structure
class mtmd_context_params(Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("print_timings", c_bool),
        ("n_threads", c_int),
        ("verbosity", c_int),
        ("image_marker", c_char_p),  # const char*
        ("media_marker", c_char_p),  # const char*
    ]

# Define input chunk type enum
mtmd_input_chunk_type = c_int
(
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
    MTMD_INPUT_CHUNK_TYPE_AUDIO,
) = (0, 1, 2)

# Define slice template enum
mtmd_slice_tmpl = c_int
(
    MTMD_SLICE_TMPL_NONE,
    MTMD_SLICE_TMPL_MINICPMV_2_5,
    MTMD_SLICE_TMPL_MINICPMV_2_6,
    MTMD_SLICE_TMPL_LLAMA4,
) = (0, 1, 2, 3)

# Define whisper filters structure
class whisper_filters(Structure):
    _fields_ = [
        ("n_mel", c_int),
    ]

# Define mtmd_context structure
class mtmd_context(Structure):
    _fields_ = [
        ("ctx_v", c_void_p),  # clip_ctx*
        ("ctx_a", c_void_p),  # clip_ctx*
        ("text_model", c_void_p),  # const llama_model*
        ("image_embd_v", POINTER(c_float)),  # std::vector<float>
        ("print_timings", c_bool),
        ("n_threads", c_int),
        ("media_marker", c_char_p),  # std::string
        ("n_embd_text", c_int),
        ("img_beg", c_char_p),  # std::string
        ("img_end", c_char_p),  # std::string
        ("aud_beg", c_char_p),  # std::string
        ("aud_end", c_char_p),  # std::string
        ("slice_tmpl", c_int),  # mtmd_slice_tmpl
        ("tok_ov_img_start", llama_cpp.llama_token),
        ("tok_ov_img_end", llama_cpp.llama_token),
        ("tok_slices_start", llama_cpp.llama_token),
        ("tok_slices_end", llama_cpp.llama_token),
        ("tok_sli_img_start", llama_cpp.llama_token),
        ("tok_sli_img_end", llama_cpp.llama_token),
        ("tok_sli_img_mid", llama_cpp.llama_token),
        ("tok_row_end", llama_cpp.llama_token),
        ("tok_row_end_trail", c_bool),
        ("ov_img_first", c_bool),
        ("use_mrope", c_bool),
        ("w_filters", whisper_filters),
    ]

# Define bitmap structure
class mtmd_bitmap(Structure):
    _fields_ = [
        ("nx", c_uint32),
        ("ny", c_uint32),
        ("data", POINTER(c_uint8)),  # Vector represented as pointer
        ("id", c_char_p),
        ("is_audio", c_bool),
    ]

# Define image tokens structure
class mtmd_image_tokens(Structure):
    _fields_ = [
        ("nx", c_uint32),
        ("ny", c_uint32),
        ("use_mrope_pos", c_bool),
        ("batch_f32", c_void_p),  # clip_image_f32_batch
        ("id", c_char_p),
    ]

# Define audio tokens structure
class mtmd_audio_tokens(Structure):
    _fields_ = [
        ("n_tokens", c_uint32),
        ("batch_f32", c_void_p),  # clip_image_f32_batch
        ("id", c_char_p),
    ]

# Define input chunk structure
class mtmd_input_chunk(Structure):
    _fields_ = [
        ("type", mtmd_input_chunk_type),
        ("tokens_text", POINTER(llama_cpp.llama_token)),  # Vector represented as pointer
        ("tokens_image", c_void_p),  # mtmd_image_tokens_ptr
        ("tokens_audio", c_void_p),  # mtmd_audio_tokens_ptr
    ]

# Define input chunks structure
class mtmd_input_chunks(Structure):
    _fields_ = [
        ("entries", POINTER(mtmd_input_chunk)),  # Vector represented as pointer
    ]

# Define context pointer type
mtmd_context_p = NewType("mtmd_context_p", int)
mtmd_context_p_ctypes = c_void_p

# Define bitmap pointer type
mtmd_bitmap_p = NewType("mtmd_bitmap_p", int)
mtmd_bitmap_p_ctypes = c_void_p

# Define input chunks pointer type
mtmd_input_chunks_p = NewType("mtmd_input_chunks_p", int)
mtmd_input_chunks_p_ctypes = c_void_p

# Define input chunk pointer type
mtmd_input_chunk_p = NewType("mtmd_input_chunk_p", int)
mtmd_input_chunk_p_ctypes = c_void_p

# Define image tokens pointer type
mtmd_image_tokens_p = NewType("mtmd_image_tokens_p", int)
mtmd_image_tokens_p_ctypes = c_void_p

# Define audio tokens pointer type
mtmd_audio_tokens_p = NewType("mtmd_audio_tokens_p", int)
mtmd_audio_tokens_p_ctypes = c_void_p

# Load the library
_libmtmd_base_name = "mtmd"
_libmtmd_override_path = os.environ.get("mtmd_CPP_LIB")
_libmtmd_base_path = (
    pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
    if _libmtmd_override_path is None
    else pathlib.Path()
)

_libmtmd = load_shared_library(_libmtmd_base_name, _libmtmd_base_path)
ctypes_function = ctypes_function_for_shared_library(_libmtmd)

# Add core functions
@ctypes_function(
    "mtmd_context_params_default",
    [],
    mtmd_context_params,
)
def mtmd_context_params_default() -> mtmd_context_params:
    ...

@ctypes_function(
    "mtmd_init_from_file",
    [c_char_p, llama_cpp.llama_model_p_ctypes, mtmd_context_params],
    mtmd_context_p_ctypes,
)
def mtmd_init_from_file(
    mmproj_fname: bytes,
    text_model: llama_cpp.llama_model_p,
    ctx_params: mtmd_context_params,
    /,
) -> Optional[mtmd_context_p]:
    ...

@ctypes_function(
    "mtmd_free",
    [mtmd_context_p_ctypes],
    None,
)
def mtmd_free(ctx: mtmd_context_p, /):
    ...

@ctypes_function(
    "mtmd_default_marker",
    [],
    c_char_p,
)
def mtmd_default_marker() -> bytes:
    ...

################################################
# mtmd.h
################################################

@ctypes_function(
    "mtmd_tokenize",
    [
        mtmd_context_p_ctypes,
        mtmd_input_chunks_p_ctypes,
        POINTER(mtmd_input_text),
        POINTER(mtmd_bitmap_p_ctypes),
        c_size_t,
    ],
    c_int,
)
def mtmd_tokenize(
    ctx: mtmd_context_p,
    output: mtmd_input_chunks_p,
    text: "CtypesPointer[mtmd_input_text]",
    bitmaps: "CtypesArray[mtmd_bitmap_p_ctypes]",
    n_bitmaps: Union[c_size_t, int],
    /,
) -> int:
    ...

@ctypes_function(
    "mtmd_encode_chunk",
    [mtmd_context_p_ctypes, mtmd_input_chunk_p_ctypes],
    c_int,
)
def mtmd_encode_chunk(ctx: mtmd_context_p, chunk: mtmd_input_chunk_p, /) -> int:
    ...

@ctypes_function("mtmd_get_output_embd", [mtmd_context_p_ctypes], POINTER(c_float))
def mtmd_get_output_embd(
    ctx: mtmd_context_p, /
) -> "CtypesPointer[c_float]":
    ...

@ctypes_function("mtmd_decode_use_non_causal", [mtmd_context_p_ctypes], c_bool)
def mtmd_decode_use_non_causal(ctx: mtmd_context_p, /) -> bool:
    ...

@ctypes_function("mtmd_decode_use_mrope", [mtmd_context_p_ctypes], c_bool)
def mtmd_decode_use_mrope(ctx: mtmd_context_p, /) -> bool:
    ...

@ctypes_function("mtmd_support_vision", [mtmd_context_p_ctypes], c_bool)
def mtmd_support_vision(ctx: mtmd_context_p, /) -> bool:
    ...

@ctypes_function("mtmd_support_audio", [mtmd_context_p_ctypes], c_bool)
def mtmd_support_audio(ctx: mtmd_context_p, /) -> bool:
    ...

@ctypes_function("mtmd_get_audio_bitrate", [mtmd_context_p_ctypes], c_int)
def mtmd_get_audio_bitrate(ctx: mtmd_context_p, /) -> int:
    ...

# mtmd_bitmap

@ctypes_function(
    "mtmd_bitmap_init",
    [c_uint32, c_uint32, POINTER(c_uint8)],
    mtmd_bitmap_p_ctypes,
)
def mtmd_bitmap_init(
    nx: Union[c_uint32, int],
    ny: Union[c_uint32, int],
    data: "CtypesArray[c_uint8]",
    /,
) -> Optional[mtmd_bitmap_p]:
    ...

@ctypes_function(
    "mtmd_bitmap_init_from_audio",
    [c_size_t, POINTER(c_float)],
    mtmd_bitmap_p_ctypes,
)
def mtmd_bitmap_init_from_audio(
    n_samples: Union[c_size_t, int],
    data: "CtypesArray[c_float]",
    /,
) -> Optional[mtmd_bitmap_p]:
    ...

@ctypes_function(
    "mtmd_bitmap_get_nx",
    [mtmd_bitmap_p_ctypes],
    c_uint32,
)
def mtmd_bitmap_get_nx(bitmap: mtmd_bitmap_p, /) -> int:
    ...

@ctypes_function(
    "mtmd_bitmap_get_ny",
    [mtmd_bitmap_p_ctypes],
    c_uint32,
)
def mtmd_bitmap_get_ny(bitmap: mtmd_bitmap_p, /) -> int:
    ...

@ctypes_function(
    "mtmd_bitmap_get_data",
    [mtmd_bitmap_p_ctypes],
    POINTER(c_uint8),
)
def mtmd_bitmap_get_data(
    bitmap: mtmd_bitmap_p, /
) -> "CtypesPointer[c_uint8]":
    ...

@ctypes_function(
    "mtmd_bitmap_get_n_bytes",
    [mtmd_bitmap_p_ctypes],
    c_size_t,
)
def mtmd_bitmap_get_n_bytes(bitmap: mtmd_bitmap_p, /) -> int:
    ...

@ctypes_function(
    "mtmd_bitmap_is_audio",
    [mtmd_bitmap_p_ctypes],
    c_bool,
)
def mtmd_bitmap_is_audio(bitmap: mtmd_bitmap_p, /) -> bool:
    ...

@ctypes_function(
    "mtmd_bitmap_get_id",
    [mtmd_bitmap_p_ctypes],
    c_char_p,
)
def mtmd_bitmap_get_id(bitmap: mtmd_bitmap_p, /) -> bytes:
    ...

@ctypes_function(
    "mtmd_bitmap_set_id",
    [mtmd_bitmap_p_ctypes, c_char_p],
    None,
)
def mtmd_bitmap_set_id(bitmap: mtmd_bitmap_p, id: bytes, /):
    ...

@ctypes_function(
    "mtmd_bitmap_free",
    [mtmd_bitmap_p_ctypes],
    None,
)
def mtmd_bitmap_free(bitmap: mtmd_bitmap_p, /):
    ...

# mtmd_input_chunks

@ctypes_function("mtmd_input_chunks_init", [], mtmd_input_chunks_p_ctypes)
def mtmd_input_chunks_init() -> Optional[mtmd_input_chunks_p]:
    ...

@ctypes_function("mtmd_input_chunks_size", [mtmd_input_chunks_p_ctypes], c_size_t)
def mtmd_input_chunks_size(chunks: mtmd_input_chunks_p, /) -> int:
    ...

@ctypes_function(
    "mtmd_input_chunks_get",
    [mtmd_input_chunks_p_ctypes, c_size_t],
    mtmd_input_chunk_p_ctypes,
)
def mtmd_input_chunks_get(
    chunks: mtmd_input_chunks_p, idx: Union[c_size_t, int], /
) -> Optional[mtmd_input_chunk_p]:
    ...

@ctypes_function("mtmd_input_chunks_free", [mtmd_input_chunks_p_ctypes], None)
def mtmd_input_chunks_free(chunks: mtmd_input_chunks_p, /):
    ...

# mtmd_input_chunk

@ctypes_function(
    "mtmd_input_chunk_get_type", [mtmd_input_chunk_p_ctypes], mtmd_input_chunk_type
)
def mtmd_input_chunk_get_type(chunk: mtmd_input_chunk_p, /) -> int:
    ...

@ctypes_function(
    "mtmd_input_chunk_get_tokens_text",
    [mtmd_input_chunk_p_ctypes, POINTER(c_size_t)],
    POINTER(llama_cpp.llama_token),
)
def mtmd_input_chunk_get_tokens_text(
    chunk: mtmd_input_chunk_p, n_tokens_output: "CtypesPointer[c_size_t]", /
) -> "CtypesPointer[llama_token]":
    ...

@ctypes_function(
    "mtmd_input_chunk_get_tokens_image",
    [mtmd_input_chunk_p_ctypes],
    mtmd_image_tokens_p_ctypes,
)
def mtmd_input_chunk_get_tokens_image(
    chunk: mtmd_input_chunk_p, /
) -> Optional[mtmd_image_tokens_p]:
    ...

@ctypes_function(
    "mtmd_input_chunk_get_n_tokens", [mtmd_input_chunk_p_ctypes], c_size_t
)
def mtmd_input_chunk_get_n_tokens(chunk: mtmd_input_chunk_p, /) -> int:
    ...

@ctypes_function(
    "mtmd_input_chunk_get_n_pos", [mtmd_input_chunk_p_ctypes], llama_cpp.llama_pos
)
def mtmd_input_chunk_get_n_pos(chunk: mtmd_input_chunk_p, /) -> "llama_pos":
    ...

@ctypes_function("mtmd_input_chunk_get_id", [mtmd_input_chunk_p_ctypes], c_char_p)
def mtmd_input_chunk_get_id(chunk: mtmd_input_chunk_p, /) -> bytes:
    ...

@ctypes_function(
    "mtmd_input_chunk_copy", [mtmd_input_chunk_p_ctypes], mtmd_input_chunk_p_ctypes
)
def mtmd_input_chunk_copy(
    chunk: mtmd_input_chunk_p, /
) -> Optional[mtmd_input_chunk_p]:
    ...

@ctypes_function("mtmd_input_chunk_free", [mtmd_input_chunk_p_ctypes], None)
def mtmd_input_chunk_free(chunk: mtmd_input_chunk_p, /):
    ...

# mtmd_image_tokens

@ctypes_function(
    "mtmd_image_tokens_get_n_tokens", [mtmd_image_tokens_p_ctypes], c_size_t
)
def mtmd_image_tokens_get_n_tokens(image_tokens: mtmd_image_tokens_p, /) -> int:
    ...

@ctypes_function("mtmd_image_tokens_get_nx", [mtmd_image_tokens_p_ctypes], c_size_t)
def mtmd_image_tokens_get_nx(image_tokens: mtmd_image_tokens_p, /) -> int:
    ...

@ctypes_function("mtmd_image_tokens_get_ny", [mtmd_image_tokens_p_ctypes], c_size_t)
def mtmd_image_tokens_get_ny(image_tokens: mtmd_image_tokens_p, /) -> int:
    ...

@ctypes_function("mtmd_image_tokens_get_id", [mtmd_image_tokens_p_ctypes], c_char_p)
def mtmd_image_tokens_get_id(image_tokens: mtmd_image_tokens_p, /) -> bytes:
    ...

@ctypes_function(
    "mtmd_image_tokens_get_n_pos", [mtmd_image_tokens_p_ctypes], llama_cpp.llama_pos
)
def mtmd_image_tokens_get_n_pos(
    image_tokens: mtmd_image_tokens_p, /
) -> "llama_pos":
    ...

# New helper functions for bitmap handling
@ctypes_function(
    "mtmd_helper_bitmap_init_from_file",
    [mtmd_context_p_ctypes, c_char_p],
    mtmd_bitmap_p_ctypes,
)
def mtmd_helper_bitmap_init_from_file(
    ctx: mtmd_context_p,
    fname: bytes,
    /,
) -> Optional[mtmd_bitmap_p]:
    ...

@ctypes_function(
    "mtmd_helper_eval_chunks",
    [mtmd_context_p_ctypes, llama_cpp.llama_context_p_ctypes, mtmd_input_chunks_p_ctypes, llama_cpp.llama_pos, c_int32, c_int, c_bool, POINTER(llama_cpp.llama_pos)],
    c_int,
)
def mtmd_helper_eval_chunks(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunks: mtmd_input_chunks_p,
    n_past: llama_cpp.llama_pos,
    seq_id: int,
    n_batch: int,
    logits_last: bool,
    n_past_out: "CtypesPointer[llama_cpp.llama_pos]",
    /,
) -> int:
    ...

# Audio token structure
class mtmd_audio_tokens(Structure):
    _fields_ = [
        ("n_tokens", c_uint32),
        ("batch_f32", c_void_p),  # clip_image_f32_batch
        ("id", c_char_p),
    ]

mtmd_audio_tokens_p = NewType("mtmd_audio_tokens_p", int)
mtmd_audio_tokens_p_ctypes = c_void_p

# Update mtmd_input_chunk to include audio tokens
class mtmd_input_chunk(Structure):
    _fields_ = [
        ("type", mtmd_input_chunk_type),
        ("tokens_text", POINTER(llama_cpp.llama_token)),
        ("tokens_image", mtmd_image_tokens_p_ctypes),
        ("tokens_audio", mtmd_audio_tokens_p_ctypes),
    ]

# Helper class for managing bitmaps
class BitmapManager:
    def __init__(self):
        self.entries: List[mtmd_bitmap_p] = []

    def c_ptr(self) -> "CtypesArray[mtmd_bitmap_p_ctypes]":
        arr_type = (mtmd_bitmap_p_ctypes * len(self.entries))
        return arr_type(*(entry for entry in self.entries))

    def clear(self):
        for bitmap in self.entries:
            mtmd_bitmap_free(bitmap)
        self.entries.clear()

    def add_from_memory(self, ctx: mtmd_context_p, data: bytes) -> bool:
        import numpy as np
        data_array = np.frombuffer(data, dtype=np.uint8)
        bitmap = mtmd_helper_bitmap_init_from_buf(ctx, data_array.ctypes.data_as(POINTER(c_uint8)), len(data))
        if bitmap is None:
            return False
        self.entries.append(bitmap)
        return True

    def __del__(self):
        self.clear()

# Helper class for managing input chunks
class InputChunksManager:
    def __init__(self, chunks: mtmd_input_chunks_p):
        self.ptr = chunks

    def __del__(self):
        if self.ptr:
            mtmd_input_chunks_free(self.ptr)

    def size(self) -> int:
        return mtmd_input_chunks_size(self.ptr)

    def get(self, idx: int) -> Optional[mtmd_input_chunk_p]:
        return mtmd_input_chunks_get(self.ptr, idx)

@ctypes_function(
    "mtmd_helper_get_n_tokens",
    [mtmd_input_chunks_p_ctypes],
    c_size_t,
)
def mtmd_helper_get_n_tokens(chunks: mtmd_input_chunks_p, /) -> int:
    ...

@ctypes_function(
    "mtmd_helper_get_n_pos",
    [mtmd_input_chunks_p_ctypes],
    llama_cpp.llama_pos,
)
def mtmd_helper_get_n_pos(chunks: mtmd_input_chunks_p, /) -> "llama_pos":
    ...

@ctypes_function(
    "mtmd_helper_bitmap_init_from_buf",
    [mtmd_context_p_ctypes, POINTER(c_uint8), c_size_t],
    mtmd_bitmap_p_ctypes,
)
def mtmd_helper_bitmap_init_from_buf(
    ctx: mtmd_context_p,
    buf: "CtypesArray[c_uint8]",
    len: Union[c_size_t, int],
    /,
) -> Optional[mtmd_bitmap_p]:
    ...

@ctypes_function(
    "mtmd_helper_decode_image_chunk",
    [mtmd_context_p_ctypes, llama_cpp.llama_context_p_ctypes, mtmd_input_chunk_p_ctypes, POINTER(c_float), llama_cpp.llama_pos, llama_cpp.llama_seq_id, c_int32, POINTER(llama_cpp.llama_pos)],
    c_int32,
)
def mtmd_helper_decode_image_chunk(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunk: mtmd_input_chunk_p,
    encoded_embd: "CtypesPointer[c_float]",
    n_past: llama_cpp.llama_pos,
    seq_id: llama_cpp.llama_seq_id,
    n_batch: int,
    new_n_past: "CtypesPointer[llama_cpp.llama_pos]",
    /,
) -> int:
    ...

@ctypes_function(
    "mtmd_helper_eval_chunk_single",
    [mtmd_context_p_ctypes, llama_cpp.llama_context_p_ctypes, mtmd_input_chunk_p_ctypes, llama_cpp.llama_pos, llama_cpp.llama_seq_id, c_int32, c_bool, POINTER(llama_cpp.llama_pos)],
    c_int32,
)
def mtmd_helper_eval_chunk_single(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunk: mtmd_input_chunk_p,
    n_past: llama_cpp.llama_pos,
    seq_id: llama_cpp.llama_seq_id,
    n_batch: int,
    logits_last: bool,
    new_n_past: "CtypesPointer[llama_cpp.llama_pos]",
    /,
) -> int:
    ...
