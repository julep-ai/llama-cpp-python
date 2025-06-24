from __future__ import annotations

import os
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_uint8,
    c_float,
    c_size_t,
    c_void_p,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
)
import pathlib
from typing import (
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
    from llama_cpp._ctypes_extensions import (
        CtypesArray,
    )


# Specify the base name of the shared library to load
_libmtmd_base_name = "mtmd"
_libmtmd_override_path = os.environ.get("mtmd_CPP_LIB")
_libmtmd_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib" if _libmtmd_override_path is None else pathlib.Path()

# Load the library
_libmtmd = load_shared_library(_libmtmd_base_name, _libmtmd_base_path)

ctypes_function = ctypes_function_for_shared_library(_libmtmd)


################################################
# mtmd.h
################################################

# struct clip_ctx;
clip_ctx_p = NewType("clip_ctx_p", int)
clip_ctx_p_ctypes = c_void_p


# struct mtmd_image_embed {
#     float * embed;
#     int n_image_pos;
# };
class mtmd_image_embed(Structure):
    _fields_ = [
        ("embed", POINTER(c_float)),
        ("n_image_pos", c_int),
    ]


# /** sanity check for clip <-> mtmd embed size match */
# mtmd_API bool mtmd_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip);
@ctypes_function(
    "mtmd_validate_embed_size",
    [llama_cpp.llama_context_p_ctypes, clip_ctx_p_ctypes],
    c_bool,
)
def mtmd_validate_embed_size(
    ctx_llama: llama_cpp.llama_context_p, ctx_clip: clip_ctx_p, /
) -> bool:
    ...


# /** build an image embed from image file bytes */
# mtmd_API struct mtmd_image_embed * mtmd_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
@ctypes_function(
    "mtmd_image_embed_make_with_bytes",
    [clip_ctx_p_ctypes, c_int, POINTER(c_uint8), c_int],
    POINTER(mtmd_image_embed),
)
def mtmd_image_embed_make_with_bytes(
    ctx_clip: clip_ctx_p,
    n_threads: Union[c_int, int],
    image_bytes: CtypesArray[c_uint8],
    image_bytes_length: Union[c_int, int],
    /,
) -> "_Pointer[mtmd_image_embed]":
    ...


# /** build an image embed from a path to an image filename */
# mtmd_API struct mtmd_image_embed * mtmd_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);
@ctypes_function(
    "mtmd_image_embed_make_with_filename",
    [clip_ctx_p_ctypes, c_int, c_char_p],
    POINTER(mtmd_image_embed),
)
def mtmd_image_embed_make_with_filename(
    ctx_clip: clip_ctx_p, n_threads: Union[c_int, int], image_path: bytes, /
) -> "_Pointer[mtmd_image_embed]":
    ...


# mtmd_API void mtmd_image_embed_free(struct mtmd_image_embed * embed);
# /** free an embedding made with mtmd_image_embed_make_* */
@ctypes_function("mtmd_image_embed_free", [POINTER(mtmd_image_embed)], None)
def mtmd_image_embed_free(embed: "_Pointer[mtmd_image_embed]", /):
    ...


# /** write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
# mtmd_API bool mtmd_eval_image_embed(struct llama_context * ctx_llama, const struct mtmd_image_embed * embed, int n_batch, int * n_past);
@ctypes_function(
    "mtmd_eval_image_embed",
    [
        llama_cpp.llama_context_p_ctypes,
        POINTER(mtmd_image_embed),
        c_int,
        POINTER(c_int),
    ],
    c_bool,
)
def mtmd_eval_image_embed(
    ctx_llama: llama_cpp.llama_context_p,
    embed: "_Pointer[mtmd_image_embed]",
    n_batch: Union[c_int, int],
    n_past: "_Pointer[c_int]",
    /,
) -> bool:
    ...


################################################
# clip.h
################################################


# /** load mmproj model */
# CLIP_API struct clip_ctx * clip_model_load    (const char * fname, int verbosity);
@ctypes_function("clip_model_load", [c_char_p, c_int], clip_ctx_p_ctypes)
def clip_model_load(
    fname: bytes, verbosity: Union[c_int, int], /
) -> Optional[clip_ctx_p]:
    ...


# /** free mmproj model */
# CLIP_API void clip_free(struct clip_ctx * ctx);
@ctypes_function("clip_free", [clip_ctx_p_ctypes], None)
def clip_free(ctx: clip_ctx_p, /):
    ...


# CLIP_API struct clip_image_u8  * clip_image_u8_init ();
@ctypes_function("clip_image_u8_init", [], c_void_p)
def clip_image_u8_init() -> Optional[c_void_p]:
    ...


# CLIP_API void clip_image_u8_free (struct clip_image_u8  * img);
@ctypes_function("clip_image_u8_free", [c_void_p], None)
def clip_image_u8_free(img: c_void_p, /):
    ...


# CLIP_API struct clip_image_f32_batch * clip_image_f32_batch_init();
@ctypes_function("clip_image_f32_batch_init", [], c_void_p)
def clip_image_f32_batch_init() -> Optional[c_void_p]:
    ...


# CLIP_API void clip_image_f32_batch_free(struct clip_image_f32_batch * batch);
@ctypes_function("clip_image_f32_batch_free", [c_void_p], None)
def clip_image_f32_batch_free(batch: c_void_p, /):
    ...


# /** preprocess img and store the result in res_imgs, pad_to_square may be overridden to false depending on model configuration */
# CLIP_API bool clip_image_preprocess(struct clip_ctx * ctx, const struct clip_image_u8 * img, struct clip_image_f32_batch * res_imgs );
@ctypes_function(
    "clip_image_preprocess",
    [
        clip_ctx_p_ctypes,
        c_void_p,
        c_void_p,
    ],
    c_bool,
)
def clip_image_preprocess(
    ctx: clip_ctx_p,
    img: c_void_p,
    res_imgs: c_void_p,
    /,
) -> bool:
    ...


# CLIP_API bool clip_image_batch_encode(struct clip_ctx * ctx, int n_threads, const struct clip_image_f32_batch * imgs, float * vec);
@ctypes_function(
    "clip_image_batch_encode",
    [
        clip_ctx_p_ctypes,
        c_int,
        c_void_p,
        POINTER(c_float),
    ],
    c_bool,
)
def clip_image_batch_encode(
    ctx: clip_ctx_p,
    n_threads: c_int,
    imgs: c_void_p,
    vec: c_void_p,
    /,
) -> bool:
    ...


# /** interpret bytes as an image file with length bytes_length, and use the result to populate img */
# CLIP_API bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img);
@ctypes_function(
    "clip_image_load_from_bytes",
    [
        c_void_p,
        c_size_t,
        c_void_p,
    ],
    c_bool,
)
def clip_image_load_from_bytes(
    bytes: c_void_p,
    bytes_length: c_size_t,
    img: c_void_p,
    /,
) -> bool:
    ...
