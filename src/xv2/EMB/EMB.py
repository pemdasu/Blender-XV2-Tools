import contextlib
import hashlib
import os
import struct
import tempfile
from collections.abc import Callable

import bpy

from ...utils import read_cstring

DDSD_LINEARSIZE = 0x80000
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PIXELFORMAT = 0x1000


def _set_colorspace(image: bpy.types.Image, name: str) -> None:
    with contextlib.suppress(Exception):
        cs = image.colorspace_settings
        is_data = name in ("Non-Color", "Raw")
        cs.is_data = is_data
        for _ in range(3):
            cs.name = name
            if cs.name == name:
                break
            for alt in ("Linear", "Raw", "sRGB", "Non-Color"):
                cs.name = alt
        cs.is_data = is_data
        image.update()
        image.update_tag()


def _force_image_colorspace(image: bpy.types.Image, name: str) -> bpy.types.Image:
    _set_colorspace(image, name)
    try:
        if image.colorspace_settings.name == name:
            return image
    except Exception:
        pass
    # As a last resort, duplicate the image datablock and apply the colorspace on the copy.
    with contextlib.suppress(Exception):
        dup = image.copy()
        dup.name = f"{image.name}_cs"
        _set_colorspace(dup, name)
        return dup
    return image


class EMBEntry:
    def __init__(self):
        self.index = 0
        self.name = ""
        self.data: bytes = b""


class EMBFile:
    def __init__(self):
        self.entries: list[EMBEntry] = []
        self.path = ""
        # Extra header/meta fields (to match LB parser)
        self.is_emz = False
        self.i_08: int | None = None
        self.i_10: int | None = None
        self.use_file_names: bool | None = None


EMB_SIGNATURE = 1112360227


def _normalize_source_path(path: str) -> str:
    if not path:
        return ""
    with contextlib.suppress(Exception):
        return os.path.normcase(os.path.normpath(os.path.abspath(path)))
    return path


def _source_token(path: str) -> str:
    normalized = _normalize_source_path(path)
    if not normalized:
        return "nosrc"
    digest = hashlib.sha1(normalized.encode("utf-8", "replace")).hexdigest()
    return digest[:10]


def emb_stem_from_path(emb_path: str) -> str:
    base = os.path.basename(emb_path or "")
    stem = os.path.splitext(base)[0]
    if stem.lower().endswith(".dyt"):
        stem = stem[:-4]
    if stem.lower().endswith("_dyt"):
        stem = stem[:-4]
    if stem.lower().endswith("_000"):
        stem = stem[:-4]
    return stem


def _image_matches_emb_entry(image: bpy.types.Image, source_token: str, entry_index: int) -> bool:
    try:
        image_token = str(image.get("emb_source_token", ""))
        image_index = int(image.get("emb_entry_index", -1))
        return image_token == source_token and image_index == int(entry_index)
    except Exception:
        return False


def _build_image_name(
    emb_path: str,
    entry_index: int,
    image_base: str,
) -> str:
    source_name = emb_stem_from_path(emb_path)
    base_name = os.path.basename(image_base) or f"EMB_{int(entry_index):03d}.dds"
    tex_name = os.path.splitext(base_name)[0]
    source_prefix = f"{source_name}_"
    if source_name and tex_name.lower().startswith(source_prefix.lower()):
        return tex_name
    if source_name:
        return f"{source_name}_{tex_name}"
    return tex_name


def _create_image_name(clean_name: str, source_token: str, entry_index: int) -> str:
    image_name = clean_name
    suffix = 0
    token_short = (source_token or "dup")[:6]
    while True:
        existing = bpy.data.images.get(image_name)
        if not existing or _image_matches_emb_entry(existing, source_token, entry_index):
            return image_name
        suffix += 1
        if suffix == 1:
            image_name = f"{clean_name}_{token_short}"
        else:
            image_name = f"{clean_name}_{token_short}_{suffix}"


def read_emb(path: str) -> EMBFile | None:
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 32:
        return None

    sig_pos = raw.find(struct.pack("<I", EMB_SIGNATURE))
    if sig_pos == -1 or sig_pos + 32 > len(raw):
        return None

    base = sig_pos
    view = memoryview(raw)[base:]

    emb = EMBFile()
    emb.path = path

    # Header fields (to match LB parser)
    emb.i_08 = struct.unpack_from("<H", view, 8)[0]
    emb.i_10 = struct.unpack_from("<H", view, 10)[0]
    total_entries = struct.unpack_from("<I", view, 12)[0]
    contents_offset = struct.unpack_from("<I", view, 24)[0]
    file_name_table_offset = struct.unpack_from("<I", view, 28)[0]
    emb.use_file_names = file_name_table_offset != 0

    offsets: list[int] = []
    sizes: list[int] = []
    for i in range(total_entries):
        entry_offset = contents_offset + i * 8
        if entry_offset + 8 > len(view):
            break
        rel_offset = struct.unpack_from("<I", view, entry_offset)[0]
        data_offset = rel_offset + entry_offset
        data_size = struct.unpack_from("<I", view, entry_offset + 4)[0]
        offsets.append(data_offset)
        sizes.append(data_size)

    name_offsets: list[int] = []
    if file_name_table_offset != 0:
        for i in range(total_entries):
            noff = file_name_table_offset + 4 * i
            if noff + 4 <= len(view):
                name_offsets.append(struct.unpack_from("<I", view, noff)[0])

    for i in range(len(offsets)):
        entry = EMBEntry()
        entry.index = i
        if file_name_table_offset != 0 and i < len(name_offsets):
            entry.name = read_cstring(view, name_offsets[i])
        else:
            entry.name = f"DATA{i:03d}.dds"
        entry.data = view[offsets[i] : offsets[i] + sizes[i]].tobytes()
        emb.entries.append(entry)

    return emb


def load_emb_image(
    entry: EMBEntry,
    emb_path: str,
    base_override: str | None = None,
    warn: Callable[[str], None] | None = None,
) -> bpy.types.Image | None:
    def _warn(message: str) -> None:
        if not message:
            return
        with contextlib.suppress(Exception):
            if warn:
                warn(message)

    if not entry.data:
        return None

    emb_file = os.path.basename(emb_path)
    entry_label = entry.name or f"DATA{entry.index:03d}.dds"

    sig_index = entry.data.find(b"DDS ")
    if sig_index == -1:
        _warn(f"Texture '{entry_label}' in '{emb_file}' is not a DDS texture.")
        return None

    dds_data = entry.data[sig_index:]

    # DDS sanity checks and patching to keep Blender happy.
    try:
        header_size = struct.unpack_from("<I", dds_data, 4)[0]
        if header_size != 124:
            _warn(f"Texture '{entry_label}' in '{emb_file}' has an invalid DDS header and was skipped.")
            return None
        fourcc = dds_data[84:88]
        allowed = {b"DXT1", b"DXT3", b"DXT5", b"BC1 ", b"BC2 ", b"BC3 ", b"BC4 ", b"BC5 ", b"ATI2"}
        if fourcc and fourcc not in allowed:
            fourcc_text = fourcc.decode("ascii", errors="replace").strip() or repr(fourcc)
            _warn(
                f"Texture '{entry_label}' in '{emb_file}' uses unsupported DDS format '{fourcc_text}'. "
                "Supported: DXT1, DXT3, DXT5, BC1-BC5, ATI2."
            )
            return None
        flags = struct.unpack_from("<I", dds_data, 8)[0]
        height = struct.unpack_from("<I", dds_data, 12)[0]
        width = struct.unpack_from("<I", dds_data, 16)[0]
        is_bc1 = fourcc.strip() in (b"DXT1", b"BC1")
        block_size = 8 if is_bc1 else 16
        linearsize = struct.unpack_from("<I", dds_data, 20)[0]
        need_patch = False
        new_flags = flags
        new_linearsize = linearsize
        if width and height:
            calc_size = max(1, width // 4) * max(1, height // 4) * block_size
            if not (flags & DDSD_LINEARSIZE) or linearsize == 0:
                new_flags |= DDSD_LINEARSIZE
                new_linearsize = calc_size
                need_patch = True
            for req in (DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PIXELFORMAT):
                if not (new_flags & req):
                    new_flags |= req
                    need_patch = True
        if need_patch:
            mutable = bytearray(dds_data)
            struct.pack_into("<I", mutable, 8, new_flags)
            struct.pack_into("<I", mutable, 20, new_linearsize)
            dds_data = bytes(mutable)
    except Exception:
        _warn(f"Texture '{entry_label}' in '{emb_file}' could not be parsed as DDS and was skipped.")
        return None

    image_base = base_override or (entry.name or f"EMB_{entry.index:03d}.dds")
    image_colorspace = "sRGB" if ".dyt" in image_base.lower() else "Non-Color"
    normalized_source = _normalize_source_path(emb_path)
    source_token = _source_token(normalized_source)
    clean_name = _build_image_name(
        emb_path=emb_path,
        entry_index=entry.index,
        image_base=image_base,
    )

    for existing_img in bpy.data.images:
        if _image_matches_emb_entry(existing_img, source_token, entry.index):
            target_name = _create_image_name(clean_name, source_token, entry.index)
            if existing_img.name != target_name:
                with contextlib.suppress(Exception):
                    existing_img.name = target_name
            return _force_image_colorspace(existing_img, image_colorspace)

    image_name = _create_image_name(clean_name, source_token, entry.index)

    temp_path = None
    try:
        base_name = (
            os.path.basename(base_override)
            if base_override
            else (os.path.basename(entry.name) if entry.name else f"DATA{entry.index:03d}.dds")
        )
        base_stem, base_ext = os.path.splitext(base_name)
        if not base_ext:
            base_ext = ".dds"
        # Use a unique temp filename to prevent Blender from resolving a stale image datablock
        # when multiple EMBs contain DATA000-style entry names.
        temp_name = f"{source_token}_{entry.index:03d}_{base_stem}{base_ext}"
        temp_path = os.path.join(os.path.dirname(emb_path) or tempfile.gettempdir(), temp_name)
        with open(temp_path, "wb") as tmp:
            tmp.write(dds_data)

        image = bpy.data.images.load(temp_path, check_existing=False)

        image.name = image_name
        image.filepath = temp_path
        image["emb_source"] = emb_path
        image["emb_source_norm"] = normalized_source
        image["emb_source_token"] = source_token
        image["emb_entry_index"] = entry.index
        image["emb_entry_name"] = entry.name
        image = _force_image_colorspace(image, image_colorspace)
        with contextlib.suppress(Exception):
            image.pack()
    except Exception as error:
        print("Failed to load EMB image:", entry.name, error)
        _warn(f"Texture '{entry_label}' in '{emb_file}' failed to load in Blender and was skipped.")
        image = None
    finally:
        # Remove the temp file after packing into Blender
        with contextlib.suppress(Exception):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return image


def _extract_dyt_lines(
    image: bpy.types.Image,
    base_name: str,
    block_index: int = 0,
    source_token: str = "",
) -> dict[str, bpy.types.Image]:
    results: dict[str, bpy.types.Image] = {}
    width, height = image.size
    if height <= 0 or width <= 0:
        return results

    line_height = max(1, height // 32)
    total_needed = line_height * 4
    if total_needed > height:
        line_height = height // 4

    src_pixels = list(image.pixels)
    row_stride = width * 4

    def slice_rows(src_start_row: int, rows: int) -> list[float]:
        buf: list[float] = [0.0] * (width * rows * 4)
        for r in range(rows):
            # Blender pixel data is bottom-up, so invert to grab from the top
            src_row = max(0, min(height - 1, height - 1 - (src_start_row + r)))
            src_off = src_row * row_stride
            dst_off = r * row_stride
            buf[dst_off : dst_off + row_stride] = src_pixels[src_off : src_off + row_stride]
        return buf

    labels = ["p", "r", "s", "d"]
    start_line = max(0, block_index) * 4
    def _create_dyt_name(preferred_name: str) -> str:
        candidate = preferred_name
        token_short = (source_token or "dup")[:6]
        attempt = 0
        while bpy.data.images.get(candidate):
            attempt += 1
            if attempt == 1:
                candidate = f"{preferred_name}_{token_short}"
            else:
                candidate = f"{preferred_name}_{token_short}_{attempt}"
        return candidate

    for idx, label in enumerate(labels):
        start_row = (start_line + idx) * line_height
        if start_row >= height:
            break
        buf = slice_rows(start_row, line_height)
        new_name = f"{base_name}_{block_index}_{label}"
        existing = bpy.data.images.get(new_name)
        if existing:
            try:
                valid_size = existing.size[0] == width and existing.size[1] == line_height
                has_pixels = (
                    bool(existing.has_data) and len(existing.pixels) >= width * line_height * 4
                )
                same_source = (
                    not source_token or str(existing.get("emb_source_token", "")) == source_token
                )
            except Exception:
                valid_size = False
                has_pixels = False
                same_source = False
            if valid_size and has_pixels and same_source:
                results[label] = existing
                continue
        new_img = bpy.data.images.new(
            _create_dyt_name(new_name),
            width=width,
            height=line_height,
            alpha=True,
            float_buffer=True,
        )
        new_img.pixels = buf
        if source_token:
            new_img["emb_source_token"] = source_token
        new_img.pack()
        new_img = _force_image_colorspace(new_img, "sRGB")
        results[label] = new_img

    return results


def locate_emb_files(path: str) -> tuple[EMBFile | None, EMBFile | None]:
    folder = os.path.dirname(path)
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    parts = stem.split("_")
    char_code = parts[0] if parts else stem

    candidates = [
        (False, os.path.join(folder, f"{stem}.emb")),
        (True, os.path.join(folder, f"{stem}_dyt.emb")),
        (True, os.path.join(folder, f"{stem}.dyt.emb")),
        (False, os.path.join(folder, f"{char_code}.emb")),
        (True, os.path.join(folder, f"{char_code}_dyt.emb")),
        (True, os.path.join(folder, f"{char_code}.dyt.emb")),
        (False, os.path.join(folder, f"{char_code}_000.emb")),
        (True, os.path.join(folder, f"{char_code}_000_dyt.emb")),
        (True, os.path.join(folder, f"{char_code}_000.dyt.emb")),
    ]

    main_emb = None
    dyt_emb = None
    for is_dyt, candidate in candidates:
        if not os.path.exists(candidate):
            continue
        emb_file = read_emb(candidate)
        if emb_file is None:
            continue
        if is_dyt:
            if dyt_emb is None:
                dyt_emb = emb_file
        else:
            if main_emb is None:
                main_emb = emb_file

    return main_emb, dyt_emb
