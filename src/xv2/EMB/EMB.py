import contextlib
import os
import struct
import tempfile

import bpy

from ...utils import read_cstring
from ..EMD.EMD import (
    EMD_TextureSamplerDef,
)

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


def emb_prefix_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if stem.lower().endswith(".dyt"):
        stem = stem[:-4]
    return f"{stem}_"


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
    name_prefix: str = "",
    base_override: str | None = None,
) -> bpy.types.Image | None:
    if not entry.data:
        return None

    sig_index = entry.data.find(b"DDS ")
    if sig_index == -1:
        return None

    dds_data = entry.data[sig_index:]

    # DDS sanity checks and patching to keep Blender happy.
    try:
        header_size = struct.unpack_from("<I", dds_data, 4)[0]
        if header_size != 124:
            return None
        fourcc = dds_data[84:88]
        allowed = {b"DXT1", b"DXT3", b"DXT5", b"BC1 ", b"BC2 ", b"BC3 ", b"BC4 ", b"BC5 ", b"ATI2"}
        if fourcc and fourcc not in allowed:
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
        return None

    image_base = base_override or (entry.name or f"EMB_{entry.index:03d}.dds")
    image_colorspace = "sRGB" if ".dyt" in image_base.lower() else "Non-Color"
    source_tag = os.path.splitext(os.path.basename(emb_path))[0] if emb_path else "emb"
    image_name = f"{name_prefix}{source_tag}_{image_base}"
    existing = bpy.data.images.get(image_name)
    if existing:
        return _force_image_colorspace(existing, image_colorspace)

    temp_path = None
    try:
        base_name = (
            os.path.basename(base_override)
            if base_override
            else (os.path.basename(entry.name) if entry.name else f"DATA{entry.index:03d}.dds")
        )
        base_name = f"{name_prefix}{base_name}"
        temp_path = os.path.join(os.path.dirname(emb_path) or tempfile.gettempdir(), base_name)
        with open(temp_path, "wb") as tmp:
            tmp.write(dds_data)

        image = None
        try:
            bpy.ops.image.open(
                filepath=temp_path,
                check_existing=False,
                directory=os.path.dirname(temp_path),
                files=[{"name": os.path.basename(temp_path)}],
            )
            image = bpy.data.images.get(os.path.basename(temp_path))
        except Exception:
            pass
        if image is None:
            image = bpy.data.images.load(temp_path, check_existing=False)

        image.name = image_name
        image.filepath = temp_path
        image["emb_source"] = emb_path
        image["emb_entry_index"] = entry.index
        image["emb_entry_name"] = entry.name
        image = _force_image_colorspace(image, image_colorspace)
        with contextlib.suppress(Exception):
            image.pack()
    except Exception as error:
        print("Failed to load EMB image:", entry.name, error)
        image = None
    finally:
        # Remove the temp file after packing into Blender
        with contextlib.suppress(Exception):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return image


def _extract_dyt_lines(
    image: bpy.types.Image, base_name: str, block_index: int = 0, material_name: str | None = None
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

    labels = ["primary", "rim", "spec", "secondary"]
    start_line = max(0, block_index) * 4
    for idx, label in enumerate(labels):
        start_row = (start_line + idx) * line_height
        if start_row >= height:
            break
        buf = slice_rows(start_row, line_height)
        mat_suffix = f"_{material_name}" if material_name else ""
        new_name = f"{base_name}{mat_suffix}_{block_index}_{label}"
        existing = bpy.data.images.get(new_name)
        if existing:
            try:
                valid_size = existing.size[0] == width and existing.size[1] == line_height
                has_pixels = (
                    bool(existing.has_data) and len(existing.pixels) >= width * line_height * 4
                )
            except Exception:
                valid_size = False
                has_pixels = False
            if valid_size and has_pixels:
                results[label] = existing
                continue
        new_img = bpy.data.images.new(
            new_name, width=width, height=line_height, alpha=True, float_buffer=True
        )
        new_img.pixels = buf
        new_img.pack()
        new_img = _force_image_colorspace(new_img, "sRGB")
        results[label] = new_img

    return results


def attach_emb_textures_to_material(
    mat: bpy.types.Material,
    sampler_defs: list[EMD_TextureSamplerDef],
    emb_main: EMBFile | None,
    emb_dyt: EMBFile | None,
):
    if not sampler_defs:
        return

    sources = {}
    if emb_main:
        sources["emb"] = emb_main.path
    if emb_dyt:
        sources["dyt_emb"] = emb_dyt.path
    if sources:
        mat["emb_sources"] = sources

    main_prefix = emb_prefix_from_path(emb_main.path) if emb_main else ""
    dyt_prefix = emb_prefix_from_path(emb_dyt.path) if emb_dyt else ""

    node_tree = mat.node_tree
    bsdf = None
    if node_tree and node_tree.nodes:
        bsdf = node_tree.nodes.get("Principled BSDF")

    first_main_tex_node = None
    created_nodes = []
    for sampler in sampler_defs:
        tex_index = int(sampler.texture_index)
        entry = None
        entry_dyt = None
        if emb_main and 0 <= tex_index < len(emb_main.entries):
            entry = emb_main.entries[tex_index]

        dyt_index = 0
        if emb_dyt and emb_dyt.entries:
            entry_dyt = emb_dyt.entries[min(dyt_index, len(emb_dyt.entries) - 1)]

        if entry and node_tree:
            image = load_emb_image(entry, emb_main.path, main_prefix)
            if image:
                node_name = f"EMB_Sampler_{tex_index}"
                tex_node = node_tree.nodes.get(node_name)
                if tex_node is None:
                    tex_node = node_tree.nodes.new("ShaderNodeTexImage")
                    tex_node.name = node_name
                    tex_node.label = f"Sampler {tex_index}"
                    tex_node.location = (-400, 300 - 220 * tex_index)
                tex_node.image = image
                created_nodes.append(tex_node)
                if first_main_tex_node is None:
                    first_main_tex_node = tex_node

        if entry_dyt and node_tree:
            base_name = entry_dyt.name or f"DATA{entry_dyt.index:03d}.dds"
            base_name = f"{os.path.splitext(base_name)[0]}.dyt.dds"
            image = load_emb_image(entry_dyt, emb_dyt.path, dyt_prefix, base_override=base_name)
            if image:
                line_images = _extract_dyt_lines(
                    image, f"{dyt_prefix}{os.path.splitext(base_name)[0]}"
                )
                if not line_images:
                    line_images = {"dyt": image}

                y_base = -200 - 220 * tex_index
                for i, (label, img_obj) in enumerate(line_images.items()):
                    node_name = f"EMB_DYT_{label}_{tex_index}"
                    tex_node = node_tree.nodes.get(node_name)
                    if tex_node is None:
                        tex_node = node_tree.nodes.new("ShaderNodeTexImage")
                        tex_node.name = node_name
                        tex_node.label = f"DYT {label} {tex_index}"
                        tex_node.location = (-400, y_base - 140 * i)
                    tex_node.image = img_obj
                    created_nodes.append(tex_node)

    if bsdf and created_nodes:
        tex_node = created_nodes[0]
        if not tex_node.outputs["Color"].links:
            node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    if node_tree and first_main_tex_node:
        output_node = None
        for node in node_tree.nodes:
            if node.type == "OUTPUT_MATERIAL":
                output_node = node
                break
        if output_node is None:
            output_node = node_tree.nodes.new("ShaderNodeOutputMaterial")
            output_node.location = (400, 0)

        threshold_node = node_tree.nodes.get("EMB_AlphaThreshold")
        if threshold_node is None:
            threshold_node = node_tree.nodes.new("ShaderNodeMath")
            threshold_node.name = "EMB_AlphaThreshold"
            threshold_node.label = "EMB AlphaThreshold"
            threshold_node.operation = "LESS_THAN"
            threshold_node.inputs[1].default_value = 0.5
            threshold_node.location = (
                first_main_tex_node.location.x + 200,
                first_main_tex_node.location.y,
            )

        diffuse_node = node_tree.nodes.get("EMB_Diffuse")
        if diffuse_node is None:
            diffuse_node = node_tree.nodes.new("ShaderNodeBsdfDiffuse")
            diffuse_node.name = "EMB_Diffuse"
            diffuse_node.label = "EMB Diffuse"
            diffuse_node.location = (threshold_node.location.x + 220, threshold_node.location.y)
            diffuse_node.inputs["Roughness"].default_value = 0.0

        tex_alpha_socket = first_main_tex_node.outputs.get("Alpha")
        tex_color_socket = first_main_tex_node.outputs.get("Color")
        fac_socket = threshold_node.inputs[0] if threshold_node.inputs else None

        if fac_socket and tex_alpha_socket:
            if not tex_alpha_socket.links or not any(
                link.to_node is threshold_node for link in tex_alpha_socket.links
            ):
                node_tree.links.new(tex_alpha_socket, fac_socket)
        elif (
            fac_socket
            and tex_color_socket
            and (
                not tex_color_socket.links
                or not any(link.to_node is threshold_node for link in tex_color_socket.links)
            )
        ):
            node_tree.links.new(tex_color_socket, fac_socket)

        if not threshold_node.outputs["Value"].links or not any(
            link.to_node is diffuse_node for link in threshold_node.outputs["Value"].links
        ):
            node_tree.links.new(threshold_node.outputs["Value"], diffuse_node.inputs["Color"])

        if output_node.inputs["Surface"].links:
            for link in list(output_node.inputs["Surface"].links):
                node_tree.links.remove(link)
        node_tree.links.new(diffuse_node.outputs["BSDF"], output_node.inputs["Surface"])


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
