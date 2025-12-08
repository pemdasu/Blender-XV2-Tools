import contextlib
import os
import struct
import tempfile

import bpy

from ...utils import read_cstring
from ..EMD.EMD import (
    EMD_TextureSamplerDef,
)


class EMBEntry:
    def __init__(self):
        self.index = 0
        self.name = ""
        self.data: bytes = b""


class EMBFile:
    def __init__(self):
        self.entries: list[EMBEntry] = []
        self.path = ""


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
        data = f.read()

    if struct.unpack_from("<I", data, 0)[0] != EMB_SIGNATURE:
        return None

    emb = EMBFile()
    emb.path = path

    total_entries = struct.unpack_from("<I", data, 12)[0]
    contents_offset = struct.unpack_from("<I", data, 24)[0]
    file_name_table_offset = struct.unpack_from("<I", data, 28)[0]

    offsets: list[int] = []
    sizes: list[int] = []
    for i in range(total_entries):
        entry_offset = contents_offset + i * 8
        data_offset = struct.unpack_from("<I", data, entry_offset)[0] + entry_offset
        data_size = struct.unpack_from("<I", data, entry_offset + 4)[0]
        offsets.append(data_offset)
        sizes.append(data_size)

    name_offsets: list[int] = []
    if file_name_table_offset != 0:
        for i in range(total_entries):
            name_offsets.append(struct.unpack_from("<I", data, file_name_table_offset + 4 * i)[0])

    for i in range(total_entries):
        entry = EMBEntry()
        entry.index = i
        entry.name = (
            read_cstring(data, name_offsets[i])
            if file_name_table_offset != 0 and i < len(name_offsets)
            else f"DATA{i:03d}.dds"
        )
        entry.data = data[offsets[i] : offsets[i] + sizes[i]]
        emb.entries.append(entry)

    return emb


def load_emb_image(
    entry: EMBEntry,
    emb_path: str,
    name_prefix: str = "",
    base_override: str | None = None,
) -> bpy.types.Image | None:
    image_base = base_override or (entry.name or f"EMB_{entry.index:03d}.dds")
    image_name = f"{name_prefix}{image_base}"
    existing = bpy.data.images.get(image_name)
    if existing:
        return existing

    temp_path = None
    try:
        base_name = (
            os.path.basename(base_override)
            if base_override
            else (os.path.basename(entry.name) if entry.name else f"DATA{entry.index:03d}.dds")
        )
        base_name = f"{name_prefix}{base_name}"
        temp_path = os.path.join(tempfile.gettempdir(), base_name)
        with open(temp_path, "wb") as tmp:
            tmp.write(entry.data)

        image = bpy.data.images.load(temp_path)
        image.pack()
        image.filepath = emb_path
        image["emb_source"] = emb_path
        image["emb_entry_index"] = entry.index
        image["emb_entry_name"] = entry.name
        with contextlib.suppress(Exception):
            image.colorspace_settings.name = "sRGB"
    except Exception as error:
        print("Failed to load EMB image:", entry.name, error)
        image = None
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

    return image


def _extract_dyt_lines(image: bpy.types.Image, base_name: str) -> dict[str, bpy.types.Image]:
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
    for idx, label in enumerate(labels):
        start_row = idx * line_height
        if start_row >= height:
            break
        buf = slice_rows(start_row, line_height)
        new_name = f"{base_name}_{label}"
        new_img = bpy.data.images.new(
            new_name, width=width, height=line_height, alpha=True, float_buffer=True
        )
        with contextlib.suppress(Exception):
            new_img.colorspace_settings.name = "sRGB"
        new_img.pixels = buf
        new_img.pack()
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
