import contextlib
import os
import struct

import bpy

from ...utils import float_to_half, remove_unused_vertex_groups
from .EMD import (
    EMD_SIGNATURE,
    VERTEX_BLENDWEIGHT,
    VERTEX_COLOR,
    VERTEX_COMPRESSED,
    VERTEX_NORMAL,
    VERTEX_POSITION,
    VERTEX_TANGENT,
    VERTEX_TEX2UV,
    VERTEX_TEXUV,
    EMD_File,
    EMD_Mesh,
    EMD_Model,
    EMD_Submesh,
    EMD_TextureSamplerDef,
    EMD_Triangles,
    EMD_Vertex,
    get_vertex_size_from_flags,
)


def _pad_data(data: bytearray, alignment: int) -> None:
    pad_len = (-len(data)) % alignment
    if pad_len:
        data.extend(b"\0" * pad_len)


def _create_palette_index(
    bone_name: str,
    palette_map: dict[str, int],
    palette_list: list[str],
) -> int:
    if bone_name in palette_map:
        return palette_map[bone_name]
    index = len(palette_list)
    palette_map[bone_name] = index
    palette_list.append(bone_name)
    return index


MAX_BONES_PER_TRIANGLE_GROUP = 24


def _get_or_create_triangle_group(
    required_bones: list[str],
    triangle_groups: list[EMD_Triangles],
) -> EMD_Triangles:
    for tri_group in triangle_groups:
        if tri_group.bone_palette_lookup is None:
            tri_group.bone_palette_lookup = {
                name: idx for idx, name in enumerate(tri_group.bone_names)
            }
        palette_map = tri_group.bone_palette_lookup
        current_count = len(palette_map)
        new_bones = [name for name in required_bones if name not in palette_map]
        if current_count + len(new_bones) <= MAX_BONES_PER_TRIANGLE_GROUP:
            for name in new_bones:
                _create_palette_index(name, palette_map, tri_group.bone_names)
            return tri_group

    tri_group = EMD_Triangles()
    tri_group.indices = []
    tri_group.bone_names = []
    tri_group.bone_palette_lookup = {}

    for name in required_bones[:MAX_BONES_PER_TRIANGLE_GROUP]:
        _create_palette_index(name, tri_group.bone_palette_lookup, tri_group.bone_names)

    triangle_groups.append(tri_group)
    return tri_group


def _collect_vertex_data_for_material(
    obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    material_index: int | None,
) -> tuple[list[EMD_Vertex], list[EMD_Triangles]]:
    mesh.calc_loop_triangles()
    with contextlib.suppress(Exception):
        mesh.calc_normals_split()

    # Prefer the active render UV for the primary channel; fallback to the first slot.
    uv_layer = mesh.uv_layers.active or (mesh.uv_layers[0] if mesh.uv_layers else None)
    uv2_layer = None
    if len(mesh.uv_layers) > 1:
        # Use second slot explicitly; if only one map exists, UV2 remains unset.
        uv2_layer = mesh.uv_layers[1]
    color_layer = mesh.color_attributes[0] if mesh.color_attributes else None

    vertices: list[EMD_Vertex] = []
    triangle_groups: list[EMD_Triangles] = []
    vertex_lookup: dict[tuple, int] = {}

    def gather_influences(vert: bpy.types.MeshVertex) -> list[tuple[str, float]]:
        influences: list[tuple[str, float]] = []
        for group_element in vert.groups:
            if group_element.group >= len(obj.vertex_groups):
                continue
            vertex_group = obj.vertex_groups[group_element.group]
            bone_name = vertex_group.name
            if arm_obj and arm_obj.data and bone_name not in arm_obj.data.bones:
                continue
            influences.append((bone_name, group_element.weight))

        influences.sort(key=lambda item: item[1], reverse=True)
        influences = influences[:4]

        total_weight = sum(weight_value for _bone, weight_value in influences)
        if total_weight > 1e-6:
            influences = [(bone, weight_value / total_weight) for bone, weight_value in influences]

        while len(influences) < 4:
            influences.append(("", 0.0))

        return influences

    for tri in mesh.loop_triangles:
        if material_index is not None and tri.material_index != material_index:
            continue

        tri_vertices: list[tuple[EMD_Vertex, list[tuple[str, float]]]] = []
        tri_bones_ordered: list[str] = []
        loop_indices = tri.loops if hasattr(tri, "loops") else tri.loop_indices

        for loop_idx in loop_indices:
            loop = mesh.loops[loop_idx]
            v_idx = loop.vertex_index
            vert = mesh.vertices[v_idx]
            vtx = EMD_Vertex()
            vtx.pos = tuple(vert.co)
            # Preserve sharp edges by using the per-loop split normal when available.
            vtx.normal = (
                tuple(loop.normal)
                if hasattr(loop, "normal") and loop.normal.length > 0
                else tuple(vert.normal)
            )

            if uv_layer:
                uv_val = uv_layer.data[loop_idx].uv
                vtx.uv = (uv_val.x, 1.0 - uv_val.y)
            if uv2_layer:
                uv_val = uv2_layer.data[loop_idx].uv
                vtx.uv2 = (uv_val.x, 1.0 - uv_val.y)
            if uv_layer and hasattr(loop, "tangent"):
                pass
            if color_layer:
                col = color_layer.data[loop_idx].color
                vtx.color = (
                    col[0],
                    col[1],
                    col[2],
                    col[3] if len(col) > 3 else 1.0,
                )

            influences = gather_influences(vert)
            tri_vertices.append((vtx, influences))

            for bone_name, weight_value in influences:
                if weight_value <= 0.0 or not bone_name:
                    continue
                if bone_name not in tri_bones_ordered:
                    tri_bones_ordered.append(bone_name)

        tri_group = _get_or_create_triangle_group(tri_bones_ordered, triangle_groups)
        if tri_group.bone_palette_lookup is None:
            tri_group.bone_palette_lookup = {
                name: idx for idx, name in enumerate(tri_group.bone_names)
            }
        palette_map = tri_group.bone_palette_lookup

        for vtx, influences in tri_vertices:
            bone_ids: list[int] = []
            bone_weights: list[float] = []

            for bone_name, weight_value in influences:
                bone_weights.append(float(weight_value))
                if bone_name:
                    palette_index = _create_palette_index(
                        bone_name,
                        palette_map,
                        tri_group.bone_names,
                    )
                else:
                    palette_index = 0
                bone_ids.append(int(palette_index))

            if sum(bone_weights) <= 1e-6:
                bone_ids = [bone_ids[0] if bone_ids else 0, 0, 0, 0]
                bone_weights = [0.0, 0.0, 0.0, 1.0]
            else:
                bone_ids = list(reversed(bone_ids))
                bone_weights = list(reversed(bone_weights))

            vtx.bone_ids = bone_ids
            vtx.bone_weights = bone_weights

            key = (
                tuple(vtx.pos),
                tuple(vtx.normal),
                tuple(vtx.uv),
                tuple(vtx.uv2),
                tuple(vtx.bone_ids),
                tuple(vtx.bone_weights),
            )
            if key in vertex_lookup:
                new_index = vertex_lookup[key]
            else:
                new_index = len(vertices)
                vertices.append(vtx)
                vertex_lookup[key] = new_index
            tri_group.indices.append(new_index)

    return vertices, triangle_groups


def _samplers_from_container(container) -> list[EMD_TextureSamplerDef]:
    samplers: list[EMD_TextureSamplerDef] = []
    if not hasattr(container, "emd_texture_samplers"):
        return samplers
    for item in container.emd_texture_samplers:
        sampler = EMD_TextureSamplerDef()
        sampler.flag0 = int(item.flag0)
        sampler.texture_index = int(item.texture_index)
        sampler.address_mode_u = int(item.address_mode_u)
        sampler.address_mode_v = int(item.address_mode_v)
        sampler.filtering_min = int(item.filtering_min)
        sampler.filtering_mag = int(item.filtering_mag)
        sampler.scale_u = float(item.scale_u)
        sampler.scale_v = float(item.scale_v)
        samplers.append(sampler)
    return samplers


def _default_texture_samplers() -> list[EMD_TextureSamplerDef]:
    samplers: list[EMD_TextureSamplerDef] = []
    for idx in range(2):
        sampler = EMD_TextureSamplerDef()
        sampler.texture_index = idx
        sampler.address_mode_u = 0
        sampler.address_mode_v = 0
        sampler.filtering_min = 2  # Linear
        sampler.filtering_mag = 2  # Linear
        sampler.scale_u = 1.0
        sampler.scale_v = 1.0
        samplers.append(sampler)
    return samplers


def _build_submeshes_from_object(
    obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
    mesh_data: bpy.types.Mesh | None = None,
) -> list[EMD_Submesh]:
    mesh = mesh_data or obj.data
    submeshes: list[EMD_Submesh] = []

    material_indices = list(range(len(mesh.materials))) if mesh.materials else [None]

    for mat_index in material_indices:
        mat = (
            mesh.materials[mat_index]
            if mat_index is not None and mat_index < len(mesh.materials)
            else None
        )

        vertices, triangle_groups = _collect_vertex_data_for_material(obj, arm_obj, mesh, mat_index)
        if not triangle_groups:
            continue

        sub = EMD_Submesh()
        sub.name = mat.name if mat else obj.name
        sub.vertices = vertices

        faces: list[tuple[int, int, int]] = []
        for tri_group in triangle_groups:
            for i in range(0, len(tri_group.indices), 3):
                faces.append(
                    (
                        tri_group.indices[i],
                        tri_group.indices[i + 1],
                        tri_group.indices[i + 2],
                    )
                )
        sub.faces = faces
        sub.triangle_groups = triangle_groups

        mat = (
            mesh.materials[mat_index]
            if mat_index is not None and mat_index < len(mesh.materials)
            else None
        )

        flags = 0
        if mat and "emd_vertex_flags" in mat:
            flags = int(mat["emd_vertex_flags"])
        elif obj and "emd_vertex_flags" in obj:
            flags = int(obj["emd_vertex_flags"])
        if flags == 0:
            flags = VERTEX_POSITION
            if vertices and any(v.normal != (0.0, 0.0, 0.0) for v in vertices):
                flags |= VERTEX_NORMAL
            if vertices and any(v.uv != (0.0, 0.0) for v in vertices):
                flags |= VERTEX_TEXUV
            if vertices and any(v.uv2 != (0.0, 0.0) for v in vertices):
                flags |= VERTEX_TEX2UV
            if vertices and any(v.tangent != (0.0, 0.0, 0.0) for v in vertices):
                flags |= VERTEX_TANGENT
            if vertices and any(v.color != (1.0, 1.0, 1.0, 1.0) for v in vertices):
                flags |= VERTEX_COLOR
            if arm_obj and vertices:
                flags |= VERTEX_BLENDWEIGHT

        sub.vertex_flags = flags

        if sub.vertices:
            xs = [v.pos[0] for v in sub.vertices]
            ys = [v.pos[1] for v in sub.vertices]
            zs = [v.pos[2] for v in sub.vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            min_z, max_z = min(zs), max(zs)
            size_x = max_x - min_x
            size_y = max_y - min_y
            size_z = max_z - min_z
            center_x = (max_x + min_x) / 2.0
            center_y = (max_y + min_y) / 2.0
            center_z = (max_z + min_z) / 2.0
            sub.aabb_center = (center_x, center_y, center_z, size_x)
            sub.aabb_min = (min_x, min_y, min_z, size_y)
            sub.aabb_max = (max_x, max_y, max_z, size_z)

        if mat and hasattr(mat, "emd_texture_samplers") and mat.emd_texture_samplers:
            sub.texture_sampler_defs = _samplers_from_container(mat)
        elif hasattr(obj, "emd_texture_samplers") and obj.emd_texture_samplers:
            sub.texture_sampler_defs = _samplers_from_container(obj)
        else:
            sub.texture_sampler_defs = []

        if not sub.texture_sampler_defs:
            sub.texture_sampler_defs = _default_texture_samplers()

        submeshes.append(sub)

    return submeshes


def _build_emd_from_object(
    obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
) -> EMD_File:
    emd = EMD_File()
    emd.version = int(obj.get("emd_file_version", 0x201))

    model = EMD_Model()
    model.name = obj.name
    mesh = EMD_Mesh()
    mesh.name = obj.name
    mesh.submeshes.extend(_build_submeshes_from_object(obj, arm_obj))
    model.meshes.append(mesh)
    emd.models.append(model)

    all_positions = [v.pos for sub in mesh.submeshes for v in sub.vertices]
    if all_positions:
        xs = [p[0] for p in all_positions]
        ys = [p[1] for p in all_positions]
        zs = [p[2] for p in all_positions]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        center_x = (max_x + min_x) / 2.0
        center_y = (max_y + min_y) / 2.0
        center_z = (max_z + min_z) / 2.0
        mesh.aabb_center = (center_x, center_y, center_z, size_x)
        mesh.aabb_min = (min_x, min_y, min_z, size_y)
        mesh.aabb_max = (max_x, max_y, max_z, size_z)

    return emd


def _write_emd(emd: EMD_File, path: str):
    data = bytearray()

    header_size = 28
    data.extend(b"\0" * header_size)

    model_count = len(emd.models)
    model_table_offset = len(data)
    model_ptr_positions: list[int] = []
    for _ in range(model_count):
        model_ptr_positions.append(len(data))
        data.extend(b"\0" * 4)

    _pad_data(data, 16)

    model_offsets: list[int] = []

    for model in emd.models:
        _pad_data(data, 16)
        model_off = len(data)
        model_offsets.append(model_off)

        mesh_count = len(model.meshes)
        data.extend(struct.pack("<H", getattr(model, "i_00", 0) & 0xFFFF))
        data.extend(struct.pack("<H", mesh_count & 0xFFFF))
        mesh_table_offset_rel = 8
        data.extend(struct.pack("<I", mesh_table_offset_rel))

        mesh_ptr_positions: list[int] = []
        # mesh_table_pos = len(data)
        for _ in range(mesh_count):
            mesh_ptr_positions.append(len(data))
            data.extend(b"\0" * 4)

        _pad_data(data, 16)

        for mesh_idx, mesh in enumerate(model.meshes):
            _pad_data(data, 16)
            mesh_off = len(data)

            submesh_count = len(mesh.submeshes)

            data.extend(struct.pack("<4f", *mesh.aabb_center))
            data.extend(struct.pack("<4f", *mesh.aabb_min))
            data.extend(struct.pack("<4f", *mesh.aabb_max))
            data.extend(b"\0" * 4)  # name offset placeholder
            data.extend(struct.pack("<H", getattr(mesh, "i_52", 0) & 0xFFFF))
            data.extend(struct.pack("<H", submesh_count & 0xFFFF))
            data.extend(b"\0" * 4)  # submesh table pointer placeholder

            # Mesh name
            name_rel = len(data) - mesh_off
            struct.pack_into("<I", data, mesh_off + 48, name_rel)
            data.extend(mesh.name.encode("utf8") + b"\0")
            _pad_data(data, 4)

            # Submesh pointer table
            submesh_table_rel = len(data) - mesh_off
            struct.pack_into("<I", data, mesh_off + 56, submesh_table_rel)
            submesh_ptr_positions: list[int] = []
            for _ in range(submesh_count):
                submesh_ptr_positions.append(len(data))
                data.extend(b"\0" * 4)

            _pad_data(data, 16)

            for sub_idx, sub in enumerate(mesh.submeshes):
                _pad_data(data, 16)
                struct.pack_into(
                    "<I",
                    data,
                    submesh_ptr_positions[sub_idx],
                    len(data) - mesh_off,
                )

                sub_off = len(data)
                vertex_size = get_vertex_size_from_flags(sub.vertex_flags)
                triangle_count = len(sub.triangle_groups)
                tex_def_count = len(sub.texture_sampler_defs)
                is_comp = bool(sub.vertex_flags & VERTEX_COMPRESSED)

                data.extend(struct.pack("<4f", *sub.aabb_center))
                data.extend(struct.pack("<4f", *sub.aabb_min))
                data.extend(struct.pack("<4f", *sub.aabb_max))
                data.extend(struct.pack("<I", sub.vertex_flags))
                data.extend(struct.pack("<I", vertex_size))
                data.extend(struct.pack("<I", len(sub.vertices)))
                data.extend(b"\0" * 4)  # vertex pointer placeholder
                data.extend(b"\0" * 4)  # submesh name pointer placeholder
                data.extend(b"\0")  # unknown byte
                data.extend(struct.pack("<B", tex_def_count & 0xFF))
                data.extend(struct.pack("<H", triangle_count & 0xFFFF))
                data.extend(b"\0" * 4)  # texture definitions pointer placeholder
                data.extend(b"\0" * 4)  # triangles pointer placeholder

                name_rel_sub = len(data) - sub_off
                struct.pack_into("<I", data, sub_off + 64, name_rel_sub)
                data.extend(sub.name.encode("utf8") + b"\0")
                _pad_data(data, 4)

                # Texture definitions
                tex_def_rel = len(data) - sub_off
                struct.pack_into("<I", data, sub_off + 72, tex_def_rel)
                for sampler in sub.texture_sampler_defs:
                    address_byte = (int(sampler.address_mode_v) << 4) | (
                        int(sampler.address_mode_u) & 0x0F
                    )
                    filtering_byte = (int(sampler.filtering_mag) << 4) | (
                        int(sampler.filtering_min) & 0x0F
                    )
                    data.extend(
                        struct.pack(
                            "<BB2B2f",
                            int(sampler.flag0) & 0xFF,
                            int(sampler.texture_index) & 0xFF,
                            address_byte & 0xFF,
                            filtering_byte & 0xFF,
                            float(sampler.scale_u),
                            float(sampler.scale_v),
                        )
                    )

                # Triangle pointer table
                tri_table_rel = len(data) - sub_off
                struct.pack_into("<I", data, sub_off + 76, tri_table_rel)
                tri_ptr_positions: list[int] = []
                for _ in range(triangle_count):
                    tri_ptr_positions.append(len(data))
                    data.extend(b"\0" * 4)

                # Triangles
                for tri_idx, tri_group in enumerate(sub.triangle_groups):
                    struct.pack_into(
                        "<I",
                        data,
                        tri_ptr_positions[tri_idx],
                        len(data) - sub_off,
                    )
                    tri_start = len(data)

                    face_count = len(tri_group.indices)
                    bones_count = len(tri_group.bone_names)
                    use_32 = face_count > 0xFFFF

                    data.extend(struct.pack("<I", face_count))
                    data.extend(struct.pack("<I", bones_count))
                    data.extend(struct.pack("<I", 16 if face_count > 0 else 0))
                    data.extend(b"\0" * 4)  # bone table pointer placeholder

                    if face_count > 0:
                        fmt = "<I" if use_32 else "<H"
                        for idx_val in tri_group.indices:
                            data.extend(struct.pack(fmt, int(idx_val)))

                    _pad_data(data, 4)

                    if bones_count > 0:
                        struct.pack_into(
                            "<I",
                            data,
                            tri_start + 12,
                            len(data) - tri_start,
                        )

                    bone_table_pos = len(data)
                    for _ in range(bones_count):
                        data.extend(b"\0" * 4)

                    for bone_name in tri_group.bone_names:
                        if bone_name != "NULL":
                            struct.pack_into(
                                "<I",
                                data,
                                bone_table_pos,
                                len(data) - tri_start,
                            )
                            data.extend(bone_name.encode("utf8") + b"\0")
                        bone_table_pos += 4

                    _pad_data(data, 4)

                # Vertex data last
                struct.pack_into("<I", data, sub_off + 60, len(data) - sub_off)

                for vertex in sub.vertices:
                    data.extend(struct.pack("<3f", *vertex.pos))
                    if sub.vertex_flags & VERTEX_NORMAL:
                        if is_comp:
                            for component in vertex.normal:
                                data.extend(struct.pack("<H", float_to_half(component)))
                            data.extend(b"\0\0")
                        else:
                            data.extend(struct.pack("<3f", *vertex.normal))
                    if sub.vertex_flags & VERTEX_TEXUV:
                        if is_comp:
                            u, v = vertex.uv
                            data.extend(
                                struct.pack(
                                    "<2H",
                                    float_to_half(u),
                                    float_to_half(v),
                                )
                            )
                        else:
                            data.extend(struct.pack("<2f", *vertex.uv))
                    if sub.vertex_flags & VERTEX_TEX2UV:
                        if is_comp:
                            u2, v2 = vertex.uv2
                            data.extend(
                                struct.pack(
                                    "<2H",
                                    float_to_half(u2),
                                    float_to_half(v2),
                                )
                            )
                        else:
                            data.extend(struct.pack("<2f", *vertex.uv2))
                    if sub.vertex_flags & VERTEX_TANGENT:
                        tx, ty, tz = vertex.tangent
                        if is_comp:
                            data.extend(
                                struct.pack(
                                    "<3H",
                                    float_to_half(tx),
                                    float_to_half(ty),
                                    float_to_half(tz),
                                )
                            )
                            data.extend(b"\0\0")
                        else:
                            data.extend(struct.pack("<3f", tx, ty, tz))
                    if sub.vertex_flags & VERTEX_COLOR:
                        r, g, b, a = vertex.color
                        data.extend(
                            struct.pack(
                                "<4B",
                                int(r * 255),
                                int(g * 255),
                                int(b * 255),
                                int(a * 255),
                            )
                        )
                    if sub.vertex_flags & VERTEX_BLENDWEIGHT:
                        data.extend(
                            struct.pack(
                                "<4B",
                                *[int(bi) & 0xFF for bi in vertex.bone_ids],
                            )
                        )
                        if is_comp:
                            for weight_value in vertex.bone_weights[:3]:
                                data.extend(struct.pack("<H", float_to_half(weight_value)))
                            data.extend(b"\0\0")
                        else:
                            data.extend(struct.pack("<3f", *vertex.bone_weights[:3]))

            struct.pack_into(
                "<I",
                data,
                mesh_ptr_positions[mesh_idx],
                mesh_off - model_off,
            )

    _pad_data(data, 4)
    model_name_table_offset = len(data)
    model_name_ptr_positions: list[int] = []
    for _ in emd.models:
        model_name_ptr_positions.append(len(data))
        data.extend(b"\0" * 4)

    for idx, model in enumerate(emd.models):
        name_off = len(data)
        struct.pack_into("<I", data, model_name_ptr_positions[idx], name_off)
        data.extend(model.name.encode("utf8") + b"\0")

    for idx, model_off in enumerate(model_offsets):
        struct.pack_into("<I", data, model_ptr_positions[idx], model_off)

    struct.pack_into("<I", data, 0, EMD_SIGNATURE)
    struct.pack_into("<H", data, 4, 0xFFFE)
    struct.pack_into("<H", data, 6, header_size)
    struct.pack_into(
        "<I",
        data,
        8,
        emd.version if getattr(emd, "version", None) else 0x201,
    )
    struct.pack_into("<H", data, 18, model_count)
    struct.pack_into("<I", data, 20, model_table_offset)
    struct.pack_into("<I", data, 24, model_name_table_offset)

    with open(path, "wb") as handle:
        handle.write(data)


def export_selected(
    context: bpy.types.Context,
    output_dir: str,
) -> list[str]:
    written: list[str] = []
    for obj in context.selected_objects:
        if obj.type != "MESH":
            continue
        arm = obj.parent if obj.parent and obj.parent.type == "ARMATURE" else None
        if arm is None:
            print(f"Skipping {obj.name}: requires an armature parent.")
            continue
        remove_unused_vertex_groups(obj)
        emd = _build_emd_from_object(obj, arm)
        safe_name = bpy.path.clean_name(obj.name)
        out_path = os.path.join(output_dir, f"{safe_name}.emd")
        try:
            _write_emd(emd, out_path)
            written.append(out_path)
        except Exception as error:
            print(f"Failed to export {obj.name}: {error}")
    return written


__all__ = [
    "export_selected",
]
