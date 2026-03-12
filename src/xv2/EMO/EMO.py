from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ...utils import float_to_half, read_cstring
from ...utils.binary import i16, u16, u32, u64
from ..EMA.skeleton import parse_ema_skeleton_as_esk
from ..EMD import (
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
    read_texture_sampler_defs,
    read_vertices,
)
from ..ESK import ESK_File

EMO_SIGNATURE = 1330464035  # "#EMO"
EMG_SIGNATURE = 1196246307  # "#EMG"


def _pad_data(data: bytearray, alignment: int) -> None:
    padding = (-len(data)) % alignment
    if padding:
        data.extend(b"\x00" * padding)


@dataclass
class EMGSubmesh:
    material_name: str
    texture_list_index: int
    faces: list[int] = field(default_factory=list)
    bones: list[int] = field(default_factory=list)


@dataclass
class EMGMesh:
    vertex_flags: int
    vertices: list[EMD_Vertex] = field(default_factory=list)
    texture_lists: list[list[EMD_TextureSamplerDef]] = field(default_factory=list)
    submeshes: list[EMGSubmesh] = field(default_factory=list)


@dataclass
class EMGFile:
    linked_bone_index: int
    meshes: list[EMGMesh] = field(default_factory=list)


@dataclass
class EMOPart:
    name: str
    linked_bone_name: str
    emg_files: list[EMGFile] = field(default_factory=list)


@dataclass
class EMOFile:
    version: int
    materials_count: int
    i_24: int
    skeleton: ESK_File
    parts: list[EMOPart] = field(default_factory=list)
    emd_file: EMD_File | None = None


@dataclass
class _PendingVertexPatch:
    field_offset: int
    mesh_offset: int
    vertex_blob: bytes


def _decode_fixed_cstring(data: bytes, offset: int, size: int) -> str:
    raw = data[offset : offset + size]
    return raw.split(b"\x00", 1)[0].decode("utf8", errors="ignore")


def _clone_vertex(vertex: EMD_Vertex) -> EMD_Vertex:
    cloned = EMD_Vertex()
    cloned.pos = tuple(vertex.pos)
    cloned.normal = tuple(vertex.normal)
    cloned.uv = tuple(vertex.uv)
    cloned.uv2 = tuple(vertex.uv2)
    cloned.tangent = tuple(vertex.tangent)
    cloned.color = tuple(vertex.color)
    cloned.bone_ids = list(vertex.bone_ids)
    cloned.bone_weights = list(vertex.bone_weights)
    return cloned


def parse_emo(path: str) -> EMOFile:
    with open(path, "rb") as file_handle:
        data = file_handle.read()
    return parse_emo_bytes(data)


def parse_emo_bytes(data: bytes) -> EMOFile:
    if u32(data, 0) != EMO_SIGNATURE:
        raise ValueError('EMO signature "#EMO" not found at offset 0.')

    version = int(u32(data, 8))
    parts_header_offset = int(u32(data, 12))
    skeleton_offset = int(u32(data, 16))
    i_24 = int(u64(data, 24))

    if parts_header_offset <= 0 or parts_header_offset + 8 > len(data):
        raise ValueError("EMO parts header offset is invalid.")
    if skeleton_offset <= 0:
        raise ValueError("EMO file has no skeleton offset.")

    part_count = int(u16(data, parts_header_offset + 0))
    materials_count = int(u16(data, parts_header_offset + 2))
    names_offset_rel = int(u32(data, parts_header_offset + 4))
    names_offset = parts_header_offset + names_offset_rel if names_offset_rel else 0
    part_ptr_offset = parts_header_offset + 8

    skeleton, part_indices = parse_ema_skeleton_as_esk(data, skeleton_offset)
    part_to_bone_name = _build_part_to_bone_map(part_count, part_indices, skeleton)

    parts: list[EMOPart] = []
    for part_index in range(part_count):
        part_offset = parts_header_offset + int(u32(data, part_ptr_offset + (part_index * 4)))
        name = f"part_{part_index:02d}"
        if names_offset:
            name_rel = int(u32(data, names_offset + (part_index * 4)))
            if name_rel:
                parsed_name = read_cstring(data, parts_header_offset + name_rel)
                if parsed_name:
                    name = parsed_name

        linked_bone_name = part_to_bone_name.get(part_index, "")
        emg_files = _parse_part_emgs(data, part_offset)
        parts.append(EMOPart(name=name, linked_bone_name=linked_bone_name, emg_files=emg_files))

    emd_file = convert_emo_to_emd(parts, skeleton, version=version)
    return EMOFile(
        version=version,
        materials_count=materials_count,
        i_24=i_24,
        skeleton=skeleton,
        parts=parts,
        emd_file=emd_file,
    )


def _build_part_to_bone_map(
    part_count: int,
    part_indices: list[int],
    skeleton: ESK_File,
) -> dict[int, str]:
    part_to_bone_name: dict[int, str] = {}
    for bone_index, part_index in enumerate(part_indices):
        if not (0 <= part_index < part_count):
            continue
        if part_index in part_to_bone_name:
            continue
        if 0 <= bone_index < len(skeleton.bones):
            bone_name = skeleton.bones[bone_index].name
            if bone_name:
                part_to_bone_name[part_index] = bone_name
    return part_to_bone_name


def _parse_part_emgs(data: bytes, part_offset: int) -> list[EMGFile]:
    emg_count = int(u32(data, part_offset + 0))
    emg_files: list[EMGFile] = []
    for emg_index in range(emg_count):
        emg_offset = part_offset + int(u32(data, part_offset + 4 + (emg_index * 4)))
        emg_files.append(_parse_emg(data, emg_offset))
    return emg_files


def _parse_emg(data: bytes, emg_offset: int) -> EMGFile:
    if u32(data, emg_offset) != EMG_SIGNATURE:
        raise ValueError(f'EMG signature "#EMG" not found at offset {emg_offset}.')

    linked_bone_index = int(u16(data, emg_offset + 4))
    mesh_count = int(u16(data, emg_offset + 6))
    meshes: list[EMGMesh] = []
    for mesh_index in range(mesh_count):
        mesh_offset = emg_offset + int(u32(data, emg_offset + 8 + (mesh_index * 4)))
        meshes.append(_parse_emg_mesh(data, mesh_offset))
    return EMGFile(linked_bone_index=linked_bone_index, meshes=meshes)


def _parse_emg_mesh(data: bytes, mesh_offset: int) -> EMGMesh:
    vertex_flags = int(u32(data, mesh_offset + 0))
    texture_lists_count = int(u32(data, mesh_offset + 4))
    texture_lists_offset_rel = int(u32(data, mesh_offset + 12))
    vertex_count = int(u16(data, mesh_offset + 16))
    vertex_size = int(u16(data, mesh_offset + 18))
    vertex_offset = mesh_offset + int(u32(data, mesh_offset + 20))
    submesh_count = int(u16(data, mesh_offset + 26))
    submesh_list_offset = mesh_offset + int(u32(data, mesh_offset + 28))

    texture_lists: list[list[EMD_TextureSamplerDef]] = []
    if texture_lists_count > 0 and texture_lists_offset_rel > 0:
        texture_ptr_offset = mesh_offset + texture_lists_offset_rel
        for texture_index in range(texture_lists_count):
            texture_offset = mesh_offset + int(u32(data, texture_ptr_offset + (texture_index * 4)))
            sampler_count = int(u32(data, texture_offset + 0))
            texture_lists.append(
                read_texture_sampler_defs(data, texture_offset + 4, sampler_count)
            )

    submeshes: list[EMGSubmesh] = []
    for submesh_index in range(submesh_count):
        submesh_offset = mesh_offset + int(u32(data, submesh_list_offset + (submesh_index * 4)))
        submeshes.append(_parse_emg_submesh(data, submesh_offset))

    vertices = read_vertices(vertex_flags, data, vertex_offset, vertex_count, vertex_size)
    return EMGMesh(
        vertex_flags=vertex_flags,
        vertices=vertices,
        texture_lists=texture_lists,
        submeshes=submeshes,
    )


def _parse_emg_submesh(data: bytes, submesh_offset: int) -> EMGSubmesh:
    texture_list_index = int(u16(data, submesh_offset + 16))
    face_count = int(u16(data, submesh_offset + 18))
    bone_count = int(u16(data, submesh_offset + 20))
    material_name = _decode_fixed_cstring(data, submesh_offset + 22, 32)

    faces: list[int] = []
    face_offset = submesh_offset + 54
    for face_index in range(face_count):
        faces.append(int(i16(data, face_offset + (face_index * 2))))

    bones: list[int] = []
    bone_offset = face_offset + (face_count * 2)
    for bone_index in range(bone_count):
        bones.append(int(u16(data, bone_offset + (bone_index * 2))))

    return EMGSubmesh(
        material_name=material_name,
        texture_list_index=texture_list_index,
        faces=faces,
        bones=bones,
    )


def convert_emo_to_emd(parts: list[EMOPart], skeleton: ESK_File, *, version: int) -> EMD_File:
    emd = EMD_File()
    emd.version = version

    for part_index, part in enumerate(parts):
        model = EMD_Model()
        model.name = part.linked_bone_name or part.name or f"part_{part_index:02d}"

        for emg_file in part.emg_files:
            for emg_mesh_index, emg_mesh in enumerate(emg_file.meshes):
                mesh_name = part.name or f"emg_mesh_{emg_mesh_index:02d}"
                model.meshes.extend(
                    _convert_emg_mesh_to_emd_meshes(
                        emg_mesh,
                        skeleton=skeleton,
                        part_bone_name=part.linked_bone_name,
                        mesh_name=mesh_name,
                    )
                )

        if model.meshes:
            emd.models.append(model)

    return emd


def _convert_emg_mesh_to_emd_meshes(
    emg_mesh: EMGMesh,
    *,
    skeleton: ESK_File,
    part_bone_name: str,
    mesh_name: str,
) -> list[EMD_Mesh]:
    emd_meshes: list[EMD_Mesh] = []
    for submesh_index, emg_submesh in enumerate(emg_mesh.submeshes):
        vertex_lookup: dict[int, int] = {}
        submesh_vertices: list[EMD_Vertex] = []
        mapped_indices: list[int] = []
        for source_index in emg_submesh.faces:
            if source_index < 0 or source_index >= len(emg_mesh.vertices):
                continue
            mapped_index = vertex_lookup.get(source_index)
            if mapped_index is None:
                mapped_index = len(submesh_vertices)
                vertex_lookup[source_index] = mapped_index
                submesh_vertices.append(_clone_vertex(emg_mesh.vertices[source_index]))
            mapped_indices.append(mapped_index)

        if len(mapped_indices) < 3:
            continue

        triangle_group = EMD_Triangles()
        triangle_group.indices = mapped_indices
        triangle_group.bone_names = _resolve_bone_names(emg_submesh.bones, skeleton)

        submesh = EMD_Submesh()
        submesh.name = emg_submesh.material_name or f"{mesh_name}_submesh_{submesh_index:02d}"
        submesh.vertex_flags = int(emg_mesh.vertex_flags)
        submesh.vertices = submesh_vertices
        submesh.triangle_groups = [triangle_group]
        submesh.faces = [
            (mapped_indices[i], mapped_indices[i + 1], mapped_indices[i + 2])
            for i in range(0, len(mapped_indices) - 2, 3)
        ]
        if 0 <= emg_submesh.texture_list_index < len(emg_mesh.texture_lists):
            submesh.texture_sampler_defs = list(
                emg_mesh.texture_lists[emg_submesh.texture_list_index]
            )

        if not (submesh.vertex_flags & VERTEX_BLENDWEIGHT):
            rigid_bone_name = part_bone_name or (
                triangle_group.bone_names[0] if len(triangle_group.bone_names) == 1 else ""
            )
            if rigid_bone_name:
                # Rigid EMO pieces still need a weight channel so they follow skeleton animation.
                submesh.vertex_flags |= VERTEX_BLENDWEIGHT
                triangle_group.bone_names = [rigid_bone_name]
                for vertex in submesh.vertices:
                    vertex.bone_ids = [0, 0, 0, 0]
                    vertex.bone_weights = [1.0, 0.0, 0.0, 0.0]

        mesh = EMD_Mesh()
        mesh.name = mesh_name
        mesh.submeshes = [submesh]
        emd_meshes.append(mesh)

    return emd_meshes


def _resolve_bone_names(indices: list[int], skeleton: ESK_File) -> list[str]:
    names: list[str] = []
    for index in indices:
        if 0 <= index < len(skeleton.bones):
            bone_name = skeleton.bones[index].name
            if bone_name:
                names.append(bone_name)
    return names


def _to_u16_index(value: int) -> int:
    if value < 0:
        return 0xFFFF
    return value & 0xFFFF


def _flatten_faces(faces: list[tuple[int, int, int]]) -> list[int]:
    flattened: list[int] = []
    for i0, i1, i2 in faces:
        flattened.extend((int(i0), int(i1), int(i2)))
    return flattened


def convert_emd_to_emo_parts(emd: EMD_File, skeleton: ESK_File) -> tuple[list[EMOPart], int]:
    bone_index_by_name = {
        bone.name: bone.index
        for bone in skeleton.bones
        if getattr(bone, "name", None)
    }
    parts: list[EMOPart] = []
    materials_count = 0

    for model_index, model in enumerate(emd.models):
        part_name = (model.name or f"part_{model_index:02d}").strip() or f"part_{model_index:02d}"
        linked_bone_name = part_name if part_name in bone_index_by_name else ""
        linked_bone_index = (
            int(bone_index_by_name[linked_bone_name]) if linked_bone_name else 0xFFFF
        )

        part = EMOPart(name=part_name, linked_bone_name=linked_bone_name, emg_files=[])
        for mesh in model.meshes:
            emg = EMGFile(linked_bone_index=linked_bone_index, meshes=[])
            for submesh in mesh.submeshes:
                tri_groups = [group for group in (submesh.triangle_groups or []) if group.indices]
                if not tri_groups and submesh.faces:
                    fallback = EMD_Triangles()
                    fallback.indices = _flatten_faces(submesh.faces)
                    fallback.bone_names = []
                    tri_groups = [fallback]

                for tri_group in tri_groups:
                    indices = [int(index) for index in tri_group.indices]
                    if not indices:
                        continue
                    if any(index < 0 or index > 0x7FFF for index in indices):
                        raise ValueError(
                            "EMO export only supports submesh indices in signed 16-bit range."
                        )

                    texture_samplers = list(submesh.texture_sampler_defs or [])
                    material_name = (submesh.name or "EMO_Submesh").strip() or "EMO_Submesh"
                    if len(material_name) > 32:
                        material_name = material_name[:32]

                    emg_submesh = EMGSubmesh(
                        material_name=material_name,
                        texture_list_index=0,
                        faces=indices,
                        bones=[
                            int(bone_index_by_name[name])
                            for name in (tri_group.bone_names or [])
                            if name in bone_index_by_name
                        ],
                    )

                    emg_mesh = EMGMesh(
                        vertex_flags=int(submesh.vertex_flags),
                        vertices=[_clone_vertex(vertex) for vertex in (submesh.vertices or [])],
                        texture_lists=[texture_samplers],
                        submeshes=[emg_submesh],
                    )
                    emg.meshes.append(emg_mesh)
                    materials_count += 1

            if emg.meshes:
                part.emg_files.append(emg)

        if part.emg_files:
            parts.append(part)

    return parts, materials_count


def _compute_mesh_bounds(
    vertices: list[EMD_Vertex],
) -> tuple[tuple[float, float, float, float], ...]:
    if not vertices:
        zero = (0.0, 0.0, 0.0, 0.0)
        return zero, zero, zero

    xs = [float(vertex.pos[0]) for vertex in vertices]
    ys = [float(vertex.pos[1]) for vertex in vertices]
    zs = [float(vertex.pos[2]) for vertex in vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    center_z = (min_z + max_z) * 0.5
    return (
        (center_x, center_y, center_z, size_x),
        (min_x, min_y, min_z, size_y),
        (max_x, max_y, max_z, size_z),
    )


def _compute_submesh_barycenter(
    submesh: EMGSubmesh,
    vertices: list[EMD_Vertex],
) -> tuple[float, float, float, float]:
    positions: list[tuple[float, float, float]] = []
    for index in submesh.faces:
        if 0 <= index < len(vertices):
            positions.append(tuple(vertices[index].pos))
    if not positions:
        return 0.0, 0.0, 0.0, 1.0
    count = float(len(positions))
    x = sum(position[0] for position in positions) / count
    y = sum(position[1] for position in positions) / count
    z = sum(position[2] for position in positions) / count
    return x, y, z, 1.0


def _pack_sampler_defs(samplers: list[EMD_TextureSamplerDef]) -> bytes:
    blob = bytearray()
    for sampler in samplers:
        address_byte = (int(sampler.address_mode_v) << 4) | (int(sampler.address_mode_u) & 0x0F)
        filtering_byte = (int(sampler.filtering_mag) << 4) | (int(sampler.filtering_min) & 0x0F)
        blob.extend(
            struct.pack(
                "<BBBBff",
                int(sampler.flag0) & 0xFF,
                int(sampler.texture_index) & 0xFF,
                address_byte & 0xFF,
                filtering_byte & 0xFF,
                float(sampler.scale_u),
                float(sampler.scale_v),
            )
        )
    return bytes(blob)


def _encode_vertex_blob(vertices: list[EMD_Vertex], flags: int) -> bytes:
    blob = bytearray()
    compressed = bool(flags & VERTEX_COMPRESSED)
    for vertex in vertices:
        if flags & VERTEX_POSITION:
            blob.extend(
                struct.pack(
                    "<3f",
                    float(vertex.pos[0]),
                    float(vertex.pos[1]),
                    float(vertex.pos[2]),
                )
            )
        if flags & VERTEX_NORMAL:
            if compressed:
                blob.extend(struct.pack("<H", float_to_half(float(vertex.normal[0]))))
                blob.extend(struct.pack("<H", float_to_half(float(vertex.normal[1]))))
                blob.extend(struct.pack("<H", float_to_half(float(vertex.normal[2]))))
                blob.extend(b"\x00\x00")
            else:
                blob.extend(
                    struct.pack(
                        "<3f",
                        float(vertex.normal[0]),
                        float(vertex.normal[1]),
                        float(vertex.normal[2]),
                    )
                )
        if flags & VERTEX_TEXUV:
            uv_u = float(vertex.uv[0])
            uv_v = 1.0 - float(vertex.uv[1])
            if compressed:
                blob.extend(struct.pack("<H", float_to_half(uv_u)))
                blob.extend(struct.pack("<H", float_to_half(uv_v)))
            else:
                blob.extend(struct.pack("<2f", uv_u, uv_v))
        if flags & VERTEX_TEX2UV:
            uv2_u = float(vertex.uv2[0])
            uv2_v = 1.0 - float(vertex.uv2[1])
            if compressed:
                blob.extend(struct.pack("<H", float_to_half(uv2_u)))
                blob.extend(struct.pack("<H", float_to_half(uv2_v)))
            else:
                blob.extend(struct.pack("<2f", uv2_u, uv2_v))
        if flags & VERTEX_TANGENT:
            if compressed:
                blob.extend(struct.pack("<H", float_to_half(float(vertex.tangent[0]))))
                blob.extend(struct.pack("<H", float_to_half(float(vertex.tangent[1]))))
                blob.extend(struct.pack("<H", float_to_half(float(vertex.tangent[2]))))
                blob.extend(b"\x00\x00")
            else:
                blob.extend(
                    struct.pack(
                        "<3f",
                        float(vertex.tangent[0]),
                        float(vertex.tangent[1]),
                        float(vertex.tangent[2]),
                    )
                )
        if flags & VERTEX_COLOR:
            red = max(0, min(255, int(round(float(vertex.color[0]) * 255.0))))
            green = max(0, min(255, int(round(float(vertex.color[1]) * 255.0))))
            blue = max(0, min(255, int(round(float(vertex.color[2]) * 255.0))))
            alpha = max(0, min(255, int(round(float(vertex.color[3]) * 255.0))))
            blob.extend(struct.pack("<4B", red, green, blue, alpha))
        if flags & VERTEX_BLENDWEIGHT:
            ids = [int(index) & 0xFF for index in (vertex.bone_ids or [0, 0, 0, 0])[:4]]
            while len(ids) < 4:
                ids.append(0)
            blob.extend(struct.pack("<4B", ids[0], ids[1], ids[2], ids[3]))
            weights = [
                float(weight)
                for weight in (vertex.bone_weights or [0.0, 0.0, 0.0, 0.0])[:4]
            ]
            while len(weights) < 4:
                weights.append(0.0)
            if compressed:
                blob.extend(struct.pack("<H", float_to_half(weights[0])))
                blob.extend(struct.pack("<H", float_to_half(weights[1])))
                blob.extend(struct.pack("<H", float_to_half(weights[2])))
                blob.extend(b"\x00\x00")
            else:
                blob.extend(struct.pack("<3f", weights[0], weights[1], weights[2]))
    return bytes(blob)


def _build_emg_submesh_bytes(submesh: EMGSubmesh, vertices: list[EMD_Vertex]) -> bytes:
    bary_x, bary_y, bary_z, bary_w = _compute_submesh_barycenter(submesh, vertices)
    material_name = (submesh.material_name or "").encode("utf8", errors="ignore")
    if len(material_name) > 32:
        raise ValueError("EMO export material names must be at most 32 bytes.")
    material_name_blob = material_name + (b"\x00" * (32 - len(material_name)))

    blob = bytearray()
    blob.extend(struct.pack("<4f", bary_x, bary_y, bary_z, bary_w))
    blob.extend(struct.pack("<H", int(submesh.texture_list_index) & 0xFFFF))
    blob.extend(struct.pack("<H", len(submesh.faces) & 0xFFFF))
    blob.extend(struct.pack("<H", len(submesh.bones) & 0xFFFF))
    blob.extend(material_name_blob)
    for face_index in submesh.faces:
        blob.extend(struct.pack("<h", int(face_index)))
    for bone_index in submesh.bones:
        blob.extend(struct.pack("<H", int(bone_index) & 0xFFFF))
    return bytes(blob)


def _build_emg_mesh_bytes(mesh: EMGMesh) -> tuple[bytes, list[_PendingVertexPatch]]:
    mesh_bytes = bytearray()
    texture_lists_count = len(mesh.texture_lists)
    submesh_count = len(mesh.submeshes)
    vertex_count = len(mesh.vertices)
    vertex_size = get_vertex_size_from_flags(int(mesh.vertex_flags))
    aabb_center, aabb_min, aabb_max = _compute_mesh_bounds(mesh.vertices)

    mesh_bytes.extend(struct.pack("<i", int(mesh.vertex_flags)))
    mesh_bytes.extend(struct.pack("<i", texture_lists_count))
    mesh_bytes.extend(struct.pack("<i", 0))  # I_08
    mesh_bytes.extend(struct.pack("<i", 80 if texture_lists_count > 0 else 0))
    mesh_bytes.extend(struct.pack("<H", vertex_count & 0xFFFF))
    mesh_bytes.extend(struct.pack("<H", vertex_size & 0xFFFF))
    vertex_offset_field = len(mesh_bytes)
    mesh_bytes.extend(struct.pack("<I", 0))  # patched later
    mesh_bytes.extend(struct.pack("<H", 0))  # strips
    mesh_bytes.extend(struct.pack("<H", submesh_count & 0xFFFF))
    submesh_offset_field = len(mesh_bytes)
    mesh_bytes.extend(struct.pack("<I", 0))  # patched later
    mesh_bytes.extend(struct.pack("<4f", *aabb_center))
    mesh_bytes.extend(struct.pack("<4f", *aabb_min))
    mesh_bytes.extend(struct.pack("<4f", *aabb_max))

    texture_ptrs_start = len(mesh_bytes)
    mesh_bytes.extend(b"\x00" * (4 * texture_lists_count))
    for texture_list_index, texture_list in enumerate(mesh.texture_lists):
        texture_offset = len(mesh_bytes)
        struct.pack_into(
            "<I",
            mesh_bytes,
            texture_ptrs_start + (texture_list_index * 4),
            texture_offset,
        )
        mesh_bytes.extend(struct.pack("<i", len(texture_list)))
        mesh_bytes.extend(_pack_sampler_defs(texture_list))

    if submesh_count > 0:
        submesh_ptrs_start = len(mesh_bytes)
        struct.pack_into("<I", mesh_bytes, submesh_offset_field, submesh_ptrs_start)
        mesh_bytes.extend(b"\x00" * (4 * submesh_count))
        for submesh_index, submesh in enumerate(mesh.submeshes):
            _pad_data(mesh_bytes, 16)
            submesh_offset = len(mesh_bytes)
            struct.pack_into(
                "<I",
                mesh_bytes,
                submesh_ptrs_start + (submesh_index * 4),
                submesh_offset,
            )
            mesh_bytes.extend(_build_emg_submesh_bytes(submesh, mesh.vertices))

    vertex_blob = _encode_vertex_blob(mesh.vertices, int(mesh.vertex_flags))
    pending = [
        _PendingVertexPatch(
            field_offset=vertex_offset_field,
            mesh_offset=0,
            vertex_blob=vertex_blob,
        )
    ]
    return bytes(mesh_bytes), pending


def _build_emg_bytes(emg: EMGFile) -> tuple[bytes, list[_PendingVertexPatch]]:
    emg_bytes = bytearray()
    mesh_count = len(emg.meshes)
    emg_bytes.extend(struct.pack("<I", EMG_SIGNATURE))
    emg_bytes.extend(struct.pack("<H", int(emg.linked_bone_index) & 0xFFFF))
    emg_bytes.extend(struct.pack("<H", mesh_count & 0xFFFF))
    mesh_ptrs_start = len(emg_bytes)
    emg_bytes.extend(b"\x00" * (4 * mesh_count))

    pending: list[_PendingVertexPatch] = []
    for mesh_index, mesh in enumerate(emg.meshes):
        _pad_data(emg_bytes, 16)
        mesh_start = len(emg_bytes)
        struct.pack_into("<I", emg_bytes, mesh_ptrs_start + (mesh_index * 4), mesh_start)
        mesh_blob, mesh_pending = _build_emg_mesh_bytes(mesh)
        emg_bytes.extend(mesh_blob)
        for patch in mesh_pending:
            pending.append(
                _PendingVertexPatch(
                    field_offset=mesh_start + patch.field_offset,
                    mesh_offset=mesh_start + patch.mesh_offset,
                    vertex_blob=patch.vertex_blob,
                )
            )

    return bytes(emg_bytes), pending


def _build_emo_part_bytes(
    part: EMOPart,
    linked_bone_index: int,
) -> tuple[bytes, list[_PendingVertexPatch]]:
    part_bytes = bytearray()
    emg_count = len(part.emg_files)
    part_bytes.extend(struct.pack("<i", emg_count))
    emg_ptrs_start = len(part_bytes)
    part_bytes.extend(b"\x00" * (4 * emg_count))
    _pad_data(part_bytes, 16)

    pending: list[_PendingVertexPatch] = []
    for emg_index, emg_file in enumerate(part.emg_files):
        emg_start = len(part_bytes)
        struct.pack_into("<I", part_bytes, emg_ptrs_start + (emg_index * 4), emg_start)
        effective_emg = EMGFile(
            linked_bone_index=(
                int(emg_file.linked_bone_index)
                if int(emg_file.linked_bone_index) >= 0
                else linked_bone_index
            ),
            meshes=list(emg_file.meshes),
        )
        emg_blob, emg_pending = _build_emg_bytes(effective_emg)
        part_bytes.extend(emg_blob)
        for patch in emg_pending:
            pending.append(
                _PendingVertexPatch(
                    field_offset=emg_start + patch.field_offset,
                    mesh_offset=emg_start + patch.mesh_offset,
                    vertex_blob=patch.vertex_blob,
                )
            )

    return bytes(part_bytes), pending


def _build_emo_skeleton_bytes(
    skeleton: ESK_File,
    *,
    part_index_by_bone_name: dict[str, int],
) -> bytes:
    bones = list(skeleton.bones)
    bone_count = len(bones)
    skeleton_bytes = bytearray()

    skeleton_bytes.extend(struct.pack("<H", bone_count & 0xFFFF))
    skeleton_bytes.extend(struct.pack("<H", 0))  # IK2 count
    skeleton_bytes.extend(struct.pack("<I", 0))  # IK count
    skeleton_bytes.extend(struct.pack("<I", 64))  # bone offset
    names_offset_field = len(skeleton_bytes)
    skeleton_bytes.extend(struct.pack("<I", 0))  # names offset
    skeleton_bytes.extend(struct.pack("<I", 0))  # IK2 offset
    skeleton_bytes.extend(struct.pack("<I", 0))  # IK2 names
    skeleton_bytes.extend(struct.pack("<I", 0))  # extra values
    skeleton_bytes.extend(struct.pack("<I", 0))  # abs matrix
    skeleton_bytes.extend(struct.pack("<I", 0))  # IK offset
    skeleton_bytes.extend(struct.pack("<i", 0))  # I_36
    skeleton_bytes.extend(struct.pack("<i", 0))  # I_40
    skeleton_bytes.extend(struct.pack("<i", 0))  # I_44
    skeleton_bytes.extend(struct.pack("<i", 0))  # I_48
    skeleton_bytes.extend(struct.pack("<H", 0))  # I_52
    skeleton_bytes.extend(struct.pack("<H", int(skeleton.skeleton_flag) & 0xFFFF))
    skeleton_bytes.extend(struct.pack("<Q", int(skeleton.skeleton_id) & 0xFFFFFFFFFFFFFFFF))

    for bone in bones:
        skeleton_bytes.extend(struct.pack("<H", _to_u16_index(int(bone.parent_index))))
        skeleton_bytes.extend(struct.pack("<H", _to_u16_index(int(bone.child_index))))
        skeleton_bytes.extend(struct.pack("<H", _to_u16_index(int(bone.sibling_index))))
        part_index = part_index_by_bone_name.get(str(bone.name), 0xFFFF)
        skeleton_bytes.extend(struct.pack("<H", _to_u16_index(int(part_index))))
        skeleton_bytes.extend(struct.pack("<H", 0xFFFF))  # I_08
        skeleton_bytes.extend(struct.pack("<H", 0))  # IK flag
        skeleton_bytes.extend(struct.pack("<f", 0.0))  # F_12
        matrix = bone.matrix.transposed()
        for row in range(4):
            for col in range(4):
                skeleton_bytes.extend(struct.pack("<f", float(matrix[row][col])))

    names_offset = len(skeleton_bytes)
    struct.pack_into("<I", skeleton_bytes, names_offset_field, names_offset)
    name_ptrs_start = len(skeleton_bytes)
    skeleton_bytes.extend(b"\x00" * (4 * bone_count))
    for bone_index, bone in enumerate(bones):
        name_offset = len(skeleton_bytes)
        struct.pack_into("<I", skeleton_bytes, name_ptrs_start + (bone_index * 4), name_offset)
        skeleton_bytes.extend(str(bone.name).encode("utf8", errors="ignore") + b"\x00")

    _pad_data(skeleton_bytes, 16)
    return bytes(skeleton_bytes)


def build_emo_bytes(emo_file: EMOFile) -> bytes:
    parts = list(emo_file.parts)
    part_count = len(parts)
    materials_count = int(emo_file.materials_count)
    if materials_count <= 0:
        materials_count = sum(len(emg.meshes) for part in parts for emg in part.emg_files)

    part_index_by_bone_name: dict[str, int] = {}
    bone_index_by_name = {
        bone.name: bone.index for bone in emo_file.skeleton.bones if getattr(bone, "name", None)
    }
    for part_index, part in enumerate(parts):
        if part.linked_bone_name:
            part_index_by_bone_name[part.linked_bone_name] = part_index

    out = bytearray()
    out.extend(struct.pack("<I", EMO_SIGNATURE))
    out.extend(struct.pack("<H", 0xFFFE))
    out.extend(struct.pack("<H", 32))
    out.extend(struct.pack("<I", int(emo_file.version)))
    out.extend(struct.pack("<I", 32))  # parts offset
    skeleton_offset_field = len(out)
    out.extend(struct.pack("<I", 0))
    vertices_offset_field = len(out)
    out.extend(struct.pack("<I", 0))
    out.extend(struct.pack("<Q", int(emo_file.i_24) & 0xFFFFFFFFFFFFFFFF))

    parts_header_start = len(out)
    out.extend(struct.pack("<H", part_count & 0xFFFF))
    out.extend(struct.pack("<H", materials_count & 0xFFFF))
    names_offset_field = len(out)
    out.extend(struct.pack("<I", 0))
    part_ptrs_start = len(out)
    out.extend(b"\x00" * (4 * part_count))
    _pad_data(out, 16)

    pending_vertices: list[_PendingVertexPatch] = []
    for part_index, part in enumerate(parts):
        part_start = len(out)
        part_linked_index = (
            int(bone_index_by_name.get(part.linked_bone_name, 0xFFFF))
            if part.linked_bone_name
            else 0xFFFF
        )
        part_blob, part_pending = _build_emo_part_bytes(part, part_linked_index)
        struct.pack_into(
            "<I",
            out,
            part_ptrs_start + (part_index * 4),
            part_start - parts_header_start,
        )
        out.extend(part_blob)
        for patch in part_pending:
            pending_vertices.append(
                _PendingVertexPatch(
                    field_offset=part_start + patch.field_offset,
                    mesh_offset=part_start + patch.mesh_offset,
                    vertex_blob=patch.vertex_blob,
                )
            )
        _pad_data(out, 16)

    names_start = len(out)
    struct.pack_into("<I", out, names_offset_field, names_start - parts_header_start)
    name_ptrs_start = len(out)
    out.extend(b"\x00" * (4 * part_count))
    for part_index, part in enumerate(parts):
        part_name = part.linked_bone_name or part.name or f"part_{part_index:02d}"
        name_offset = len(out)
        struct.pack_into(
            "<I",
            out,
            name_ptrs_start + (part_index * 4),
            name_offset - parts_header_start,
        )
        out.extend(part_name.encode("utf8", errors="ignore") + b"\x00")

    _pad_data(out, 64)
    skeleton_offset = len(out)
    struct.pack_into("<I", out, skeleton_offset_field, skeleton_offset)
    out.extend(
        _build_emo_skeleton_bytes(
            emo_file.skeleton,
            part_index_by_bone_name=part_index_by_bone_name,
        )
    )

    if pending_vertices:
        struct.pack_into("<I", out, vertices_offset_field, len(out))
        for patch in pending_vertices:
            vertex_offset_rel = len(out) - patch.mesh_offset
            struct.pack_into("<I", out, patch.field_offset, vertex_offset_rel)
            out.extend(patch.vertex_blob)
            _pad_data(out, 16)
    else:
        struct.pack_into("<I", out, vertices_offset_field, 0)

    return bytes(out)


def build_emo_bytes_from_emd_esk(
    emd: EMD_File,
    skeleton: ESK_File,
    *,
    version: int = 0x92C0,
    i_24: int = 0,
) -> bytes:
    parts, materials_count = convert_emd_to_emo_parts(emd, skeleton)
    emo_file = EMOFile(
        version=int(version),
        materials_count=int(materials_count),
        i_24=int(i_24),
        skeleton=skeleton,
        parts=parts,
        emd_file=emd,
    )
    return build_emo_bytes(emo_file)


__all__ = [
    "EMO_SIGNATURE",
    "EMOFile",
    "EMOPart",
    "build_emo_bytes",
    "build_emo_bytes_from_emd_esk",
    "convert_emo_to_emd",
    "convert_emd_to_emo_parts",
    "parse_emo",
    "parse_emo_bytes",
]
