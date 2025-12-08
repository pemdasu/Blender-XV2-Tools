import contextlib
import struct

from ...utils import half_to_float, read_cstring


class EMD_Vertex:
    def __init__(self):
        self.pos = (0.0, 0.0, 0.0)
        self.normal = (0.0, 0.0, 0.0)
        self.uv = (0.0, 0.0)
        self.uv2 = (0.0, 0.0)
        self.tangent = (0.0, 0.0, 0.0)
        self.color = (1.0, 1.0, 1.0, 1.0)
        self.bone_ids = [0, 0, 0, 0]
        self.bone_weights = [0.0, 0.0, 0.0, 0.0]


class EMD_Triangles:
    def __init__(self):
        self.indices: list[int] = []
        self.bone_names: list[str] = []
        self.bone_palette_lookup: dict[str, int] | None = None


class EMD_TextureSamplerDef:
    def __init__(self):
        self.flag0 = 0
        self.texture_index = 0
        self.address_mode_u = 0
        self.address_mode_v = 0
        self.filtering_min = 0
        self.filtering_mag = 0
        self.scale_u = 1.0
        self.scale_v = 1.0


class EMD_Submesh:
    def __init__(self):
        self.name = ""
        self.vertex_flags = 0
        self.vertices: list[EMD_Vertex] = []
        self.faces: list[tuple[int, int, int]] = []
        self.triangle_groups: list[EMD_Triangles] = []
        self.texture_sampler_defs: list[EMD_TextureSamplerDef] = []
        self.aabb_center = (0.0, 0.0, 0.0, 0.0)
        self.aabb_min = (0.0, 0.0, 0.0, 0.0)
        self.aabb_max = (0.0, 0.0, 0.0, 0.0)


class EMD_Mesh:
    def __init__(self):
        self.name = ""
        self.aabb_center = (0.0, 0.0, 0.0, 0.0)
        self.aabb_min = (0.0, 0.0, 0.0, 0.0)
        self.aabb_max = (0.0, 0.0, 0.0, 0.0)
        self.submeshes: list[EMD_Submesh] = []


class EMD_Model:
    def __init__(self):
        self.name = ""
        self.meshes: list[EMD_Mesh] = []


class EMD_File:
    def __init__(self):
        self.version = 0
        self.models: list[EMD_Model] = []


VERTEX_POSITION = 0x1
VERTEX_NORMAL = 0x2
VERTEX_TEXUV = 0x4
VERTEX_TEX2UV = 0x8
VERTEX_COLOR = 0x40
VERTEX_TANGENT = 0x80
VERTEX_BLENDWEIGHT = 0x200
VERTEX_COMPRESSED = 0x8000

ADDRESS_MODE_LABELS = {
    0: "Wrap",
    1: "Mirror",
    2: "Clamp",
}

FILTERING_LABELS = {
    0: "None",
    1: "Point",
    2: "Linear",
}

EMD_SIGNATURE = 1145914659


def get_vertex_size_from_flags(flags: int) -> int:
    size = 0
    is_comp = bool(flags & VERTEX_COMPRESSED)

    if flags & VERTEX_POSITION:
        size += 3 * 4
    if flags & VERTEX_NORMAL:
        size += (4 * 2) if is_comp else (3 * 4)
    if flags & VERTEX_TEXUV:
        size += 2 * (2 if is_comp else 4)
    if flags & VERTEX_TEX2UV:
        size += 2 * (2 if is_comp else 4)
    if flags & VERTEX_TANGENT:
        size += (4 * 2) if is_comp else (3 * 4)
    if flags & VERTEX_COLOR:
        size += 4
    if flags & VERTEX_BLENDWEIGHT:
        size += 4 + ((4 * 2) if is_comp else (3 * 4))

    return size


def read_texture_sampler_defs(data: bytes, offset: int, count: int) -> list[EMD_TextureSamplerDef]:
    sampler_defs: list[EMD_TextureSamplerDef] = []
    ptr = offset

    for _ in range(count):
        sampler = EMD_TextureSamplerDef()
        sampler.flag0 = data[ptr + 0]
        sampler.texture_index = data[ptr + 1]

        address_byte = data[ptr + 2]
        filtering_byte = data[ptr + 3]

        sampler.address_mode_u = address_byte & 0x0F
        sampler.address_mode_v = (address_byte >> 4) & 0x0F
        sampler.filtering_min = filtering_byte & 0x0F
        sampler.filtering_mag = (filtering_byte >> 4) & 0x0F

        sampler.scale_u = struct.unpack_from("<f", data, ptr + 4)[0]
        sampler.scale_v = struct.unpack_from("<f", data, ptr + 8)[0]

        sampler_defs.append(sampler)
        ptr += 12

    return sampler_defs


def sampler_def_to_prop_dict(sampler: EMD_TextureSamplerDef) -> dict:
    return {
        "flag0": int(sampler.flag0),
        "texture_index": int(sampler.texture_index),
        "address_mode_u": int(sampler.address_mode_u),
        "address_mode_v": int(sampler.address_mode_v),
        "address_mode_u_label": ADDRESS_MODE_LABELS.get(
            sampler.address_mode_u, f"Unknown_{sampler.address_mode_u}"
        ),
        "address_mode_v_label": ADDRESS_MODE_LABELS.get(
            sampler.address_mode_v, f"Unknown_{sampler.address_mode_v}"
        ),
        "filtering_min": int(sampler.filtering_min),
        "filtering_mag": int(sampler.filtering_mag),
        "filtering_min_label": FILTERING_LABELS.get(
            sampler.filtering_min, f"Unknown_{sampler.filtering_min}"
        ),
        "filtering_mag_label": FILTERING_LABELS.get(
            sampler.filtering_mag, f"Unknown_{sampler.filtering_mag}"
        ),
        "scale_u": float(sampler.scale_u),
        "scale_v": float(sampler.scale_v),
    }


def set_sampler_custom_properties(target, samplers: list[EMD_TextureSamplerDef]):
    legacy_prefixes = ("texture_sampler_def_", "emd_texture_sampler_def_")
    for key in list(target.keys()):
        if key.startswith(legacy_prefixes):
            with contextlib.suppress(Exception):
                del target[key]
    for root_key in ("texture_sampler_defs", "emd_texture_sampler_defs"):
        with contextlib.suppress(Exception):
            del target[root_key]

    sampler_dict = {}
    for sampler_index, sampler in enumerate(samplers):
        prefix = f"emd_texture_sampler_def_{sampler_index}_"
        target[prefix + "flag0"] = int(sampler.flag0)
        target[prefix + "texture_index"] = int(sampler.texture_index)
        target[prefix + "address_mode_u"] = int(sampler.address_mode_u)
        target[prefix + "address_mode_v"] = int(sampler.address_mode_v)
        target[prefix + "address_mode_u_label"] = ADDRESS_MODE_LABELS.get(
            sampler.address_mode_u, f"Unknown_{sampler.address_mode_u}"
        )
        target[prefix + "address_mode_v_label"] = ADDRESS_MODE_LABELS.get(
            sampler.address_mode_v, f"Unknown_{sampler.address_mode_v}"
        )
        target[prefix + "filtering_min"] = int(sampler.filtering_min)
        target[prefix + "filtering_mag"] = int(sampler.filtering_mag)
        target[prefix + "filtering_min_label"] = FILTERING_LABELS.get(
            sampler.filtering_min, f"Unknown_{sampler.filtering_min}"
        )
        target[prefix + "filtering_mag_label"] = FILTERING_LABELS.get(
            sampler.filtering_mag, f"Unknown_{sampler.filtering_mag}"
        )
        target[prefix + "scale_u"] = float(sampler.scale_u)
        target[prefix + "scale_v"] = float(sampler.scale_v)
        sampler_dict[str(sampler_index)] = sampler_def_to_prop_dict(sampler)

    target["emd_texture_sampler_defs"] = sampler_dict


def read_vertices(
    flags: int, data: bytes, offset: int, vertex_count: int, vertex_size: int
) -> list[EMD_Vertex]:
    vertices: list[EMD_Vertex] = []
    is_compressed = bool(flags & VERTEX_COMPRESSED)
    vertex_pointer = offset

    for _ in range(vertex_count):
        vertex = EMD_Vertex()
        bytes_read = 0

        if flags & VERTEX_POSITION:
            vertex.pos = struct.unpack_from("<3f", data, vertex_pointer + bytes_read)
            bytes_read += get_vertex_size_from_flags(VERTEX_POSITION)

        if flags & VERTEX_NORMAL:
            if is_compressed:
                nx = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 0)[0]
                )
                ny = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 2)[0]
                )
                nz = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 4)[0]
                )
                vertex.normal = (nx, ny, nz)
                bytes_read += get_vertex_size_from_flags(VERTEX_NORMAL | VERTEX_COMPRESSED)
            else:
                vertex.normal = struct.unpack_from("<3f", data, vertex_pointer + bytes_read)
                bytes_read += get_vertex_size_from_flags(VERTEX_NORMAL)

        if flags & VERTEX_TEXUV:
            if is_compressed:
                u = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 0)[0]
                )
                v = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 2)[0]
                )
                vertex.uv = (u, 1.0 - v)
                bytes_read += get_vertex_size_from_flags(VERTEX_TEXUV | VERTEX_COMPRESSED)
            else:
                u, v = struct.unpack_from("<2f", data, vertex_pointer + bytes_read)
                vertex.uv = (u, 1.0 - v)
                bytes_read += get_vertex_size_from_flags(VERTEX_TEXUV)

        if flags & VERTEX_TEX2UV:
            if is_compressed:
                u2 = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 0)[0]
                )
                v2 = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 2)[0]
                )
                vertex.uv2 = (u2, 1.0 - v2)
                bytes_read += get_vertex_size_from_flags(VERTEX_TEX2UV | VERTEX_COMPRESSED)
            else:
                u2, v2 = struct.unpack_from("<2f", data, vertex_pointer + bytes_read)
                vertex.uv2 = (u2, 1.0 - v2)
                bytes_read += get_vertex_size_from_flags(VERTEX_TEX2UV)

        if flags & VERTEX_TANGENT:
            if is_compressed:
                tx = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 0)[0]
                )
                ty = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 2)[0]
                )
                tz = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 4)[0]
                )
                vertex.tangent = (tx, ty, tz)
                bytes_read += get_vertex_size_from_flags(VERTEX_TANGENT | VERTEX_COMPRESSED)
            else:
                tx, ty, tz = struct.unpack_from("<3f", data, vertex_pointer + bytes_read)
                vertex.tangent = (tx, ty, tz)
                bytes_read += get_vertex_size_from_flags(VERTEX_TANGENT)

        if flags & VERTEX_COLOR:
            r, g, b, a = struct.unpack_from("<4B", data, vertex_pointer + bytes_read)
            vertex.color = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
            bytes_read += get_vertex_size_from_flags(VERTEX_COLOR)

        if flags & VERTEX_BLENDWEIGHT:
            bone_id0, bone_id1, bone_id2, bone_id3 = struct.unpack_from(
                "<4B", data, vertex_pointer + bytes_read
            )
            vertex.bone_ids = [bone_id0, bone_id1, bone_id2, bone_id3]

            if is_compressed:
                weight0 = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 4)[0]
                )
                weight1 = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 6)[0]
                )
                weight2 = half_to_float(
                    struct.unpack_from("<H", data, vertex_pointer + bytes_read + 8)[0]
                )
                bytes_read += get_vertex_size_from_flags(VERTEX_BLENDWEIGHT | VERTEX_COMPRESSED)
            else:
                weight0, weight1, weight2 = struct.unpack_from(
                    "<3f", data, vertex_pointer + bytes_read + 4
                )
                bytes_read += get_vertex_size_from_flags(VERTEX_BLENDWEIGHT)

            weight3 = 1.0 - (weight0 + weight1 + weight2)
            vertex.bone_weights = [weight0, weight1, weight2, weight3]

        if bytes_read != vertex_size:
            raise ValueError(f"VertexSize mismatch: expected {vertex_size}, got {bytes_read}")

        vertex_pointer += vertex_size
        vertices.append(vertex)

    return vertices


def parse_emd(path: str) -> EMD_File:
    with open(path, "rb") as file_handle:
        data = file_handle.read()

    if struct.unpack_from("<I", data, 0)[0] != EMD_SIGNATURE:
        raise ValueError("EMD_SIGNATURE not found at 0x0")

    emd = EMD_File()
    emd.version = struct.unpack_from("<I", data, 8)[0]

    model_table_count = struct.unpack_from("<H", data, 18)[0]
    model_table_offset = struct.unpack_from("<I", data, 20)[0]
    model_name_table_offset = struct.unpack_from("<I", data, 24)[0]

    model_ptr = model_table_offset
    name_ptr = model_name_table_offset

    for _ in range(model_table_count):
        model_off_rel = struct.unpack_from("<I", data, model_ptr)[0]
        name_off = struct.unpack_from("<I", data, name_ptr)[0]

        if model_off_rel != 0:
            model_off = model_off_rel
            model = EMD_Model()

            if name_off != 0:
                model.name = read_cstring(data, name_off)
            else:
                model.name = ""

            mesh_count = struct.unpack_from("<H", data, model_off + 2)[0]
            mesh_table_offset = model_off + struct.unpack_from("<I", data, model_off + 4)[0]
            mesh_ptr = mesh_table_offset

            for _m in range(mesh_count):
                mesh_off_rel = struct.unpack_from("<I", data, mesh_ptr)[0]
                mesh_ptr += 4
                mesh_off = model_off + mesh_off_rel

                mesh = EMD_Mesh()

                name_rel = struct.unpack_from("<I", data, mesh_off + 48)[0]
                if name_rel != 0:
                    mesh.name = read_cstring(data, mesh_off + name_rel)
                else:
                    mesh.name = ""

                submesh_count = struct.unpack_from("<H", data, mesh_off + 54)[0]
                submesh_table_offset = mesh_off + struct.unpack_from("<I", data, mesh_off + 56)[0]
                sub_ptr = submesh_table_offset

                for _s in range(submesh_count):
                    sub_off_rel = struct.unpack_from("<I", data, sub_ptr)[0]
                    sub_ptr += 4
                    sub_off = mesh_off + sub_off_rel

                    sub = EMD_Submesh()

                    sub.vertex_flags = struct.unpack_from("<I", data, sub_off + 48)[0]
                    vertex_size = struct.unpack_from("<I", data, sub_off + 52)[0]
                    vertex_count = struct.unpack_from("<I", data, sub_off + 56)[0]
                    vertex_rel = struct.unpack_from("<I", data, sub_off + 60)[0]
                    vertex_off = sub_off + vertex_rel

                    sub.vertices = read_vertices(
                        sub.vertex_flags, data, vertex_off, vertex_count, vertex_size
                    )

                    sub_name_rel = struct.unpack_from("<I", data, sub_off + 64)[0]
                    if sub_name_rel != 0:
                        sub.name = read_cstring(data, sub_off + sub_name_rel)
                    else:
                        sub.name = ""

                    texture_definition_count = data[sub_off + 69]
                    texture_definition_rel = struct.unpack_from("<I", data, sub_off + 72)[0]
                    if texture_definition_count > 0 and texture_definition_rel != 0:
                        texture_definition_off = sub_off + texture_definition_rel
                        sub.texture_sampler_defs = read_texture_sampler_defs(
                            data,
                            texture_definition_off,
                            texture_definition_count,
                        )

                    triangle_count = struct.unpack_from("<H", data, sub_off + 70)[0]
                    triangles_table_offset = (
                        sub_off + struct.unpack_from("<I", data, sub_off + 76)[0]
                    )
                    tri_ptr = triangles_table_offset

                    for _t in range(triangle_count):
                        tri_rel = struct.unpack_from("<I", data, tri_ptr)[0]
                        tri_ptr += 4
                        tri_off = sub_off + tri_rel

                        tri = EMD_Triangles()

                        face_count = struct.unpack_from("<I", data, tri_off + 0)[0]
                        bone_name_count = struct.unpack_from("<I", data, tri_off + 4)[0]
                        face_table_rel = struct.unpack_from("<I", data, tri_off + 8)[0]
                        bone_name_table_rel = struct.unpack_from("<I", data, tri_off + 12)[0]

                        face_ptr = tri_off + face_table_rel if face_table_rel != 0 else tri_off + 16

                        is32 = face_count > 65535
                        indices: list[int] = []

                        if is32:
                            for _f in range(face_count):
                                idx = struct.unpack_from("<I", data, face_ptr)[0]
                                face_ptr += 4
                                indices.append(idx)
                        else:
                            for _f in range(face_count):
                                idx = struct.unpack_from("<H", data, face_ptr)[0]
                                face_ptr += 2
                                indices.append(idx)

                        tri.indices = indices

                        for face_start in range(0, len(indices), 3):
                            if face_start + 2 < len(indices):
                                sub.faces.append(
                                    (
                                        indices[face_start],
                                        indices[face_start + 1],
                                        indices[face_start + 2],
                                    )
                                )

                        bone_names: list[str] = []
                        if bone_name_count > 0 and bone_name_table_rel != 0:
                            bone_name_table_off = tri_off + bone_name_table_rel
                            for bi in range(bone_name_count):
                                name_rel = struct.unpack_from(
                                    "<I", data, bone_name_table_off + 4 * bi
                                )[0]
                                if name_rel != 0:
                                    name_off = tri_off + name_rel
                                    bone_names.append(read_cstring(data, name_off))
                                else:
                                    bone_names.append("")
                        tri.bone_names = bone_names

                        sub.triangle_groups.append(tri)

                    mesh.submeshes.append(sub)

                model.meshes.append(mesh)

            emd.models.append(model)

        model_ptr += 4
        name_ptr += 4

    return emd
