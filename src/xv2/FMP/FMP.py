from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import mathutils

from ...utils import read_cstring
from ...utils.binary import f32, i32, is_valid_offset, u16

FMP_SIGNATURE = 1347241507


@dataclass(slots=True)
class FMPTransform:
    matrix: mathutils.Matrix = field(default_factory=lambda: mathutils.Matrix.Identity(4))


@dataclass(slots=True)
class FMPLOD:
    distance: float = 0.0
    nsk_file: str = ""
    emm_file: str = ""


@dataclass(slots=True)
class FMPVisual:
    name: str = ""
    i_04: int = 0
    lods: list[FMPLOD] = field(default_factory=list)
    emb_file: str = ""
    ema_file: str = ""
    i_24: int = -1
    i_28: int = -1
    i_36: int = -1
    f_40: float = 60.0
    f_44: float = 60.0


@dataclass(slots=True)
class FMPEntity:
    i_04: int = 0
    visual: FMPVisual | None = None
    transform: FMPTransform = field(default_factory=FMPTransform)


@dataclass(slots=True)
class FMPInstanceTransform:
    matrix: mathutils.Matrix = field(default_factory=lambda: mathutils.Matrix.Identity(4))
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(slots=True)
class FMPInstanceGroup:
    index: int = 0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_distance: float = 0.0
    min_bounds: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_bounds: tuple[float, float, float] = (0.0, 0.0, 0.0)
    transforms: list[FMPInstanceTransform] = field(default_factory=list)


@dataclass(slots=True)
class FMPInstanceBVHNode:
    index: int = 0
    flags: int = 0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    group_indices: list[int] = field(default_factory=list)
    children: list[FMPInstanceBVHNode] = field(default_factory=list)


@dataclass(slots=True)
class FMPInstanceData:
    groups: list[FMPInstanceGroup] = field(default_factory=list)
    bvh_root: FMPInstanceBVHNode | None = None


@dataclass(slots=True)
class FMPHavokGroupParameters:
    param1: int = 0
    param2: int = 0


@dataclass(slots=True)
class FMPObjectSubPart:
    i_00: int = 0
    i_02: int = 0
    i_04: int = 0
    i_06: int = 0
    i_08: int = 0
    mass: float = 0.0
    f_16: float = 0.0
    f_20: float = 0.0
    f_24: float = 0.0
    f_28: float = 0.0
    f_32: float = 0.0
    f_36: float = 0.0
    f_40: float = 0.0
    f_44: float = 0.0
    f_48: float = 0.0
    f_52: float = 0.0
    width: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(slots=True)
class FMPColliderInstance:
    i_20: int = 0
    i_22: int = 0xFFFF
    f_24: float = 0.0
    f_28: float = 0.0
    matrix: FMPTransform = field(default_factory=FMPTransform)
    havok_group_parameters: list[FMPHavokGroupParameters] = field(default_factory=list)
    subpart1: FMPObjectSubPart | None = None
    subpart2: FMPObjectSubPart | None = None
    action_offset: int = 0


@dataclass(slots=True)
class FMPCollisionVertex:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class FMPCollisionVertexData:
    faces: list[int] = field(default_factory=list)
    vertices: list[FMPCollisionVertex] = field(default_factory=list)


@dataclass(slots=True)
class FMPHvkCollisionData:
    i_00: int = 0
    hvk_file: bytes = b""


@dataclass(slots=True)
class FMPHavokEntry:
    group: int = 0
    fragment_group: int = 0


@dataclass(slots=True)
class FMPCollider:
    index: int = 0
    name: str = ""
    child_idx: int = 0xFFFF
    unk_a0: int = 0
    sibling_idx: int = 0xFFFF
    parent_idx: int = 0xFFFF
    havok_entries: list[FMPHavokEntry] = field(default_factory=list)
    collision_vertex_data: FMPCollisionVertexData | None = None
    hvk_collision_data: FMPHvkCollisionData | None = None


@dataclass(slots=True)
class FMPCollisionGroup:
    index: int = 0
    name: str = ""
    collider_count: int = 0
    collider_offset: int = 0
    colliders: list[FMPCollider] = field(default_factory=list)


@dataclass(slots=True)
class FMPObject:
    idx: int = -1
    name: str = ""
    i_04: int = 0
    initial_entity_index: int = 0
    hitbox_group_index: int = 65535
    hitbox_instances_offset: int = 0
    action_offset: int = 0
    flags: int = 0
    f_32: float = 0.0
    entities: list[FMPEntity] = field(default_factory=list)
    transform: FMPTransform = field(default_factory=FMPTransform)
    instance_data: FMPInstanceData | None = None
    collider_instances: list[FMPColliderInstance] = field(default_factory=list)


@dataclass(slots=True)
class FMPFile:
    version: int = 0
    i_08: int = 18
    i_12: int = 0
    i_96: tuple[int, int, int, int] = (0, 0, 0, 0)
    objects: list[FMPObject] = field(default_factory=list)
    collision_groups: list[FMPCollisionGroup] = field(default_factory=list)


AXIS_XV2_TO_BLENDER = mathutils.Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)
AXIS_BLENDER_TO_XV2 = AXIS_XV2_TO_BLENDER.inverted()


def map_name_from_path(path: str) -> str:
    return Path(path).stem


def sanitize_name(value: str, fallback: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return fallback
    return cleaned.replace("\\", "_").replace("/", "_")


def to_blender_axis(matrix: mathutils.Matrix) -> mathutils.Matrix:
    return AXIS_XV2_TO_BLENDER @ matrix @ AXIS_BLENDER_TO_XV2


def to_xv2_axis(matrix: mathutils.Matrix) -> mathutils.Matrix:
    return AXIS_BLENDER_TO_XV2 @ matrix @ AXIS_XV2_TO_BLENDER


def pick_entity_lod(entity: FMPEntity) -> tuple[int, FMPLOD] | None:
    if entity.visual is None:
        return None
    for lod_index, lod in enumerate(entity.visual.lods):
        if lod.nsk_file:
            return lod_index, lod
    return None


def resolve_nsk_path(base_dir: Path, nsk_value: str) -> Path:
    rel_path = nsk_value.replace("\\", os.sep).replace("/", os.sep)
    candidate = (base_dir / rel_path).resolve()
    if candidate.exists():
        return candidate
    return base_dir / rel_path


def resolve_optional_asset_path(base_dir: Path, asset_value: str) -> Path | None:
    raw_value = (asset_value or "").strip()
    if not raw_value:
        return None

    rel_path = raw_value.replace("\\", os.sep).replace("/", os.sep)
    candidate = (base_dir / rel_path).resolve()
    if candidate.exists():
        return candidate

    fallback = base_dir / rel_path
    return fallback if fallback.exists() else None


def iter_object_instance_matrices(object_data: FMPObject) -> list[mathutils.Matrix]:
    instance_data = getattr(object_data, "instance_data", None)
    if instance_data is None:
        return [object_data.transform.matrix]

    instance_matrices: list[mathutils.Matrix] = []
    for group in getattr(instance_data, "groups", []):
        for instance_transform in getattr(group, "transforms", []):
            instance_matrices.append(object_data.transform.matrix @ instance_transform.matrix)

    return instance_matrices if instance_matrices else [object_data.transform.matrix]


def to_json_string(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def serialize_instance_bvh(node: FMPInstanceBVHNode | None) -> dict | None:
    if node is None:
        return None
    return {
        "index": int(node.index),
        "flags": int(node.flags),
        "center": [float(node.center[0]), float(node.center[1]), float(node.center[2])],
        "group_indices": [int(group_index) for group_index in node.group_indices],
        "children": [
            child_json
            for child_json in (serialize_instance_bvh(child) for child in node.children)
            if child_json is not None
        ],
    }


def serialize_instance_data(instance_data: FMPInstanceData | None) -> dict | None:
    if instance_data is None:
        return None

    groups: list[dict] = []
    for group in instance_data.groups:
        groups.append(
            {
                "index": int(group.index),
                "center": [
                    float(group.center[0]),
                    float(group.center[1]),
                    float(group.center[2]),
                ],
                "max_distance": float(group.max_distance),
                "min_bounds": [
                    float(group.min_bounds[0]),
                    float(group.min_bounds[1]),
                    float(group.min_bounds[2]),
                ],
                "max_bounds": [
                    float(group.max_bounds[0]),
                    float(group.max_bounds[1]),
                    float(group.max_bounds[2]),
                ],
                "transforms": [
                    {
                        "position": [
                            float(transform.position[0]),
                            float(transform.position[1]),
                            float(transform.position[2]),
                        ],
                        "rotation": [
                            float(transform.rotation[0]),
                            float(transform.rotation[1]),
                            float(transform.rotation[2]),
                        ],
                        "scale": [
                            float(transform.scale[0]),
                            float(transform.scale[1]),
                            float(transform.scale[2]),
                        ],
                    }
                    for transform in group.transforms
                ],
            }
        )

    return {
        "groups": groups,
        "bvh_root": serialize_instance_bvh(instance_data.bvh_root),
    }


def serialize_visual_lods(entity: FMPEntity) -> list[dict]:
    if entity.visual is None:
        return []
    return [
        {
            "index": int(lod_index),
            "distance": float(lod.distance),
            "nsk_file": str(lod.nsk_file),
            "emm_file": str(lod.emm_file),
        }
        for lod_index, lod in enumerate(entity.visual.lods)
    ]


def normalize_cache_path(path_value: str) -> str:
    if not path_value:
        return ""
    return os.path.normcase(os.path.normpath(os.path.abspath(path_value)))


def _read_depot_strings(data: bytes, depot_offset: int, depot_count: int) -> list[str]:
    if depot_count <= 0 or not is_valid_offset(data, depot_offset, depot_count * 4):
        return []

    values: list[str] = []
    for index in range(depot_count):
        value_offset = i32(data, depot_offset + (index * 4))
        if is_valid_offset(data, value_offset):
            values.append(read_cstring(data, value_offset))
        else:
            values.append("")
    return values


def _depot_value(values: list[str], index: int) -> str:
    if 0 <= index < len(values):
        return values[index]
    return ""


def _read_transform(data: bytes, offset: int) -> FMPTransform:
    if not is_valid_offset(data, offset, 48):
        return FMPTransform()

    values = struct.unpack_from("<12f", data, offset)
    row_matrix = mathutils.Matrix(
        (
            (values[0], values[1], values[2], 0.0),
            (values[3], values[4], values[5], 0.0),
            (values[6], values[7], values[8], 0.0),
            (values[9], values[10], values[11], 1.0),
        )
    )

    return FMPTransform(matrix=row_matrix.transposed())


def _compose_instance_transform_matrix(
    px: float,
    py: float,
    pz: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    sx: float,
    sy: float,
    sz: float,
) -> mathutils.Matrix:
    scale_row = mathutils.Matrix(
        (
            (sx, 0.0, 0.0, 0.0),
            (0.0, sy, 0.0, 0.0),
            (0.0, 0.0, sz, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    rotation_col = (
        mathutils.Euler(
            (
                math.radians(rx_deg),
                math.radians(ry_deg),
                math.radians(rz_deg),
            ),
            "XYZ",
        )
        .to_matrix()
        .to_4x4()
    )
    rotation_row = rotation_col.transposed()

    translation_row = mathutils.Matrix.Identity(4)
    translation_row[3][0] = px
    translation_row[3][1] = py
    translation_row[3][2] = pz

    return (scale_row @ rotation_row @ translation_row).transposed()


def _read_instance_transform(data: bytes, offset: int) -> FMPInstanceTransform:
    if not is_valid_offset(data, offset, 36):
        return FMPInstanceTransform()

    px = f32(data, offset + 0)
    py = f32(data, offset + 4)
    pz = f32(data, offset + 8)
    rx = math.degrees(f32(data, offset + 12))
    ry = math.degrees(f32(data, offset + 16))
    rz = math.degrees(f32(data, offset + 20))
    sx = f32(data, offset + 24)
    sy = f32(data, offset + 28)
    sz = f32(data, offset + 32)

    return FMPInstanceTransform(
        matrix=_compose_instance_transform_matrix(px, py, pz, rx, ry, rz, sx, sy, sz),
        position=(px, py, pz),
        rotation=(rx, ry, rz),
        scale=(sx, sy, sz),
    )


def _read_instance_group(
    data: bytes,
    offset: int,
    index: int,
    node_transform_offset: int,
    node_transform_count: int,
) -> FMPInstanceGroup:
    group = FMPInstanceGroup(index=index)
    if not is_valid_offset(data, offset, 48):
        return group

    group.center = (
        f32(data, offset + 0),
        f32(data, offset + 4),
        f32(data, offset + 8),
    )
    group.max_distance = f32(data, offset + 12)
    group.min_bounds = (
        f32(data, offset + 16),
        f32(data, offset + 20),
        f32(data, offset + 24),
    )
    group.max_bounds = (
        f32(data, offset + 28),
        f32(data, offset + 32),
        f32(data, offset + 36),
    )

    transform_count = i32(data, offset + 40)
    node_transform_index = i32(data, offset + 44)

    if (
        transform_count <= 0
        or node_transform_index < 0
        or node_transform_count <= 0
        or node_transform_offset <= 0
    ):
        return group

    remaining = max(0, node_transform_count - node_transform_index)
    read_count = min(transform_count, remaining)
    first_transform_offset = node_transform_offset + (node_transform_index * 36)

    if read_count <= 0 or not is_valid_offset(data, first_transform_offset, read_count * 36):
        return group

    for transform_index in range(read_count):
        transform_offset = first_transform_offset + (transform_index * 36)
        group.transforms.append(_read_instance_transform(data, transform_offset))

    return group


def _read_instance_bvh_node(
    data: bytes,
    offset: int,
    index: int = 0,
    depth: int = 0,
) -> FMPInstanceBVHNode | None:
    if depth > 64 or not is_valid_offset(data, offset, 4):
        return None

    is_leaf = data[offset]
    flags = data[offset + 1]
    indices_count = u16(data, offset + 2)

    node = FMPInstanceBVHNode(index=index, flags=int(flags))
    if is_leaf == 0:
        if not is_valid_offset(data, offset, 48):
            return node
        node.center = (
            f32(data, offset + 4),
            f32(data, offset + 8),
            f32(data, offset + 12),
        )
        for child_index in range(8):
            child_offset = i32(data, offset + 16 + (child_index * 4))
            if child_offset <= 0 or child_offset == offset:
                continue
            child_node = _read_instance_bvh_node(
                data,
                child_offset,
                index=child_index,
                depth=depth + 1,
            )
            if child_node is not None:
                node.children.append(child_node)
    elif is_leaf == 1:
        if is_valid_offset(data, offset + 4, indices_count * 2):
            for idx in range(indices_count):
                node.group_indices.append(int(u16(data, offset + 4 + (idx * 2))))
    return node


def _read_instance_data(data: bytes, offset: int) -> FMPInstanceData | None:
    if not is_valid_offset(data, offset, 20):
        return None

    node_transform_count = i32(data, offset + 0)
    node_count = i32(data, offset + 4)
    node_offset = i32(data, offset + 8)
    hierarchy_node_offset = i32(data, offset + 12)
    node_transform_offset = i32(data, offset + 16)

    if node_count <= 0 and hierarchy_node_offset <= 0:
        return None

    instance_data = FMPInstanceData()

    if node_count > 0 and is_valid_offset(data, node_offset, node_count * 48):
        for node_index in range(node_count):
            group_offset = node_offset + (node_index * 48)
            instance_data.groups.append(
                _read_instance_group(
                    data,
                    group_offset,
                    node_index,
                    node_transform_offset,
                    node_transform_count,
                )
            )

    if hierarchy_node_offset > 0 and is_valid_offset(data, hierarchy_node_offset, 4):
        instance_data.bvh_root = _read_instance_bvh_node(data, hierarchy_node_offset, index=0)

    return instance_data if (instance_data.groups or instance_data.bvh_root is not None) else None


def _read_havok_group_parameters(
    data: bytes,
    offset: int,
    count: int,
) -> list[FMPHavokGroupParameters]:
    if count <= 0 or offset <= 0 or not is_valid_offset(data, offset, count * 8):
        return []

    values: list[FMPHavokGroupParameters] = []
    for index in range(count):
        current_offset = offset + (index * 8)
        values.append(
            FMPHavokGroupParameters(
                param1=i32(data, current_offset + 0),
                param2=i32(data, current_offset + 4),
            )
        )
    return values


def _read_object_subpart(data: bytes, offset: int) -> FMPObjectSubPart | None:
    if offset <= 0 or not is_valid_offset(data, offset, 84):
        return None

    return FMPObjectSubPart(
        i_00=u16(data, offset + 0),
        i_02=u16(data, offset + 2),
        i_04=u16(data, offset + 4),
        i_06=u16(data, offset + 6),
        i_08=i32(data, offset + 8),
        mass=f32(data, offset + 12),
        f_16=f32(data, offset + 16),
        f_20=f32(data, offset + 20),
        f_24=f32(data, offset + 24),
        f_28=f32(data, offset + 28),
        f_32=f32(data, offset + 32),
        f_36=f32(data, offset + 36),
        f_40=f32(data, offset + 40),
        f_44=f32(data, offset + 44),
        f_48=f32(data, offset + 48),
        f_52=f32(data, offset + 52),
        width=(
            f32(data, offset + 56),
            f32(data, offset + 60),
            f32(data, offset + 64),
        ),
        quaternion=(
            f32(data, offset + 68),
            f32(data, offset + 72),
            f32(data, offset + 76),
            f32(data, offset + 80),
        ),
    )


def _read_collider_instance(data: bytes, offset: int) -> FMPColliderInstance:
    instance = FMPColliderInstance()
    if offset <= 0 or not is_valid_offset(data, offset, 80):
        return instance

    index_pair_count = i32(data, offset + 0)
    index_pair_offset = i32(data, offset + 4)
    subpart1_offset = i32(data, offset + 8)
    subpart2_offset = i32(data, offset + 12)
    action_offset = i32(data, offset + 16)

    instance.i_20 = u16(data, offset + 20)
    instance.i_22 = u16(data, offset + 22)
    instance.f_24 = f32(data, offset + 24)
    instance.f_28 = f32(data, offset + 28)
    instance.matrix = _read_transform(data, offset + 32)
    instance.havok_group_parameters = _read_havok_group_parameters(
        data,
        index_pair_offset,
        index_pair_count,
    )
    instance.subpart1 = _read_object_subpart(data, subpart1_offset)
    instance.subpart2 = _read_object_subpart(data, subpart2_offset)
    instance.action_offset = action_offset
    return instance


def _read_collider_instances(
    data: bytes,
    offset: int,
    count: int,
) -> list[FMPColliderInstance]:
    if count <= 0 or offset <= 0 or not is_valid_offset(data, offset, count * 80):
        return []

    return [_read_collider_instance(data, offset + (index * 80)) for index in range(count)]


def _read_collision_vertex_data(
    data: bytes,
    vertex_offset: int,
    vertex_count: int,
    face_indices_offset: int,
    face_indices_count: int,
) -> FMPCollisionVertexData | None:
    if vertex_count <= 0 or face_indices_count <= 0:
        return None
    if vertex_offset <= 0 or face_indices_offset <= 0:
        return None
    if not is_valid_offset(data, vertex_offset, vertex_count * 24):
        return None
    if not is_valid_offset(data, face_indices_offset, face_indices_count * 2):
        return None

    vertex_data = FMPCollisionVertexData()
    for face_index in range(face_indices_count):
        vertex_data.faces.append(int(u16(data, face_indices_offset + (face_index * 2))))

    for vertex_index in range(vertex_count):
        current_offset = vertex_offset + (vertex_index * 24)
        vertex_data.vertices.append(
            FMPCollisionVertex(
                position=(
                    f32(data, current_offset + 0),
                    f32(data, current_offset + 4),
                    f32(data, current_offset + 8),
                ),
                normal=(
                    f32(data, current_offset + 12),
                    f32(data, current_offset + 16),
                    f32(data, current_offset + 20),
                ),
            )
        )

    return vertex_data


def _read_hvk_collision_data(data: bytes, offset: int) -> FMPHvkCollisionData | None:
    if offset <= 0 or not is_valid_offset(data, offset, 12):
        return None

    hvk_size = i32(data, offset + 4)
    hvk_offset = i32(data, offset + 8)
    hvk_file = b""
    if hvk_size > 0 and hvk_offset > 0 and is_valid_offset(data, hvk_offset, hvk_size):
        hvk_file = data[hvk_offset : hvk_offset + hvk_size]

    return FMPHvkCollisionData(
        i_00=i32(data, offset + 0),
        hvk_file=hvk_file,
    )


def _read_havok_entries(
    data: bytes,
    list_offset: int,
    list_count: int,
) -> list[FMPHavokEntry]:
    if (
        list_count <= 0
        or list_offset <= 0
        or not is_valid_offset(data, list_offset, list_count * 8)
    ):
        return []

    entries: list[FMPHavokEntry] = []
    for group_index in range(list_count):
        group_header_offset = list_offset + (group_index * 8)
        group_entry_count = i32(data, group_header_offset + 0)
        group_entries_offset = i32(data, group_header_offset + 4)
        if (
            group_entry_count <= 0
            or group_entries_offset <= 0
            or not is_valid_offset(data, group_entries_offset, group_entry_count * 40)
        ):
            continue

        for entry_index in range(group_entry_count):
            entry_offset = group_entries_offset + (entry_index * 40)
            entries.append(
                FMPHavokEntry(
                    group=group_index,
                    fragment_group=i32(data, entry_offset + 0),
                )
            )

    return entries


def _read_collision_group_collider(
    data: bytes,
    offset: int,
    index: int,
    old_version: bool,
) -> FMPCollider:
    collider = FMPCollider(index=index)
    if not is_valid_offset(data, offset, 40):
        return collider

    name_offset = i32(data, offset + 0)
    if name_offset != -1 and is_valid_offset(data, name_offset):
        collider.name = read_cstring(data, name_offset)
    collider.child_idx = int(u16(data, offset + 4))
    collider.unk_a0 = int(u16(data, offset + 6))
    collider.sibling_idx = int(u16(data, offset + 8))
    collider.parent_idx = int(u16(data, offset + 10))
    havok_list_count = i32(data, offset + 12)
    havok_list_offset = i32(data, offset + 16)
    collider.havok_entries = _read_havok_entries(
        data,
        havok_list_offset,
        max(0, havok_list_count),
    )

    hvk_geometry_offset = 0
    vertex_data_count = 0
    vertex_data_offset = 0
    face_indices_count = 0
    face_indices_offset = 0

    if old_version:
        vertex_data_count = i32(data, offset + 20)
        vertex_data_offset = i32(data, offset + 24)
        face_indices_count = i32(data, offset + 28)
        face_indices_offset = i32(data, offset + 32)
    else:
        hvk_geometry_offset = i32(data, offset + 20)
        vertex_data_count = i32(data, offset + 24)
        vertex_data_offset = i32(data, offset + 28)
        face_indices_count = i32(data, offset + 32)
        face_indices_offset = i32(data, offset + 36)

    collider.collision_vertex_data = _read_collision_vertex_data(
        data,
        vertex_data_offset,
        max(0, vertex_data_count),
        face_indices_offset,
        max(0, face_indices_count),
    )
    if not old_version and hvk_geometry_offset > 0:
        collider.hvk_collision_data = _read_hvk_collision_data(data, hvk_geometry_offset)

    return collider


def _read_collision_groups(
    data: bytes,
    offset: int,
    count: int,
    old_version: bool,
) -> list[FMPCollisionGroup]:
    if count <= 0 or offset <= 0 or not is_valid_offset(data, offset, count * 12):
        return []

    groups: list[FMPCollisionGroup] = []
    for index in range(count):
        current_offset = offset + (index * 12)
        name_offset = i32(data, current_offset + 0)
        name = read_cstring(data, name_offset) if is_valid_offset(data, name_offset) else ""
        groups.append(
            FMPCollisionGroup(
                index=index,
                name=name,
                collider_count=max(i32(data, current_offset + 4), 0),
                collider_offset=i32(data, current_offset + 8),
            )
        )

    for group in groups:
        if group.collider_count <= 0 or group.collider_offset <= 0:
            continue
        if not is_valid_offset(data, group.collider_offset, group.collider_count * 40):
            continue
        for collider_index in range(group.collider_count):
            collider_offset = group.collider_offset + (collider_index * 40)
            group.colliders.append(
                _read_collision_group_collider(
                    data,
                    collider_offset,
                    collider_index,
                    old_version,
                )
            )

    return groups


def _read_visual(
    data: bytes,
    visual_offset: int,
    depot1_nsk: list[str],
    depot2_emb: list[str],
    depot3_emm: list[str],
    depot4_ema: list[str],
) -> FMPVisual:
    visual = FMPVisual()
    if not is_valid_offset(data, visual_offset, 52):
        return visual

    name_offset = i32(data, visual_offset)
    if is_valid_offset(data, name_offset):
        visual.name = read_cstring(data, name_offset)

    visual.i_04 = i32(data, visual_offset + 4)
    emb_index = i32(data, visual_offset + 16)
    ema_index = i32(data, visual_offset + 32)
    visual.emb_file = _depot_value(depot2_emb, emb_index)
    visual.ema_file = _depot_value(depot4_ema, ema_index)
    visual.i_24 = i32(data, visual_offset + 24)
    visual.i_28 = i32(data, visual_offset + 28)
    visual.i_36 = i32(data, visual_offset + 36)
    visual.f_40 = f32(data, visual_offset + 40)
    visual.f_44 = f32(data, visual_offset + 44)

    lod_count = max(i32(data, visual_offset + 8), 0)
    if lod_count <= 1:
        nsk_index = i32(data, visual_offset + 12)
        emm_index = i32(data, visual_offset + 20)
        distance = f32(data, visual_offset + 48)
        visual.lods.append(
            FMPLOD(
                distance=distance,
                nsk_file=_depot_value(depot1_nsk, nsk_index),
                emm_file=_depot_value(depot3_emm, emm_index),
            )
        )
        return visual

    nsk_table_offset = i32(data, visual_offset + 12)
    emm_table_offset = i32(data, visual_offset + 20)
    distance_table_offset = i32(data, visual_offset + 48)

    if not is_valid_offset(data, nsk_table_offset, lod_count * 4):
        nsk_table_offset = 0
    if not is_valid_offset(data, emm_table_offset, lod_count * 4):
        emm_table_offset = 0
    if not is_valid_offset(data, distance_table_offset, lod_count * 4):
        distance_table_offset = 0

    for lod_index in range(lod_count):
        nsk_index = i32(data, nsk_table_offset + (lod_index * 4)) if nsk_table_offset else -1
        emm_index = i32(data, emm_table_offset + (lod_index * 4)) if emm_table_offset else -1
        distance = (
            f32(data, distance_table_offset + (lod_index * 4)) if distance_table_offset else 0.0
        )
        visual.lods.append(
            FMPLOD(
                distance=distance,
                nsk_file=_depot_value(depot1_nsk, nsk_index),
                emm_file=_depot_value(depot3_emm, emm_index),
            )
        )

    return visual


def _read_entity(
    data: bytes,
    entity_offset: int,
    depot1_nsk: list[str],
    depot2_emb: list[str],
    depot3_emm: list[str],
    depot4_ema: list[str],
) -> FMPEntity:
    entity = FMPEntity()
    if not is_valid_offset(data, entity_offset, 56):
        return entity

    visual_offset = i32(data, entity_offset)
    entity.i_04 = i32(data, entity_offset + 4)
    entity.transform = _read_transform(data, entity_offset + 8)

    if is_valid_offset(data, visual_offset, 52):
        entity.visual = _read_visual(
            data,
            visual_offset,
            depot1_nsk,
            depot2_emb,
            depot3_emm,
            depot4_ema,
        )

    return entity


def _read_object(
    data: bytes,
    object_index: int,
    object_offset: int,
    depot1_nsk: list[str],
    depot2_emb: list[str],
    depot3_emm: list[str],
    depot4_ema: list[str],
    collision_groups: list[FMPCollisionGroup],
) -> FMPObject:
    obj = FMPObject(idx=object_index)
    if not is_valid_offset(data, object_offset, 84):
        return obj

    name_offset = i32(data, object_offset)
    if is_valid_offset(data, name_offset):
        obj.name = read_cstring(data, name_offset)

    obj.i_04 = u16(data, object_offset + 4)
    obj.initial_entity_index = u16(data, object_offset + 6)
    obj.hitbox_group_index = u16(data, object_offset + 8)
    obj.hitbox_instances_offset = i32(data, object_offset + 12)
    obj.action_offset = i32(data, object_offset + 16)
    obj.flags = u16(data, object_offset + 10)
    obj.f_32 = f32(data, object_offset + 32)
    obj.transform = _read_transform(data, object_offset + 36)

    entity_count = i32(data, object_offset + 20)
    entity_offset = i32(data, object_offset + 24)
    hierarchy_offset = i32(data, object_offset + 28)

    if entity_count > 0 and is_valid_offset(data, entity_offset, entity_count * 56):
        for entity_index in range(entity_count):
            current_offset = entity_offset + (entity_index * 56)
            obj.entities.append(
                _read_entity(
                    data,
                    current_offset,
                    depot1_nsk,
                    depot2_emb,
                    depot3_emm,
                    depot4_ema,
                )
            )

    if hierarchy_offset > 0:
        obj.instance_data = _read_instance_data(data, hierarchy_offset)

    if (
        obj.hitbox_group_index != 0xFFFF
        and 0 <= obj.hitbox_group_index < len(collision_groups)
        and obj.hitbox_instances_offset > 0
    ):
        collider_count = collision_groups[obj.hitbox_group_index].collider_count
        obj.collider_instances = _read_collider_instances(
            data,
            obj.hitbox_instances_offset,
            collider_count,
        )

    return obj


def parse_fmp_bytes(data: bytes) -> FMPFile:
    if len(data) < 112:
        raise ValueError("FMP file is too small.")

    if i32(data, 0) != FMP_SIGNATURE:
        raise ValueError("Invalid FMP signature. This is not a .map file.")

    version = i32(data, 4)
    is_old_version = (version & 0xFFF00) == 0
    i_08 = i32(data, 8)
    i_12 = i32(data, 12)
    i_96 = (
        i32(data, 96),
        i32(data, 100),
        i32(data, 104),
        i32(data, 108),
    )
    object_count = i32(data, 48)
    object_offset = i32(data, 52)
    collision_group_count = i32(data, 56)
    collision_group_offset = i32(data, 60)

    depot1_count = i32(data, 64)
    depot1_offset = i32(data, 68)
    depot2_count = i32(data, 72)
    depot2_offset = i32(data, 76)
    depot3_count = i32(data, 80)
    depot3_offset = i32(data, 84)
    depot4_count = i32(data, 88)
    depot4_offset = i32(data, 92)

    depot1_nsk = _read_depot_strings(data, depot1_offset, depot1_count)
    depot2_emb = _read_depot_strings(data, depot2_offset, depot2_count)
    depot3_emm = _read_depot_strings(data, depot3_offset, depot3_count)
    depot4_ema = _read_depot_strings(data, depot4_offset, depot4_count)
    collision_groups = _read_collision_groups(
        data,
        collision_group_offset,
        collision_group_count,
        is_old_version,
    )

    fmp = FMPFile(
        version=version,
        i_08=i_08,
        i_12=i_12,
        i_96=i_96,
        collision_groups=collision_groups,
    )
    if object_count <= 0:
        return fmp

    if not is_valid_offset(data, object_offset, object_count * 84):
        raise ValueError("Invalid FMP object table offset/count.")

    for index in range(object_count):
        current_offset = object_offset + (index * 84)
        fmp.objects.append(
            _read_object(
                data,
                index,
                current_offset,
                depot1_nsk,
                depot2_emb,
                depot3_emm,
                depot4_ema,
                collision_groups,
            )
        )

    return fmp


def parse_fmp(path: str | Path) -> FMPFile:
    with open(path, "rb") as file_handle:
        return parse_fmp_bytes(file_handle.read())


__all__ = [
    "AXIS_BLENDER_TO_XV2",
    "AXIS_XV2_TO_BLENDER",
    "FMPColliderInstance",
    "FMPCollider",
    "FMPCollisionVertex",
    "FMPCollisionVertexData",
    "FMPCollisionGroup",
    "FMPFile",
    "FMPHavokGroupParameters",
    "FMPHavokEntry",
    "FMPHvkCollisionData",
    "FMPInstanceBVHNode",
    "FMPInstanceData",
    "FMPInstanceGroup",
    "FMPInstanceTransform",
    "FMPLOD",
    "FMPObject",
    "FMPObjectSubPart",
    "FMPEntity",
    "FMPTransform",
    "FMPVisual",
    "FMP_SIGNATURE",
    "iter_object_instance_matrices",
    "map_name_from_path",
    "normalize_cache_path",
    "parse_fmp",
    "parse_fmp_bytes",
    "pick_entity_lod",
    "resolve_nsk_path",
    "resolve_optional_asset_path",
    "sanitize_name",
    "serialize_instance_bvh",
    "serialize_instance_data",
    "serialize_visual_lods",
    "to_blender_axis",
    "to_json_string",
    "to_xv2_axis",
]
