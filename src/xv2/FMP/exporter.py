from __future__ import annotations

import copy
import json
import math
import os
import re
import struct
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import bpy
import mathutils

from ...utils.binary import is_valid_offset
from .FMP import (
    AXIS_BLENDER_TO_XV2,
    FMP_SIGNATURE,
    FMPLOD,
    FMPColliderInstance,
    FMPEntity,
    FMPFile,
    FMPHavokGroupParameters,
    FMPInstanceBVHNode,
    FMPInstanceData,
    FMPInstanceGroup,
    FMPInstanceTransform,
    FMPObject,
    FMPObjectSubPart,
    FMPTransform,
    FMPVisual,
    parse_fmp_bytes,
    to_xv2_axis,
)

_OBJECT_NAME_RE = re.compile(r"^(?P<base>.+)_object(?:_(?P<instance>\d{3}))?$")
_ENTITY_NAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<entity>.+)_entity(?:_(?P<entity_idx>\d{2}))(?:_(?P<instance_idx>\d{3}))?$"
)
_AXIS3_BLENDER_TO_XV2 = AXIS_BLENDER_TO_XV2.to_3x3()
CollisionMeshData = tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[int],
    int | None,
]


@dataclass(slots=True)
class FMPExportLOD:
    index: int
    distance: float
    nsk_file: str
    emm_file: str


@dataclass(slots=True)
class FMPExportEntity:
    index: int | None
    i_04: int
    name: str
    visual_i04: int
    visual_i24: int
    visual_i28: int
    visual_i36: int
    visual_f40: float
    visual_f44: float
    emb_file: str
    ema_file: str
    lod_index: int
    lods: list[FMPExportLOD] = field(default_factory=list)
    transform_matrix: mathutils.Matrix = field(default_factory=lambda: mathutils.Matrix.Identity(4))


@dataclass(slots=True)
class FMPExportInstance:
    index: int
    transform_matrix: mathutils.Matrix = field(default_factory=lambda: mathutils.Matrix.Identity(4))


@dataclass(slots=True)
class FMPExportObject:
    index: int | None
    name: str
    i_04: int
    initial_entity_index: int
    flags: int
    f_32: float
    instance_data_json: str
    instances: list[FMPExportInstance] = field(default_factory=list)
    entities: list[FMPExportEntity] = field(default_factory=list)
    collider_instances: list[FMPColliderInstance] = field(default_factory=list)
    collision_meshes: dict[int, CollisionMeshData] = field(default_factory=dict)


@dataclass(slots=True)
class FMPExportPlan:
    source_path: str
    version: int
    i_08: int
    i_12: int
    i_96: tuple[int, int, int, int]
    objects: list[FMPExportObject] = field(default_factory=list)


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _parse_json(value: str, default):
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _is_map_object(obj: bpy.types.Object) -> bool:
    return obj.type == "EMPTY" and (
        "fmp_object_index" in obj or _OBJECT_NAME_RE.match(obj.name) is not None
    )


def _is_entity(obj: bpy.types.Object) -> bool:
    return obj.type == "EMPTY" and ("fmp_entity_index" in obj or "_entity" in obj.name)


def _get_map_root() -> bpy.types.Object | None:
    active_obj = bpy.context.active_object
    if active_obj and active_obj.get("fmp_source_path"):
        return active_obj
    parent = active_obj.parent if active_obj else None
    if parent and parent.get("fmp_source_path"):
        return parent
    return None


def _is_collider(obj: bpy.types.Object) -> bool:
    return obj.type == "EMPTY" and "fmp_collider_index" in obj


def _object_name(obj: bpy.types.Object) -> str:
    object_name = obj.get("fmp_object_name")
    if object_name is not None:
        return _as_str(object_name, obj.name)
    match = _OBJECT_NAME_RE.match(obj.name)
    if match is not None:
        return _as_str(match.group("base"), obj.name)
    return obj.name


def _instance_index(obj: bpy.types.Object, default_index: int) -> int:
    if "fmp_instance_index" in obj:
        return _as_int(obj.get("fmp_instance_index"), default_index)
    match = _OBJECT_NAME_RE.match(obj.name)
    if match is None:
        return default_index
    inst_token = match.group("instance")
    return int(inst_token) if inst_token else default_index


def _get_entity_lods(entity_obj: bpy.types.Object) -> tuple[int, list[FMPExportLOD]]:
    lod_json = _parse_json(_as_str(entity_obj.get("fmp_lods_json", "")), default=[])
    lods: list[FMPExportLOD] = []
    if isinstance(lod_json, list):
        for lod_index, lod_item in enumerate(lod_json):
            if not isinstance(lod_item, dict):
                continue
            lods.append(
                FMPExportLOD(
                    index=_as_int(lod_item.get("index"), lod_index),
                    distance=_as_float(lod_item.get("distance"), 0.0),
                    nsk_file=_as_str(lod_item.get("nsk_file"), ""),
                    emm_file=_as_str(lod_item.get("emm_file"), ""),
                )
            )
    if not lods:
        fallback_nsk = _as_str(entity_obj.get("fmp_nsk_file"), "")
        fallback_emm = _as_str(entity_obj.get("fmp_emm_file"), "")
        fallback_dist = _as_float(entity_obj.get("fmp_lod_distance"), 10000.0)
        if fallback_nsk or fallback_emm:
            lods.append(
                FMPExportLOD(
                    index=0,
                    distance=fallback_dist,
                    nsk_file=fallback_nsk,
                    emm_file=fallback_emm,
                )
            )

    if not lods:
        arm_child = next((child for child in entity_obj.children if child.type == "ARMATURE"), None)
        if arm_child is not None:
            nsk_name = _as_str(arm_child.get("nsk_source_name"), arm_child.name)
            fallback_nsk = (
                f"{nsk_name}.nsk"
                if nsk_name and not nsk_name.lower().endswith(".nsk")
                else nsk_name
            )
            fallback_emm = (
                f"{fallback_nsk[:-4]}.emm" if fallback_nsk.lower().endswith(".nsk") else ""
            )
            lods.append(
                FMPExportLOD(
                    index=0,
                    distance=10000.0,
                    nsk_file=fallback_nsk,
                    emm_file=fallback_emm,
                )
            )

    lod_index = _as_int(entity_obj.get("fmp_lod_index"), 0)
    return lod_index, lods


def _entity_name(entity_obj: bpy.types.Object, fallback: str) -> str:
    if "fmp_visual_name" in entity_obj:
        return _as_str(entity_obj.get("fmp_visual_name"), fallback)
    if "fmp_parent_entity" in entity_obj:
        return _as_str(entity_obj.get("fmp_parent_entity"), fallback)
    match = _ENTITY_NAME_RE.match(entity_obj.name)
    if match is not None:
        return _as_str(match.group("entity"), fallback)
    return fallback


def _build_entity_from_empty(entity_obj: bpy.types.Object, fallback_index: int) -> FMPExportEntity:
    lod_index, lods = _get_entity_lods(entity_obj)
    fallback_name = f"entity_{fallback_index:03d}"
    entity_name = _entity_name(entity_obj, fallback_name)
    return FMPExportEntity(
        index=_as_int(entity_obj.get("fmp_entity_index"), fallback_index)
        if "fmp_entity_index" in entity_obj
        else None,
        i_04=_as_int(entity_obj.get("fmp_entity_i04"), 0),
        name=entity_name,
        visual_i04=_as_int(entity_obj.get("fmp_visual_i04"), 0),
        visual_i24=_as_int(entity_obj.get("fmp_visual_i24"), -1),
        visual_i28=_as_int(entity_obj.get("fmp_visual_i28"), -1),
        visual_i36=_as_int(entity_obj.get("fmp_visual_i36"), -1),
        visual_f40=_as_float(entity_obj.get("fmp_visual_f40"), 60.0),
        visual_f44=_as_float(entity_obj.get("fmp_visual_f44"), 60.0),
        emb_file=_as_str(entity_obj.get("fmp_emb_file"), ""),
        ema_file=_as_str(entity_obj.get("fmp_ema_file"), ""),
        lod_index=lod_index,
        lods=lods,
        transform_matrix=to_xv2_axis(entity_obj.matrix_local.copy()),
    )


def _build_entity_from_armature(arm_obj: bpy.types.Object, fallback_index: int) -> FMPExportEntity:
    nsk_name = _as_str(arm_obj.get("nsk_source_name"), arm_obj.name)
    nsk_file = f"{nsk_name}.nsk" if nsk_name and not nsk_name.lower().endswith(".nsk") else nsk_name
    emm_file = f"{nsk_file[:-4]}.emm" if nsk_file.lower().endswith(".nsk") else ""
    lods = [FMPExportLOD(index=0, distance=10000.0, nsk_file=nsk_file, emm_file=emm_file)]
    return FMPExportEntity(
        index=None,
        i_04=0,
        name=arm_obj.name,
        visual_i04=0,
        visual_i24=-1,
        visual_i28=-1,
        visual_i36=-1,
        visual_f40=60.0,
        visual_f44=60.0,
        emb_file="",
        ema_file="",
        lod_index=0,
        lods=lods,
        transform_matrix=to_xv2_axis(arm_obj.matrix_local.copy()),
    )


def _collect_entities(obj_empty: bpy.types.Object) -> list[FMPExportEntity]:
    entity_children = [child for child in obj_empty.children if _is_entity(child)]
    if entity_children:
        return [
            _build_entity_from_empty(entity_obj, fallback_index)
            for fallback_index, entity_obj in enumerate(entity_children)
        ]

    armatures = [child for child in obj_empty.children if child.type == "ARMATURE"]
    return [
        _build_entity_from_armature(armature_obj, fallback_index)
        for fallback_index, armature_obj in enumerate(armatures)
    ]


def _iter_descendants(root_obj: bpy.types.Object) -> Iterator[bpy.types.Object]:
    stack = list(root_obj.children)
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.children)


def _resolve_entity_nsk_path(entity_obj: bpy.types.Object) -> str:
    direct_nsk = _as_str(entity_obj.get("fmp_nsk_file"), "").strip()
    if direct_nsk:
        return direct_nsk

    lod_index = _as_int(entity_obj.get("fmp_lod_index"), 0)
    lod_json = _parse_json(_as_str(entity_obj.get("fmp_lods_json", "")), default=[])
    if isinstance(lod_json, list):
        fallback_item = None
        for lod_item in lod_json:
            if not isinstance(lod_item, dict):
                continue
            if _as_int(lod_item.get("index"), -1) == lod_index:
                return _as_str(lod_item.get("nsk_file"), "").strip()
            if fallback_item is None and _as_str(lod_item.get("nsk_file"), "").strip():
                fallback_item = lod_item
        if fallback_item is not None:
            return _as_str(fallback_item.get("nsk_file"), "").strip()

    return ""


def _find_entity_armature(entity_obj: bpy.types.Object) -> bpy.types.Object | None:
    for child in entity_obj.children:
        if child.type == "ARMATURE":
            return child

    for child in entity_obj.children:
        if child.instance_type != "COLLECTION" or child.instance_collection is None:
            continue
        for collection_obj in child.instance_collection.all_objects:
            if collection_obj.type == "ARMATURE":
                return collection_obj

    for descendant in _iter_descendants(entity_obj):
        if descendant.type == "ARMATURE":
            return descendant

    return None


def _export_linked_nsk_files(
    map_output_path: str,
    map_root: bpy.types.Object | None,
    warn: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    if map_root is None:
        return 0, 0

    from ..NSK.exporter import export_nsk

    map_output = Path(map_output_path).resolve()
    output_base = map_output.parent
    map_stem = map_output.stem.strip()
    stage_folder_name = map_stem.split("_", 1)[0].strip() if map_stem else ""
    stage_output_base = output_base / stage_folder_name if stage_folder_name else output_base
    entity_objects = [obj for obj in _iter_descendants(map_root) if _is_entity(obj)]
    active_obj = bpy.context.active_object
    active_entity: bpy.types.Object | None = None
    while active_obj is not None:
        if _is_entity(active_obj):
            active_entity = active_obj
            break
        active_obj = active_obj.parent
    if active_entity is not None:
        entity_objects.sort(key=lambda obj: 0 if obj is active_entity else 1)
    exported_by_path: dict[str, bpy.types.Object] = {}
    export_success = 0
    export_failed = 0

    for entity_obj in entity_objects:
        nsk_relative = _resolve_entity_nsk_path(entity_obj)
        if not nsk_relative:
            continue

        nsk_armature = _find_entity_armature(entity_obj)
        if nsk_armature is None:
            if warn:
                warn(f"NSK export skipped '{entity_obj.name}': no armature found.")
            continue

        normalized_rel = nsk_relative.replace("\\", os.sep).replace("/", os.sep).strip()
        if not normalized_rel:
            continue

        nsk_target_path = Path(normalized_rel)
        if nsk_target_path.is_absolute():
            output_path = nsk_target_path
        else:
            first_part = nsk_target_path.parts[0] if nsk_target_path.parts else ""
            if stage_folder_name and first_part.lower() == stage_folder_name.lower():
                output_path = output_base / nsk_target_path
            else:
                output_path = stage_output_base / nsk_target_path

        dedupe_key = os.path.normcase(os.path.normpath(str(output_path)))
        if dedupe_key in exported_by_path:
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok, error = export_nsk(str(output_path), nsk_armature)
        if ok:
            exported_by_path[dedupe_key] = nsk_armature
            export_success += 1
        else:
            export_failed += 1
            if warn:
                warn(
                    f"Failed linked NSK export for '{entity_obj.name}' -> '{output_path.name}': "
                    f"{error or 'unknown error'}"
                )

    return export_success, export_failed


def _subpart_from_props(
    collider_empty: bpy.types.Object,
    prefix: str,
) -> FMPObjectSubPart | None:
    if (
        f"{prefix}_i_00" not in collider_empty
        and f"{prefix}_mass" not in collider_empty
        and f"{prefix}_width_x" not in collider_empty
    ):
        return None

    return FMPObjectSubPart(
        i_00=_as_int(collider_empty.get(f"{prefix}_i_00"), 0) & 0xFFFF,
        i_02=_as_int(collider_empty.get(f"{prefix}_i_02"), 0) & 0xFFFF,
        i_04=_as_int(collider_empty.get(f"{prefix}_i_04"), 0) & 0xFFFF,
        i_06=_as_int(collider_empty.get(f"{prefix}_i_06"), 0) & 0xFFFF,
        i_08=_as_int(collider_empty.get(f"{prefix}_i_08"), 0),
        mass=_as_float(collider_empty.get(f"{prefix}_mass"), 0.0),
        f_16=_as_float(collider_empty.get(f"{prefix}_f_16"), 0.0),
        f_20=_as_float(collider_empty.get(f"{prefix}_f_20"), 0.0),
        f_24=_as_float(collider_empty.get(f"{prefix}_f_24"), 0.0),
        f_28=_as_float(collider_empty.get(f"{prefix}_f_28"), 0.0),
        f_32=_as_float(collider_empty.get(f"{prefix}_f_32"), 0.0),
        f_36=_as_float(collider_empty.get(f"{prefix}_f_36"), 0.0),
        f_40=_as_float(collider_empty.get(f"{prefix}_f_40"), 0.0),
        f_44=_as_float(collider_empty.get(f"{prefix}_f_44"), 0.0),
        f_48=_as_float(collider_empty.get(f"{prefix}_f_48"), 0.0),
        f_52=_as_float(collider_empty.get(f"{prefix}_f_52"), 0.0),
        width=(
            _as_float(collider_empty.get(f"{prefix}_width_x"), 0.0),
            _as_float(collider_empty.get(f"{prefix}_width_y"), 0.0),
            _as_float(collider_empty.get(f"{prefix}_width_z"), 0.0),
        ),
        quaternion=(
            _as_float(collider_empty.get(f"{prefix}_quat_x"), 0.0),
            _as_float(collider_empty.get(f"{prefix}_quat_y"), 0.0),
            _as_float(collider_empty.get(f"{prefix}_quat_z"), 0.0),
            _as_float(collider_empty.get(f"{prefix}_quat_w"), 1.0),
        ),
    )


def _collect_colliders(obj_empty: bpy.types.Object) -> list[FMPColliderInstance]:
    collider_empties = [child for child in obj_empty.children if _is_collider(child)]
    collider_empties.sort(
        key=lambda obj: (_as_int(obj.get("fmp_collider_index"), 10_000_000), obj.name)
    )

    colliders: list[FMPColliderInstance] = []
    for collider_empty in collider_empties:
        collider = FMPColliderInstance(
            i_20=_as_int(collider_empty.get("fmp_collider_i20"), 0) & 0xFFFF,
            i_22=_as_int(collider_empty.get("fmp_collider_i22"), 0xFFFF) & 0xFFFF,
            f_24=_as_float(collider_empty.get("fmp_collider_f24"), 0.0),
            f_28=_as_float(collider_empty.get("fmp_collider_f28"), 0.0),
            matrix=FMPTransform(matrix=to_xv2_axis(collider_empty.matrix_local.copy())),
            action_offset=_as_int(collider_empty.get("fmp_collider_action_offset"), 0),
        )

        params_json = _parse_json(
            _as_str(collider_empty.get("fmp_collider_havok_params_json"), ""),
            default=[],
        )
        if isinstance(params_json, list):
            collider.havok_group_parameters = [
                FMPHavokGroupParameters(
                    param1=_as_int(item.get("param1"), 0),
                    param2=_as_int(item.get("param2"), 0),
                )
                for item in params_json
                if isinstance(item, dict)
            ]

        collider.subpart1 = _subpart_from_props(collider_empty, "fmp_collider_subpart1")
        collider.subpart2 = _subpart_from_props(collider_empty, "fmp_collider_subpart2")
        colliders.append(collider)

    return colliders


def _to_xv2_point(position: mathutils.Vector) -> tuple[float, float, float]:
    vec = _AXIS3_BLENDER_TO_XV2 @ position
    return (float(vec.x), float(vec.y), float(vec.z))


def _to_xv2_normal(normal: mathutils.Vector) -> tuple[float, float, float]:
    vec = _AXIS3_BLENDER_TO_XV2 @ normal
    if vec.length_squared > 0.0:
        vec.normalize()
    return (float(vec.x), float(vec.y), float(vec.z))


def _collect_collision_meshes(
    obj_empty: bpy.types.Object,
) -> dict[int, CollisionMeshData]:
    collision_meshes: dict[int, CollisionMeshData] = {}
    collider_empties = [child for child in obj_empty.children if _is_collider(child)]
    for collider_empty in collider_empties:
        collider_index = _as_int(collider_empty.get("fmp_collider_index"), -1)
        if collider_index < 0:
            continue

        preferred_meshes = [
            child
            for child in collider_empty.children
            if child.type == "MESH" and bool(child.get("fmp_collision_mesh"))
        ]
        mesh_obj = preferred_meshes[0] if preferred_meshes else None
        if mesh_obj is None:
            mesh_obj = next(
                (child for child in collider_empty.children if child.type == "MESH"),
                None,
            )
        if mesh_obj is None:
            continue

        collision_type: int | None = None
        if mesh_obj.data is None:
            continue

        if "fmp_collision_type" in mesh_obj:
            raw_collision_type = mesh_obj.get("fmp_collision_type")
            if raw_collision_type is not None:
                collision_type = _as_int(raw_collision_type, 0)

        mesh_data = mesh_obj.data
        mesh_local = mesh_obj.matrix_local
        mesh_local3 = mesh_local.to_3x3()
        flip_winding = mesh_local3.determinant() < 0.0

        vertices = [
            _to_xv2_point((mesh_local @ vertex.co.to_4d()).to_3d()) for vertex in mesh_data.vertices
        ]
        normals: list[tuple[float, float, float]] = []
        for vertex in mesh_data.vertices:
            normal = mesh_local3 @ vertex.normal
            if flip_winding:
                normal = -normal
            normals.append(_to_xv2_normal(normal))

        if not vertices:
            continue

        faces: list[int] = []
        mesh_data.calc_loop_triangles()
        for triangle in mesh_data.loop_triangles:
            v0, v1, v2 = triangle.vertices
            if v0 in (v1, v2) or v1 == v2:
                continue
            if flip_winding:
                faces.extend((int(v0), int(v2), int(v1)))
            else:
                faces.extend((int(v0), int(v1), int(v2)))

        if not vertices or not faces:
            continue

        collision_meshes[collider_index] = (vertices, normals, faces, collision_type)

    return collision_meshes


def _group_objects(map_root: bpy.types.Object) -> list[list[bpy.types.Object]]:
    grouped: dict[str, list[bpy.types.Object]] = defaultdict(list)

    for child in map_root.children:
        if not _is_map_object(child):
            continue
        if "fmp_object_index" in child:
            group_key = f"idx:{_as_int(child.get('fmp_object_index'))}"
        else:
            group_key = f"name:{_object_name(child)}"
        grouped[group_key].append(child)

    groups = list(grouped.values())
    groups.sort(key=lambda group: min(obj.name for obj in group))
    return groups


def collect_map_export_plan(
    map_root: bpy.types.Object,
    warn: Callable[[str], None] | None = None,
    include_collision_meshes: bool = False,
) -> FMPExportPlan:
    plan = FMPExportPlan(
        source_path=_as_str(map_root.get("fmp_source_path"), ""),
        version=_as_int(map_root.get("fmp_version"), 0),
        i_08=_as_int(map_root.get("fmp_i08"), 18),
        i_12=_as_int(map_root.get("fmp_i12"), 0),
        i_96=(
            _as_int(map_root.get("fmp_i96_0"), 0),
            _as_int(map_root.get("fmp_i96_1"), 0),
            _as_int(map_root.get("fmp_i96_2"), 0),
            _as_int(map_root.get("fmp_i96_3"), 0),
        ),
    )

    for object_group in _group_objects(map_root):
        object_group_sorted = sorted(
            object_group,
            key=lambda obj: (_instance_index(obj, 0), obj.name),
        )
        base_obj = object_group_sorted[0]
        object_name = _object_name(base_obj)

        instance_json = _as_str(base_obj.get("fmp_instance_data_json"), "")
        export_object = FMPExportObject(
            index=_as_int(base_obj.get("fmp_object_index"), -1)
            if "fmp_object_index" in base_obj
            else None,
            name=object_name,
            i_04=_as_int(base_obj.get("fmp_i_04"), _as_int(base_obj.get("fmp_object_index"), 0)),
            initial_entity_index=_as_int(base_obj.get("fmp_initial_entity_index"), 0),
            flags=_as_int(base_obj.get("fmp_flags"), 0),
            f_32=_as_float(base_obj.get("fmp_f_32"), 0.0),
            instance_data_json=instance_json,
        )

        for fallback_idx, object_instance in enumerate(object_group_sorted):
            export_object.instances.append(
                FMPExportInstance(
                    index=_instance_index(object_instance, fallback_idx),
                    transform_matrix=to_xv2_axis(object_instance.matrix_local.copy()),
                )
            )

        export_object.entities = _collect_entities(base_obj)
        export_object.collider_instances = _collect_colliders(base_obj)
        if include_collision_meshes:
            merged_collision_meshes: dict[int, CollisionMeshData] = {}
            for object_instance in object_group_sorted:
                instance_collision_meshes = _collect_collision_meshes(object_instance)
                if instance_collision_meshes:
                    merged_collision_meshes.update(instance_collision_meshes)
            export_object.collision_meshes = merged_collision_meshes
        if not export_object.entities and warn:
            warn(f"Object '{object_name}' has no entity empties or armatures; exporting it empty.")

        plan.objects.append(export_object)

    return plan


def collect_map_export_plan_from_active(
    warn: Callable[[str], None] | None = None,
    include_collision_meshes: bool = False,
) -> FMPExportPlan | None:
    map_root = _get_map_root()
    if map_root is None:
        return None
    return collect_map_export_plan(
        map_root,
        warn=warn,
        include_collision_meshes=include_collision_meshes,
    )


def _matrix_to_transform(matrix: mathutils.Matrix) -> FMPTransform:
    return FMPTransform(matrix=matrix.copy())


def _matrix_to_instance_transform(
    matrix: mathutils.Matrix,
    rotation_compat: tuple[float, float, float] | None = None,
) -> FMPInstanceTransform:
    loc, rot, scale = matrix.decompose()
    compat_euler = None
    if rotation_compat is not None:
        compat_euler = mathutils.Euler(
            (
                math.radians(float(rotation_compat[0])),
                math.radians(float(rotation_compat[1])),
                math.radians(float(rotation_compat[2])),
            ),
            "XYZ",
        )
    euler = rot.to_euler("XYZ", compat_euler)
    return FMPInstanceTransform(
        matrix=matrix.copy(),
        position=(float(loc.x), float(loc.y), float(loc.z)),
        rotation=(
            math.degrees(float(euler.x)),
            math.degrees(float(euler.y)),
            math.degrees(float(euler.z)),
        ),
        scale=(float(scale.x), float(scale.y), float(scale.z)),
    )


def _bvh_from_json(node_json: object, group_count: int) -> FMPInstanceBVHNode | None:
    if not isinstance(node_json, dict):
        return None
    node = FMPInstanceBVHNode(
        index=_as_int(node_json.get("index"), 0),
        flags=_as_int(node_json.get("flags"), 0),
    )
    center = node_json.get("center")
    if isinstance(center, list) and len(center) >= 3:
        node.center = (
            _as_float(center[0], 0.0),
            _as_float(center[1], 0.0),
            _as_float(center[2], 0.0),
        )
    group_indices = node_json.get("group_indices")
    if isinstance(group_indices, list):
        for group_index in group_indices:
            parsed_index = _as_int(group_index, -1)
            if 0 <= parsed_index < group_count:
                node.group_indices.append(parsed_index)
    children = node_json.get("children")
    if isinstance(children, list):
        for child_json in children:
            child = _bvh_from_json(child_json, group_count)
            if child is not None:
                node.children.append(child)
    return node


def _make_leaf_bvh(group_count: int) -> FMPInstanceBVHNode | None:
    if group_count <= 0:
        return None
    return FMPInstanceBVHNode(
        index=0,
        flags=0,
        center=(0.0, 0.0, 0.0),
        group_indices=list(range(group_count)),
        children=[],
    )


def _build_instance_data_for_object(
    plan_object: FMPExportObject,
    object_base_matrix: mathutils.Matrix,
    source_has_instance_data: bool = False,
) -> FMPInstanceData | None:
    if not plan_object.instances:
        return None

    has_source_instance_data = source_has_instance_data or bool(
        (plan_object.instance_data_json or "").strip()
    )
    if len(plan_object.instances) <= 1 and not has_source_instance_data:
        return None

    instance_json = _parse_json(plan_object.instance_data_json, default={})
    existing_groups_json = (
        instance_json.get("groups", []) if isinstance(instance_json, dict) else []
    )

    base_matrix = object_base_matrix.copy()
    inv_base = base_matrix.inverted_safe()

    def _vec3_from_json(value, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        if isinstance(value, list) and len(value) >= 3:
            return (
                _as_float(value[0], fallback[0]),
                _as_float(value[1], fallback[1]),
                _as_float(value[2], fallback[2]),
            )
        return fallback

    def _transform_from_json(value) -> FMPInstanceTransform:
        value_dict = value if isinstance(value, dict) else {}
        position = _vec3_from_json(value_dict.get("position"), (0.0, 0.0, 0.0))
        rotation = _vec3_from_json(value_dict.get("rotation"), (0.0, 0.0, 0.0))
        scale = _vec3_from_json(value_dict.get("scale"), (1.0, 1.0, 1.0))
        return FMPInstanceTransform(
            position=position,
            rotation=rotation,
            scale=scale,
            matrix=mathutils.Matrix.Identity(4),
        )

    groups: list[FMPInstanceGroup] = []
    next_transform_index = 0
    has_source_groups = isinstance(existing_groups_json, list) and len(existing_groups_json) > 0
    source_transform_defaults: list[FMPInstanceTransform] = []
    if has_source_groups:
        for group_defaults in existing_groups_json:
            group_dict = group_defaults if isinstance(group_defaults, dict) else {}
            source_transforms = (
                group_dict.get("transforms")
                if isinstance(group_dict.get("transforms"), list)
                else []
            )
            source_transform_defaults.extend(
                _transform_from_json(source_transform) for source_transform in source_transforms
            )

    relative_transforms_by_instance: list[tuple[int, FMPInstanceTransform]] = []
    for instance in plan_object.instances:
        source_index = int(instance.index)
        relative_matrix = inv_base @ instance.transform_matrix
        rotation_compat = None
        if 0 <= source_index < len(source_transform_defaults):
            rotation_compat = source_transform_defaults[source_index].rotation
        relative_transforms_by_instance.append(
            (
                source_index,
                _matrix_to_instance_transform(relative_matrix, rotation_compat=rotation_compat),
            )
        )

    if source_transform_defaults:
        slot_count = max(len(source_transform_defaults), len(relative_transforms_by_instance))
        resolved_transforms: list[FMPInstanceTransform | None] = [None] * slot_count
        fallback_instances: list[FMPInstanceTransform] = []

        for source_index, transform in relative_transforms_by_instance:
            if 0 <= source_index < slot_count and resolved_transforms[source_index] is None:
                resolved_transforms[source_index] = transform
            else:
                fallback_instances.append(transform)

        fallback_cursor = 0
        for slot_index in range(slot_count):
            if resolved_transforms[slot_index] is not None:
                continue
            if fallback_cursor < len(fallback_instances):
                fallback_transform = fallback_instances[fallback_cursor]
                fallback_cursor += 1
                if slot_index < len(source_transform_defaults):
                    fallback_transform = _matrix_to_instance_transform(
                        fallback_transform.matrix,
                        rotation_compat=source_transform_defaults[slot_index].rotation,
                    )
                resolved_transforms[slot_index] = fallback_transform
                continue
            if slot_index < len(source_transform_defaults):
                resolved_transforms[slot_index] = source_transform_defaults[slot_index]
                continue
            resolved_transforms[slot_index] = FMPInstanceTransform()

        relative_transforms = [
            transform if transform is not None else FMPInstanceTransform()
            for transform in resolved_transforms
        ]
    else:
        relative_transforms = [transform for _, transform in relative_transforms_by_instance]

    if has_source_groups:
        for group_fallback_index, group_defaults in enumerate(existing_groups_json):
            group_dict = group_defaults if isinstance(group_defaults, dict) else {}
            source_group_index = _as_int(group_dict.get("index"), group_fallback_index)
            source_transforms = (
                group_dict.get("transforms")
                if isinstance(group_dict.get("transforms"), list)
                else []
            )
            source_transform_count = len(source_transforms)

            group_transforms: list[FMPInstanceTransform] = []
            if source_transform_count > 0:
                end_index = min(
                    next_transform_index + source_transform_count, len(relative_transforms)
                )
                group_transforms = relative_transforms[next_transform_index:end_index]
                next_transform_index = end_index

            fallback_center = group_transforms[0].position if group_transforms else (0.0, 0.0, 0.0)
            center = _vec3_from_json(group_dict.get("center"), fallback_center)
            min_bounds = _vec3_from_json(group_dict.get("min_bounds"), center)
            max_bounds = _vec3_from_json(group_dict.get("max_bounds"), center)
            max_distance = _as_float(group_dict.get("max_distance"), 0.0)

            groups.append(
                FMPInstanceGroup(
                    index=source_group_index,
                    center=center,
                    max_distance=max_distance,
                    min_bounds=min_bounds,
                    max_bounds=max_bounds,
                    transforms=group_transforms,
                )
            )

    while next_transform_index < len(relative_transforms):
        transform = relative_transforms[next_transform_index]
        next_transform_index += 1
        center = transform.position
        groups.append(
            FMPInstanceGroup(
                index=len(groups),
                center=center,
                max_distance=0.0,
                min_bounds=center,
                max_bounds=center,
                transforms=[transform],
            )
        )

    bvh_root = None
    if isinstance(instance_json, dict):
        bvh_root = _bvh_from_json(instance_json.get("bvh_root"), len(groups))
    if bvh_root is None:
        bvh_root = _make_leaf_bvh(len(groups))

    return FMPInstanceData(groups=groups, bvh_root=bvh_root)


def _entity_to_fmp(entity: FMPExportEntity, fallback: FMPEntity | None = None) -> FMPEntity:
    out_entity = copy.deepcopy(fallback) if fallback is not None else FMPEntity()
    out_entity.i_04 = int(entity.i_04)
    out_entity.transform = _matrix_to_transform(entity.transform_matrix)

    out_visual = copy.deepcopy(out_entity.visual) if out_entity.visual is not None else FMPVisual()
    out_visual.name = entity.name
    out_visual.i_04 = int(entity.visual_i04)
    out_visual.i_24 = int(entity.visual_i24)
    out_visual.i_28 = int(entity.visual_i28)
    out_visual.i_36 = int(entity.visual_i36)
    out_visual.f_40 = float(entity.visual_f40)
    out_visual.f_44 = float(entity.visual_f44)
    out_visual.emb_file = entity.emb_file
    out_visual.ema_file = entity.ema_file
    out_visual.lods = [
        FMPLOD(
            distance=float(lod.distance),
            nsk_file=lod.nsk_file,
            emm_file=lod.emm_file,
        )
        for lod in sorted(entity.lods, key=lambda lod: (lod.index, lod.distance))
    ]
    out_entity.visual = out_visual
    return out_entity


def _object_from_plan(plan_object: FMPExportObject, fallback: FMPObject | None = None) -> FMPObject:
    out_object = copy.deepcopy(fallback) if fallback is not None else FMPObject()
    out_object.name = plan_object.name
    out_object.i_04 = int(plan_object.i_04)
    out_object.initial_entity_index = int(plan_object.initial_entity_index)
    out_object.flags = int(plan_object.flags)
    out_object.f_32 = float(plan_object.f_32)

    fallback_has_instance_data = fallback is not None and fallback.instance_data is not None
    plan_has_instance_data = bool((plan_object.instance_data_json or "").strip())
    has_instance_data = fallback_has_instance_data or plan_has_instance_data

    if plan_object.instances:
        if has_instance_data and fallback is not None:
            out_object.transform = _matrix_to_transform(fallback.transform.matrix)
        else:
            out_object.transform = _matrix_to_transform(plan_object.instances[0].transform_matrix)
    elif fallback is None:
        out_object.transform = FMPTransform()

    out_object.instance_data = _build_instance_data_for_object(
        plan_object,
        out_object.transform.matrix,
        source_has_instance_data=has_instance_data,
    )

    out_entities: list[FMPEntity] = copy.deepcopy(fallback.entities) if fallback is not None else []

    ordered_entities = sorted(
        plan_object.entities,
        key=lambda entity: (entity.index if entity.index is not None else 10_000_000),
    )
    for fallback_idx, entity in enumerate(ordered_entities):
        target_index = entity.index
        if target_index is not None and 0 <= target_index < len(out_entities):
            out_entities[target_index] = _entity_to_fmp(entity, fallback=out_entities[target_index])
            continue

        fallback_entity = None
        if fallback is not None and fallback_idx < len(fallback.entities):
            fallback_entity = fallback.entities[fallback_idx]

        out_entities.append(_entity_to_fmp(entity, fallback=fallback_entity))

    out_object.entities = out_entities
    if plan_object.collider_instances:
        out_object.collider_instances = copy.deepcopy(plan_object.collider_instances)
    elif fallback is None:
        out_object.collider_instances = []
    return out_object


def _merge_plan_into_source(plan: FMPExportPlan, source_fmp: FMPFile) -> FMPFile:
    merged = copy.deepcopy(source_fmp)
    if plan.version > 0:
        merged.version = int(plan.version)
    for plan_object in plan.objects:
        target_index = plan_object.index
        if target_index is not None and 0 <= target_index < len(merged.objects):
            merged.objects[target_index] = _object_from_plan(
                plan_object, merged.objects[target_index]
            )
            continue
        merged.objects.append(_object_from_plan(plan_object, None))

    for object_index, obj in enumerate(merged.objects):
        obj.idx = object_index
    return merged


def _append_aligned(buffer: bytearray, alignment: int = 4) -> None:
    if alignment <= 1:
        return
    pad = (-len(buffer)) % alignment
    if pad:
        buffer.extend(b"\x00" * pad)


def _append_blob(buffer: bytearray, blob: bytes, alignment: int = 4) -> int:
    _append_aligned(buffer, alignment=alignment)
    offset = len(buffer)
    buffer.extend(blob)
    return offset


def _pack_transform(matrix: mathutils.Matrix) -> bytes:
    row = matrix.transposed()
    values = (
        float(row[0][0]),
        float(row[0][1]),
        float(row[0][2]),
        float(row[1][0]),
        float(row[1][1]),
        float(row[1][2]),
        float(row[2][0]),
        float(row[2][1]),
        float(row[2][2]),
        float(row[3][0]),
        float(row[3][1]),
        float(row[3][2]),
    )
    return struct.pack("<12f", *values)


def _align_buffer_size(buffer: bytearray, alignment: int = 4) -> int:
    if alignment <= 1:
        return len(buffer)
    pad = (-len(buffer)) % alignment
    if pad:
        buffer.extend(b"\x00" * pad)
    return len(buffer)


def _append_collision_vertices_to_buffer(
    buffer: bytearray,
    vertices: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]],
) -> int:
    write_offset = _align_buffer_size(buffer, 4)
    for vertex, normal in zip(vertices, normals, strict=True):
        vx, vy, vz = vertex
        nx, ny, nz = normal
        buffer.extend(
            struct.pack(
                "<6f",
                float(vx),
                float(vy),
                float(vz),
                float(nx),
                float(ny),
                float(nz),
            )
        )
    return write_offset


def _append_collision_faces_to_buffer(buffer: bytearray, faces: list[int]) -> int:
    write_offset = _align_buffer_size(buffer, 2)
    for face_index in faces:
        buffer.extend(struct.pack("<H", int(face_index) & 0xFFFF))
    return write_offset


def _hvk_be_u32(data: bytearray, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def _hvk_read_packed(data: bytearray, offset: int) -> tuple[int, int]:
    value = data[offset]
    if (value & 0x80) == 0:
        return int(value), offset + 1
    if (value & 0x40) == 0:
        return int(((value << 8) | data[offset + 1]) & 0x3FFF), offset + 2
    if (value & 0x20) == 0:
        return int(
            ((value << 16) | (data[offset + 1] << 8) | data[offset + 2]) & 0x1FFFFF
        ), offset + 3
    return int(
        ((value << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3])
        & 0x1FFFFFFF
    ), offset + 4


def _hvk_parse_part(
    data: bytearray,
    offset: int,
) -> dict[str, object] | None:
    if offset < 0 or (offset + 8) > len(data):
        return None

    size_and_flags = _hvk_be_u32(data, offset)
    size = int(size_and_flags & 0x3FFFFFFF)
    flags = int(size_and_flags & 0xC0000000)
    if size < 8 or (offset + size) > len(data):
        return None

    signature = bytes(data[offset + 4 : offset + 8]).decode("ascii", "ignore")
    children: list[dict[str, object]] = []
    if signature in {"TAG0", "TYPE", "INDX"}:
        child_offset = offset + 8
        end_offset = offset + size
        while child_offset < end_offset:
            child = _hvk_parse_part(data, child_offset)
            if child is None:
                return None
            children.append(child)
            child_size = int(child["size"])
            if child_size <= 0:
                return None
            child_offset += child_size
        if child_offset != end_offset:
            return None

    return {
        "offset": int(offset),
        "size": int(size),
        "flags": flags,
        "signature": signature,
        "children": children,
    }


def _hvk_find_part(part: dict[str, object], signature: str) -> dict[str, object] | None:
    if part.get("signature") == signature:
        return part
    for child in part.get("children", []):
        found = _hvk_find_part(child, signature)
        if found is not None:
            return found
    return None


def _hvk_rebuild_part(
    source_data: bytearray,
    part: dict[str, object],
    replacement_body_by_offset: dict[int, bytes],
) -> bytes:
    signature = str(part.get("signature", ""))
    part_offset = int(part.get("offset", 0))
    part_size = int(part.get("size", 0))
    part_flags = int(part.get("flags", 0))
    children = list(part.get("children", []))

    if signature in {"TAG0", "TYPE", "INDX"}:
        body = b"".join(
            _hvk_rebuild_part(source_data, child, replacement_body_by_offset) for child in children
        )
    elif part_offset in replacement_body_by_offset:
        body = replacement_body_by_offset[part_offset]
    else:
        body_start = part_offset + 8
        body_end = part_offset + part_size
        if body_start < 0 or body_end > len(source_data) or body_start > body_end:
            body = b""
        else:
            body = bytes(source_data[body_start:body_end])

    packed_size = (len(body) + 8) | (part_flags & 0xC0000000)
    header = struct.pack(">I", int(packed_size)) + signature.encode("ascii", "ignore")[:4].ljust(
        4, b"\x00"
    )
    return header + body


def _hvk_read_strings(data: bytearray, offset: int, size: int) -> list[str]:
    start = offset + 8
    end = offset + size
    if start < 0 or end > len(data) or start > end:
        return []
    strings: list[str] = []
    current = start
    while current < end:
        next_null = data.find(0, current, end)
        if next_null < 0:
            break
        strings.append(bytes(data[current:next_null]).decode("utf-8", "ignore"))
        current = next_null + 1
    return strings


def _hvk_parse_type_names(
    data: bytearray,
    tstr_part: dict[str, object] | None,
    tnam_part: dict[str, object] | None,
) -> list[str]:
    if tstr_part is None or tnam_part is None:
        return []

    tstr_list = _hvk_read_strings(data, int(tstr_part["offset"]), int(tstr_part["size"]))
    if not tstr_list:
        return []

    tnam_offset = int(tnam_part["offset"]) + 8
    tnam_end = int(tnam_part["offset"]) + int(tnam_part["size"])
    if tnam_offset >= tnam_end or tnam_end > len(data):
        return []

    type_count, tnam_offset = _hvk_read_packed(data, tnam_offset)
    if type_count <= 0:
        return []

    type_names = ["None"] * type_count
    for type_id in range(1, type_count):
        if tnam_offset >= tnam_end:
            break
        name_index, tnam_offset = _hvk_read_packed(data, tnam_offset)
        template_count, tnam_offset = _hvk_read_packed(data, tnam_offset)
        if 0 <= name_index < len(tstr_list):
            type_names[type_id] = tstr_list[name_index]
        for _ in range(template_count):
            if tnam_offset >= tnam_end:
                break
            _, tnam_offset = _hvk_read_packed(data, tnam_offset)
            _, tnam_offset = _hvk_read_packed(data, tnam_offset)
    return type_names


def _hvk_parse_item_records(
    data: bytearray,
    item_part: dict[str, object] | None,
) -> list[dict[str, int]]:
    if item_part is None:
        return []

    start = int(item_part["offset"]) + 8
    end = int(item_part["offset"]) + int(item_part["size"])
    if start < 0 or end > len(data) or start > end:
        return []

    records: list[dict[str, int]] = []
    current = start
    while current + 12 <= end:
        flag, data_offset, count = struct.unpack_from("<III", data, current)
        records.append(
            {
                "type_id": int(flag & 0xFFFFFF),
                "kind": int((flag >> 28) & 0xF),
                "data_offset": int(data_offset),
                "count": int(count),
                "item_offset": int(current),
            }
        )
        current += 12
    return records


def _hvk_read_vector4_item(
    payload: memoryview,
    record: dict[str, int],
) -> tuple[list[tuple[float, float, float]], list[float]] | None:
    count = int(record["count"])
    data_offset = int(record["data_offset"])
    if count <= 0 or data_offset < 0:
        return None
    required_size = count * 16
    if (data_offset + required_size) > len(payload):
        return None

    points: list[tuple[float, float, float]] = []
    w_values: list[float] = []
    for index in range(count):
        base_offset = data_offset + (index * 16)
        x, y, z, w = struct.unpack_from("<4f", payload, base_offset)
        points.append((float(x), float(y), float(z)))
        w_values.append(float(w))
    return points, w_values


def _hvk_write_vector4_item(
    payload: bytearray,
    record: dict[str, int],
    points: list[tuple[float, float, float]],
    w_values: list[float],
) -> bool:
    count = int(record["count"])
    data_offset = int(record["data_offset"])
    if len(points) != count or len(w_values) != count:
        return False
    if count <= 0 or data_offset < 0:
        return False
    if (data_offset + (count * 16)) > len(payload):
        return False
    for index in range(count):
        base_offset = data_offset + (index * 16)
        x, y, z = points[index]
        w = w_values[index]
        struct.pack_into("<4f", payload, base_offset, float(x), float(y), float(z), float(w))
    return True


def _hvk_read_triangle_item(
    payload: memoryview,
    record: dict[str, int],
) -> tuple[list[tuple[int, int, int]], list[int]] | None:
    count = int(record["count"])
    data_offset = int(record["data_offset"])
    if count <= 0 or data_offset < 0:
        return None
    required_size = count * 16
    if (data_offset + required_size) > len(payload):
        return None

    triangles: list[tuple[int, int, int]] = []
    w_values: list[int] = []
    for index in range(count):
        base_offset = data_offset + (index * 16)
        i0, i1, i2, iw = struct.unpack_from("<4i", payload, base_offset)
        triangles.append((int(i0), int(i1), int(i2)))
        w_values.append(int(iw))
    return triangles, w_values


def _hvk_write_triangle_item(
    payload: bytearray,
    record: dict[str, int],
    triangles: list[tuple[int, int, int]],
    w_values: list[int],
) -> bool:
    count = int(record["count"])
    data_offset = int(record["data_offset"])
    if len(triangles) != count or len(w_values) != count:
        return False
    if count <= 0 or data_offset < 0:
        return False
    if (data_offset + (count * 16)) > len(payload):
        return False
    for index in range(count):
        base_offset = data_offset + (index * 16)
        i0, i1, i2 = triangles[index]
        iw = w_values[index]
        struct.pack_into("<4i", payload, base_offset, int(i0), int(i1), int(i2), int(iw))
    return True


def _hvk_triangle_bounds(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    triangle_index: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if triangle_index < 0 or triangle_index >= len(triangles):
        return None
    i0, i1, i2 = triangles[triangle_index]
    if (
        i0 < 0
        or i1 < 0
        or i2 < 0
        or i0 >= len(vertices)
        or i1 >= len(vertices)
        or i2 >= len(vertices)
    ):
        return None

    v0 = vertices[i0]
    v1 = vertices[i1]
    v2 = vertices[i2]
    min_v = (
        min(v0[0], v1[0], v2[0]),
        min(v0[1], v1[1], v2[1]),
        min(v0[2], v1[2], v2[2]),
    )
    max_v = (
        max(v0[0], v1[0], v2[0]),
        max(v0[1], v1[1], v2[1]),
        max(v0[2], v1[2], v2[2]),
    )
    return min_v, max_v


def _hvk_bounds_centroid(
    min_v: tuple[float, float, float],
    max_v: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        (float(min_v[0]) + float(max_v[0])) * 0.5,
        (float(min_v[1]) + float(max_v[1])) * 0.5,
        (float(min_v[2]) + float(max_v[2])) * 0.5,
    )


def _hvk_distance3(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    dz = float(left[2]) - float(right[2])
    return math.sqrt((dx * dx) + (dy * dy) + (dz * dz))


def _hvk_node_overall_bounds(
    node: dict[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    mins = node["mins"]
    maxs = node["maxs"]
    indices = node["indices"]
    children = node["children"]

    valid_slots: list[int] = []
    for slot_index in range(4):
        if int(indices[slot_index]) != -1 or children[slot_index] is not None:
            valid_slots.append(slot_index)
    if not valid_slots:
        return None

    min_v = (
        min(float(mins[slot_index][0]) for slot_index in valid_slots),
        min(float(mins[slot_index][1]) for slot_index in valid_slots),
        min(float(mins[slot_index][2]) for slot_index in valid_slots),
    )
    max_v = (
        max(float(maxs[slot_index][0]) for slot_index in valid_slots),
        max(float(maxs[slot_index][1]) for slot_index in valid_slots),
        max(float(maxs[slot_index][2]) for slot_index in valid_slots),
    )
    return min_v, max_v


def _hvk_make_empty_node(is_leaf: bool) -> dict[str, object]:
    inf = float("inf")
    ninf = float("-inf")
    return {
        "mins": [(inf, inf, inf), (inf, inf, inf), (inf, inf, inf), (inf, inf, inf)],
        "maxs": [(ninf, ninf, ninf), (ninf, ninf, ninf), (ninf, ninf, ninf), (ninf, ninf, ninf)],
        "indices": [-1, -1, -1, -1],
        "children": [None, None, None, None],
        "centroid": (0.0, 0.0, 0.0),
        "is_leaf": bool(is_leaf),
    }


def _hvk_group_triangle_nodes(
    triangle_nodes: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: set[int] = set()
    out_nodes: list[dict[str, object]] = []

    for index in range(len(triangle_nodes)):
        if index in grouped:
            continue

        base = triangle_nodes[index]
        distances: list[tuple[float, int]] = []
        for other_index in range(len(triangle_nodes)):
            if other_index in grouped or other_index == index:
                continue
            distance = _hvk_distance3(
                base["centroid"],
                triangle_nodes[other_index]["centroid"],
            )
            distances.append((distance, other_index))
        distances.sort(key=lambda item: item[0])

        group_indices = [index]
        for _distance, other_index in distances:
            if len(group_indices) >= 4:
                break
            if other_index not in grouped:
                group_indices.append(other_index)

        node = _hvk_make_empty_node(is_leaf=True)
        centroid_sum = [0.0, 0.0, 0.0]
        for slot_index, tri_index in enumerate(group_indices):
            tri_node = triangle_nodes[tri_index]
            grouped.add(tri_index)
            node["mins"][slot_index] = tri_node["min"]
            node["maxs"][slot_index] = tri_node["max"]
            node["indices"][slot_index] = int(tri_node["triangle_index"])
            tri_centroid = tri_node["centroid"]
            centroid_sum[0] += float(tri_centroid[0])
            centroid_sum[1] += float(tri_centroid[1])
            centroid_sum[2] += float(tri_centroid[2])

        count = max(1, len(group_indices))
        node["centroid"] = (
            centroid_sum[0] / count,
            centroid_sum[1] / count,
            centroid_sum[2] / count,
        )
        out_nodes.append(node)

    return out_nodes


def _hvk_group_internal_nodes(
    input_nodes: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: set[int] = set()
    out_nodes: list[dict[str, object]] = []

    for index in range(len(input_nodes)):
        if index in grouped:
            continue

        base = input_nodes[index]
        distances: list[tuple[float, int]] = []
        for other_index in range(len(input_nodes)):
            if other_index in grouped or other_index == index:
                continue
            distance = _hvk_distance3(
                base["centroid"],
                input_nodes[other_index]["centroid"],
            )
            distances.append((distance, other_index))
        distances.sort(key=lambda item: item[0])

        group_indices = [index]
        for _distance, other_index in distances:
            if len(group_indices) >= 4:
                break
            if other_index not in grouped:
                grouped.add(other_index)
                group_indices.append(other_index)

        node = _hvk_make_empty_node(is_leaf=False)
        centroid_sum = [0.0, 0.0, 0.0]

        for slot_index, child_index in enumerate(group_indices):
            child_node = input_nodes[child_index]
            child_bounds = _hvk_node_overall_bounds(child_node)
            if child_bounds is None:
                continue
            child_min, child_max = child_bounds
            node["mins"][slot_index] = child_min
            node["maxs"][slot_index] = child_max
            node["children"][slot_index] = child_node
            child_centroid = _hvk_bounds_centroid(child_min, child_max)
            centroid_sum[0] += float(child_centroid[0])
            centroid_sum[1] += float(child_centroid[1])
            centroid_sum[2] += float(child_centroid[2])

        count = max(1, len(group_indices))
        node["centroid"] = (
            centroid_sum[0] / count,
            centroid_sum[1] / count,
            centroid_sum[2] / count,
        )
        out_nodes.append(node)

    return out_nodes


def _hvk_flatten_nodes(root: dict[str, object]) -> list[dict[str, object]]:
    flat: list[dict[str, object]] = []

    def visit(node: dict[str, object]) -> None:
        flat.append(node)
        children = node["children"]
        indices = node["indices"]
        for slot_index in range(3, -1, -1):
            child = children[slot_index]
            if child is None:
                continue
            indices[slot_index] = len(flat)
            visit(child)

    visit(root)
    return flat


def _hvk_build_simd_tree_blob(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    template_node_bytes: bytes,
) -> tuple[bytes, int] | None:
    if not triangles:
        return None

    triangle_nodes: list[dict[str, object]] = []
    for triangle_index in range(len(triangles)):
        bounds = _hvk_triangle_bounds(vertices, triangles, triangle_index)
        if bounds is None:
            return None
        min_v, max_v = bounds
        triangle_nodes.append(
            {
                "triangle_index": int(triangle_index),
                "min": min_v,
                "max": max_v,
                "centroid": _hvk_bounds_centroid(min_v, max_v),
            }
        )

    grouped_nodes = _hvk_group_triangle_nodes(triangle_nodes)
    if not grouped_nodes:
        return None
    while len(grouped_nodes) > 1:
        grouped_nodes = _hvk_group_internal_nodes(grouped_nodes)
    flat_nodes = _hvk_flatten_nodes(grouped_nodes[0])

    node_stride = 112
    template = bytearray(template_node_bytes)
    template = bytearray(node_stride) if len(template) < node_stride else template[:node_stride]

    blob = bytearray(template)
    for node in flat_nodes:
        node_blob = bytearray(template)
        mins = node["mins"]
        maxs = node["maxs"]
        indices = node["indices"]
        children = node["children"]
        is_leaf = bool(node["is_leaf"])

        for slot_index in range(4):
            has_entry = int(indices[slot_index]) != -1 or children[slot_index] is not None
            if not has_entry:
                continue
            slot_min = mins[slot_index]
            slot_max = maxs[slot_index]

            struct.pack_into("<f", node_blob, 0 + (slot_index * 4), float(slot_min[0]))
            struct.pack_into("<f", node_blob, 32 + (slot_index * 4), float(slot_min[1]))
            struct.pack_into("<f", node_blob, 64 + (slot_index * 4), float(slot_min[2]))
            struct.pack_into("<f", node_blob, 16 + (slot_index * 4), float(slot_max[0]))
            struct.pack_into("<f", node_blob, 48 + (slot_index * 4), float(slot_max[1]))
            struct.pack_into("<f", node_blob, 80 + (slot_index * 4), float(slot_max[2]))

            if is_leaf:
                code = (int(indices[slot_index]) * 2) + 1
            else:
                code = (int(indices[slot_index]) + 1) * 2
            struct.pack_into("<i", node_blob, 96 + (slot_index * 4), int(code))

        blob.extend(node_blob)

    return bytes(blob), int(len(flat_nodes) + 1)


def _patch_hvk_mesh_data(
    hvk_data: bytearray,
    edited_vertices: list[tuple[float, float, float]],
    edited_faces: list[int],
) -> bytes | None:
    if not hvk_data or not edited_vertices:
        return None

    root_part = _hvk_parse_part(hvk_data, 0)
    if root_part is None or root_part.get("signature") != "TAG0":
        return None

    data_part = _hvk_find_part(root_part, "DATA")
    tstr_part = _hvk_find_part(root_part, "TSTR")
    tnam_part = _hvk_find_part(root_part, "TNAM")
    item_part = _hvk_find_part(root_part, "ITEM")
    if data_part is None or item_part is None:
        return None

    payload_offset = int(data_part["offset"]) + 8
    payload_size = int(data_part["size"]) - 8
    if payload_offset < 0 or payload_size <= 0 or (payload_offset + payload_size) > len(hvk_data):
        return None

    payload_view = memoryview(hvk_data)[payload_offset : payload_offset + payload_size]
    payload = bytearray(payload_view.tobytes())

    type_names = _hvk_parse_type_names(hvk_data, tstr_part, tnam_part)
    item_records = _hvk_parse_item_records(hvk_data, item_part)
    if not item_records:
        return None

    # Resolve type names used in this Havok blob.
    named_records: dict[str, list[dict[str, int]]] = defaultdict(list)
    indexed_records: list[dict[str, int]] = []
    for index, record in enumerate(item_records):
        record_copy = dict(record)
        record_copy["index"] = int(index)
        indexed_records.append(record_copy)
        type_id = int(record["type_id"])
        if 0 <= type_id < len(type_names):
            type_name = type_names[type_id]
            if type_name:
                named_records[type_name].append(record_copy)

    vector_records = named_records.get("hkVector4", [])
    if not vector_records:
        return None
    vertex_record = max(vector_records, key=lambda record: int(record["count"]))
    vertex_read = _hvk_read_vector4_item(memoryview(payload), vertex_record)
    if vertex_read is None:
        return None
    old_hvk_vertices, _old_hvk_w = vertex_read
    if not old_hvk_vertices:
        return None

    new_hvk_vertices = list(edited_vertices)
    new_hvk_w = [0.5] * len(new_hvk_vertices)

    current_triangles: list[tuple[int, int, int]] = []
    triangle_record: dict[str, int] | None = None
    triangle_records = named_records.get("hkGeometry::Triangle", [])
    if triangle_records:
        triangle_record = max(triangle_records, key=lambda record: int(record["count"]))
        triangle_read = _hvk_read_triangle_item(memoryview(payload), triangle_record)
        if triangle_read is None:
            return None
        old_triangles, _old_triangle_w = triangle_read
        current_triangles = old_triangles

    if edited_faces and (len(edited_faces) % 3) != 0:
        return None
    if edited_faces:
        max_index = len(new_hvk_vertices) - 1
        rebuilt_triangles: list[tuple[int, int, int]] = []
        for face_offset in range(0, len(edited_faces), 3):
            i0 = int(edited_faces[face_offset + 0])
            i1 = int(edited_faces[face_offset + 1])
            i2 = int(edited_faces[face_offset + 2])
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 > max_index or i1 > max_index or i2 > max_index:
                return None
            rebuilt_triangles.append((i0, i1, i2))
        current_triangles = rebuilt_triangles

    # Match Xv2CoreLib behavior (triangle extra int remains default/zeroed).
    new_triangle_w = [0] * len(current_triangles) if triangle_record is not None else []

    if current_triangles:
        shape_key_bits = int(math.ceil(math.log(max(1, len(current_triangles)), 2)))
        if shape_key_bits > 24:
            return None

    simd_item_offset = -1
    simd_blob: bytes | None = None
    simd_node_count = 0
    simd_records = named_records.get("hkcdSimdTree::Node", [])
    if simd_records and current_triangles:
        simd_record_old = max(simd_records, key=lambda record: int(record["count"]))
        simd_item_offset = int(simd_record_old["item_offset"])
        simd_old_offset = int(simd_record_old["data_offset"])
        simd_old_count = int(simd_record_old["count"])
        simd_template = b""
        if simd_old_count > 0 and simd_old_offset >= 0 and (simd_old_offset + 112) <= len(payload):
            simd_template = bytes(payload[simd_old_offset : simd_old_offset + 112])
        simd_result = _hvk_build_simd_tree_blob(
            new_hvk_vertices,
            current_triangles,
            simd_template,
        )
        if simd_result is None:
            return None
        simd_blob, simd_node_count = simd_result

    old_offset_to_indices: dict[int, list[int]] = defaultdict(list)
    for record in indexed_records:
        old_offset_to_indices[int(record["data_offset"])].append(int(record["index"]))

    old_offsets_sorted = sorted(old_offset_to_indices.keys())
    old_starts = [offset for offset in old_offsets_sorted if offset >= 0]
    if not old_starts:
        return None

    old_next_offset: dict[int, int] = {}
    for index, start in enumerate(old_starts):
        end = old_starts[index + 1] if index + 1 < len(old_starts) else len(payload)
        if start > len(payload) or end > len(payload) or end < start:
            return None
        old_next_offset[start] = end

    vertex_item_offset = int(vertex_record["item_offset"])
    triangle_item_offset = (
        int(triangle_record["item_offset"]) if triangle_record is not None else -1
    )
    updated_records = [dict(record) for record in indexed_records]
    new_payload = bytearray()
    new_offset_by_old_offset: dict[int, int] = {}

    for old_offset in old_starts:
        group_indices = old_offset_to_indices.get(old_offset, [])
        old_end = old_next_offset.get(old_offset, old_offset)
        original_blob = bytes(payload[old_offset:old_end])

        group_has_vertex = any(
            int(indexed_records[group_index]["item_offset"]) == vertex_item_offset
            for group_index in group_indices
        )
        group_has_triangle = any(
            int(indexed_records[group_index]["item_offset"]) == triangle_item_offset
            for group_index in group_indices
        )
        group_has_simd = simd_item_offset >= 0 and any(
            int(indexed_records[group_index]["item_offset"]) == simd_item_offset
            for group_index in group_indices
        )
        if group_has_vertex and group_has_triangle:
            return None

        if group_has_vertex:
            new_blob = bytearray()
            for point_index, point in enumerate(new_hvk_vertices):
                x, y, z = point
                w = new_hvk_w[point_index]
                new_blob.extend(struct.pack("<4f", float(x), float(y), float(z), float(w)))
        elif group_has_triangle:
            new_blob = bytearray()
            for tri_index, triangle in enumerate(current_triangles):
                i0, i1, i2 = triangle
                iw = new_triangle_w[tri_index]
                new_blob.extend(struct.pack("<4i", int(i0), int(i1), int(i2), int(iw)))
        elif group_has_simd and simd_blob is not None:
            new_blob = bytearray(simd_blob)
        else:
            new_blob = bytearray(original_blob)

        while (len(new_payload) % 16) != (old_offset % 16):
            new_payload.extend(b"\x00")
        group_new_offset = len(new_payload)
        new_payload.extend(new_blob)
        new_offset_by_old_offset[old_offset] = group_new_offset

        for group_index in group_indices:
            record = updated_records[group_index]
            record["data_offset"] = int(group_new_offset)
            type_id = int(record["type_id"])
            type_name = type_names[type_id] if 0 <= type_id < len(type_names) else ""
            if group_has_vertex and type_name == "hkVector4":
                record["count"] = int(len(new_hvk_vertices))
            if group_has_triangle and type_name == "hkGeometry::Triangle":
                record["count"] = int(len(current_triangles))
            if group_has_simd and type_name == "hkcdSimdTree::Node" and simd_blob is not None:
                record["count"] = int(simd_node_count)

    new_item_body = bytearray()
    for record in updated_records:
        flag = ((int(record["kind"]) & 0xF) << 28) | (int(record["type_id"]) & 0xFFFFFF)
        new_item_body.extend(
            struct.pack(
                "<III",
                int(flag),
                int(record["data_offset"]),
                int(record["count"]),
            )
        )

    replacement_body_by_offset = {
        int(data_part["offset"]): bytes(new_payload),
        int(item_part["offset"]): bytes(new_item_body),
    }
    rebuilt = _hvk_rebuild_part(hvk_data, root_part, replacement_body_by_offset)
    if not rebuilt:
        return None

    return bytes(rebuilt)


def _flatten_instance_transforms(
    instance_data: FMPInstanceData | None,
) -> list[FMPInstanceTransform]:
    if instance_data is None:
        return []
    transforms: list[FMPInstanceTransform] = []
    for group in instance_data.groups:
        transforms.extend(group.transforms)
    return transforms


def _write_subpart_at(buffer: bytearray, offset: int, subpart: FMPObjectSubPart) -> None:
    if offset <= 0 or (offset + 84) > len(buffer):
        return
    struct.pack_into("<H", buffer, offset + 0, int(subpart.i_00) & 0xFFFF)
    struct.pack_into("<H", buffer, offset + 2, int(subpart.i_02) & 0xFFFF)
    struct.pack_into("<H", buffer, offset + 4, int(subpart.i_04) & 0xFFFF)
    struct.pack_into("<H", buffer, offset + 6, int(subpart.i_06) & 0xFFFF)
    struct.pack_into("<i", buffer, offset + 8, int(subpart.i_08))
    struct.pack_into("<f", buffer, offset + 12, float(subpart.mass))
    struct.pack_into("<f", buffer, offset + 16, float(subpart.f_16))
    struct.pack_into("<f", buffer, offset + 20, float(subpart.f_20))
    struct.pack_into("<f", buffer, offset + 24, float(subpart.f_24))
    struct.pack_into("<f", buffer, offset + 28, float(subpart.f_28))
    struct.pack_into("<f", buffer, offset + 32, float(subpart.f_32))
    struct.pack_into("<f", buffer, offset + 36, float(subpart.f_36))
    struct.pack_into("<f", buffer, offset + 40, float(subpart.f_40))
    struct.pack_into("<f", buffer, offset + 44, float(subpart.f_44))
    struct.pack_into("<f", buffer, offset + 48, float(subpart.f_48))
    struct.pack_into("<f", buffer, offset + 52, float(subpart.f_52))
    struct.pack_into("<f", buffer, offset + 56, float(subpart.width[0]))
    struct.pack_into("<f", buffer, offset + 60, float(subpart.width[1]))
    struct.pack_into("<f", buffer, offset + 64, float(subpart.width[2]))
    struct.pack_into("<f", buffer, offset + 68, float(subpart.quaternion[0]))
    struct.pack_into("<f", buffer, offset + 72, float(subpart.quaternion[1]))
    struct.pack_into("<f", buffer, offset + 76, float(subpart.quaternion[2]))
    struct.pack_into("<f", buffer, offset + 80, float(subpart.quaternion[3]))


def _can_reuse_source_layout(source_fmp: FMPFile, merged_fmp: FMPFile) -> bool:
    if len(source_fmp.objects) != len(merged_fmp.objects):
        return False

    for source_obj, merged_obj in zip(source_fmp.objects, merged_fmp.objects, strict=True):
        if len(source_obj.entities) != len(merged_obj.entities):
            return False
        if len(source_obj.collider_instances) != len(merged_obj.collider_instances):
            return False

        source_instance = source_obj.instance_data
        merged_instance = merged_obj.instance_data
        if (source_instance is None) != (merged_instance is None):
            return False

        if source_instance is not None and merged_instance is not None:
            if len(source_instance.groups) != len(merged_instance.groups):
                return False
            if len(_flatten_instance_transforms(source_instance)) != len(
                _flatten_instance_transforms(merged_instance)
            ):
                return False

        for source_entity, merged_entity in zip(
            source_obj.entities, merged_obj.entities, strict=True
        ):
            source_visual = source_entity.visual
            merged_visual = merged_entity.visual
            if (source_visual is None) != (merged_visual is None):
                return False
            if source_visual is None or merged_visual is None:
                continue
            if len(source_visual.lods) != len(merged_visual.lods):
                return False
            if source_visual.emb_file != merged_visual.emb_file:
                return False
            if source_visual.ema_file != merged_visual.ema_file:
                return False
            for source_lod, merged_lod in zip(source_visual.lods, merged_visual.lods, strict=True):
                if source_lod.nsk_file != merged_lod.nsk_file:
                    return False
                if source_lod.emm_file != merged_lod.emm_file:
                    return False

    return True


def _write_into_source_layout(
    source_bytes: bytes,
    merged_fmp: FMPFile,
    collision_meshes_by_object: dict[int, dict[int, CollisionMeshData]] | None = None,
) -> bytes:
    out = bytearray(source_bytes)
    rewritten_hvk_cache: dict[int, tuple[int, int]] = {}

    def i32(offset: int) -> int:
        return struct.unpack_from("<i", out, offset)[0]

    object_count = i32(48)
    object_offset = i32(52)
    collision_group_count = i32(56)
    collision_group_offset = i32(60)
    is_old_version = (i32(4) & 0xFFF00) == 0
    if object_count != len(merged_fmp.objects):
        raise ValueError("Object count mismatch for in-place patch export.")
    if object_offset <= 0:
        raise ValueError("Invalid object table offset for in-place patch export.")

    for object_index, obj in enumerate(merged_fmp.objects):
        record_offset = object_offset + (84 * object_index)
        if record_offset < 0 or (record_offset + 84) > len(out):
            raise ValueError("Object table range is out of bounds during in-place patch export.")

        object_collision_meshes = {}
        if collision_meshes_by_object:
            object_collision_meshes = collision_meshes_by_object.get(int(object_index), {})

        object_flags = int(obj.flags)

        struct.pack_into("<H", out, record_offset + 4, int(obj.i_04) & 0xFFFF)
        struct.pack_into("<H", out, record_offset + 6, int(obj.initial_entity_index) & 0xFFFF)
        struct.pack_into("<H", out, record_offset + 10, int(object_flags) & 0xFFFF)
        struct.pack_into("<f", out, record_offset + 32, float(obj.f_32))
        out[record_offset + 36 : record_offset + 84] = _pack_transform(obj.transform.matrix)

        entity_count = i32(record_offset + 20)
        entity_offset = i32(record_offset + 24)
        if entity_count != len(obj.entities):
            raise ValueError("Entity count mismatch for in-place patch export.")

        for entity_index, entity in enumerate(obj.entities):
            current_entity_offset = entity_offset + (56 * entity_index)
            if current_entity_offset < 0 or (current_entity_offset + 56) > len(out):
                raise ValueError(
                    "Entity table range is out of bounds during in-place patch export."
                )

            struct.pack_into("<i", out, current_entity_offset + 4, int(entity.i_04))
            out[current_entity_offset + 8 : current_entity_offset + 56] = _pack_transform(
                entity.transform.matrix
            )

            visual_offset = i32(current_entity_offset + 0)
            visual = entity.visual
            if visual is None:
                if visual_offset != 0:
                    raise ValueError("Visual mismatch for in-place patch export.")
                continue

            if visual_offset <= 0 or (visual_offset + 52) > len(out):
                raise ValueError(
                    "Visual table range is out of bounds during in-place patch export."
                )

            struct.pack_into("<i", out, visual_offset + 4, int(visual.i_04))
            struct.pack_into("<i", out, visual_offset + 24, int(visual.i_24))
            struct.pack_into("<i", out, visual_offset + 28, int(visual.i_28))
            struct.pack_into("<i", out, visual_offset + 36, int(visual.i_36))
            struct.pack_into("<f", out, visual_offset + 40, float(visual.f_40))
            struct.pack_into("<f", out, visual_offset + 44, float(visual.f_44))

            lod_count = i32(visual_offset + 8)
            if lod_count != len(visual.lods):
                raise ValueError("LOD count mismatch for in-place patch export.")

            if lod_count == 1 and visual.lods:
                struct.pack_into("<f", out, visual_offset + 48, float(visual.lods[0].distance))
            elif lod_count > 1:
                distance_table_offset = i32(visual_offset + 48)
                if distance_table_offset <= 0 or (distance_table_offset + (4 * lod_count)) > len(
                    out
                ):
                    raise ValueError(
                        "LOD distance table offset is invalid for in-place patch export."
                    )
                for lod_index, lod in enumerate(visual.lods):
                    struct.pack_into(
                        "<f",
                        out,
                        distance_table_offset + (4 * lod_index),
                        float(lod.distance),
                    )

        hierarchy_offset = i32(record_offset + 28)
        instance_data = obj.instance_data
        if instance_data is None:
            if hierarchy_offset != 0:
                raise ValueError("Instance data mismatch for in-place patch export.")
        else:
            if hierarchy_offset <= 0 or (hierarchy_offset + 20) > len(out):
                raise ValueError("Hierarchy offset is invalid for in-place patch export.")

            node_transform_count = i32(hierarchy_offset + 0)
            node_count = i32(hierarchy_offset + 4)
            node_offset = i32(hierarchy_offset + 8)
            node_transform_offset = i32(hierarchy_offset + 16)
            if node_count != len(instance_data.groups):
                raise ValueError("Instance group count mismatch for in-place patch export.")

            flat_transforms = _flatten_instance_transforms(instance_data)
            if node_transform_count != len(flat_transforms):
                raise ValueError("Instance transform count mismatch for in-place patch export.")

            next_transform_index = 0
            for group_index, group in enumerate(instance_data.groups):
                group_offset = node_offset + (48 * group_index)
                if group_offset < 0 or (group_offset + 48) > len(out):
                    raise ValueError(
                        "Instance group range is out of bounds during in-place patch export."
                    )

                transform_count = len(group.transforms)
                struct.pack_into("<f", out, group_offset + 0, float(group.center[0]))
                struct.pack_into("<f", out, group_offset + 4, float(group.center[1]))
                struct.pack_into("<f", out, group_offset + 8, float(group.center[2]))
                struct.pack_into("<f", out, group_offset + 12, float(group.max_distance))
                struct.pack_into("<f", out, group_offset + 16, float(group.min_bounds[0]))
                struct.pack_into("<f", out, group_offset + 20, float(group.min_bounds[1]))
                struct.pack_into("<f", out, group_offset + 24, float(group.min_bounds[2]))
                struct.pack_into("<f", out, group_offset + 28, float(group.max_bounds[0]))
                struct.pack_into("<f", out, group_offset + 32, float(group.max_bounds[1]))
                struct.pack_into("<f", out, group_offset + 36, float(group.max_bounds[2]))
                struct.pack_into("<i", out, group_offset + 40, int(transform_count))
                struct.pack_into("<i", out, group_offset + 44, int(next_transform_index))
                next_transform_index += transform_count

            for transform_index, transform in enumerate(flat_transforms):
                transform_offset = node_transform_offset + (36 * transform_index)
                if transform_offset < 0 or (transform_offset + 36) > len(out):
                    raise ValueError(
                        "Instance transform range is out of bounds during in-place patch export."
                    )

                struct.pack_into("<f", out, transform_offset + 0, float(transform.position[0]))
                struct.pack_into("<f", out, transform_offset + 4, float(transform.position[1]))
                struct.pack_into("<f", out, transform_offset + 8, float(transform.position[2]))
                struct.pack_into(
                    "<f",
                    out,
                    transform_offset + 12,
                    math.radians(float(transform.rotation[0])),
                )
                struct.pack_into(
                    "<f",
                    out,
                    transform_offset + 16,
                    math.radians(float(transform.rotation[1])),
                )
                struct.pack_into(
                    "<f",
                    out,
                    transform_offset + 20,
                    math.radians(float(transform.rotation[2])),
                )
                struct.pack_into("<f", out, transform_offset + 24, float(transform.scale[0]))
                struct.pack_into("<f", out, transform_offset + 28, float(transform.scale[1]))
                struct.pack_into("<f", out, transform_offset + 32, float(transform.scale[2]))

        collider_instances_offset = i32(record_offset + 12)
        if obj.collider_instances and collider_instances_offset > 0:
            for collider_index, collider in enumerate(obj.collider_instances):
                collider_offset = collider_instances_offset + (80 * collider_index)
                if collider_offset < 0 or (collider_offset + 80) > len(out):
                    break

                existing_param_count = i32(collider_offset + 0)
                existing_param_offset = i32(collider_offset + 4)
                existing_subpart1_offset = i32(collider_offset + 8)
                existing_subpart2_offset = i32(collider_offset + 12)

                struct.pack_into("<H", out, collider_offset + 20, int(collider.i_20) & 0xFFFF)
                struct.pack_into("<H", out, collider_offset + 22, int(collider.i_22) & 0xFFFF)
                struct.pack_into("<f", out, collider_offset + 24, float(collider.f_24))
                struct.pack_into("<f", out, collider_offset + 28, float(collider.f_28))
                out[collider_offset + 32 : collider_offset + 80] = _pack_transform(
                    collider.matrix.matrix
                )
                if collider.action_offset > 0:
                    struct.pack_into("<i", out, collider_offset + 16, int(collider.action_offset))

                if (
                    collider.havok_group_parameters
                    and existing_param_offset > 0
                    and existing_param_count == len(collider.havok_group_parameters)
                    and is_valid_offset(out, existing_param_offset, existing_param_count * 8)
                ):
                    for param_index, param in enumerate(collider.havok_group_parameters):
                        param_offset = existing_param_offset + (param_index * 8)
                        struct.pack_into("<i", out, param_offset + 0, int(param.param1))
                        struct.pack_into("<i", out, param_offset + 4, int(param.param2))

                if collider.subpart1 is not None and existing_subpart1_offset > 0:
                    _write_subpart_at(out, existing_subpart1_offset, collider.subpart1)
                if collider.subpart2 is not None and existing_subpart2_offset > 0:
                    _write_subpart_at(out, existing_subpart2_offset, collider.subpart2)

        if object_collision_meshes:
            hitbox_group_index = struct.unpack_from("<H", out, record_offset + 8)[0]
            if (
                hitbox_group_index != 0xFFFF
                and collision_group_count > 0
                and collision_group_offset > 0
                and hitbox_group_index < collision_group_count
            ):
                group_record_offset = collision_group_offset + (hitbox_group_index * 12)
                if group_record_offset >= 0 and (group_record_offset + 12) <= len(out):
                    group_collider_count = i32(group_record_offset + 4)
                    group_collider_offset = i32(group_record_offset + 8)
                    if group_collider_count > 0 and group_collider_offset > 0:
                        for collider_index, mesh_data in object_collision_meshes.items():
                            if collider_index < 0 or collider_index >= group_collider_count:
                                continue
                            vertices, normals, faces, collision_type = mesh_data
                            if not vertices or not faces or len(vertices) != len(normals):
                                continue
                            collider_record_offset = group_collider_offset + (collider_index * 40)
                            if collider_record_offset < 0 or (collider_record_offset + 40) > len(
                                out
                            ):
                                continue

                            havok_list_count = i32(collider_record_offset + 12)
                            havok_list_offset = i32(collider_record_offset + 16)
                            if is_old_version:
                                vertex_count_offset = collider_record_offset + 20
                                vertex_data_offset_offset = collider_record_offset + 24
                                face_count_offset = collider_record_offset + 28
                                face_data_offset_offset = collider_record_offset + 32
                            else:
                                vertex_count_offset = collider_record_offset + 24
                                vertex_data_offset_offset = collider_record_offset + 28
                                face_count_offset = collider_record_offset + 32
                                face_data_offset_offset = collider_record_offset + 36

                            existing_vertex_count = i32(vertex_count_offset)
                            existing_vertex_offset = i32(vertex_data_offset_offset)
                            existing_face_count = i32(face_count_offset)
                            existing_face_offset = i32(face_data_offset_offset)

                            write_vertices = vertices
                            write_normals = normals
                            write_faces = faces

                            if any(
                                face_index < 0
                                or face_index > 0xFFFF
                                or face_index >= len(write_vertices)
                                for face_index in write_faces
                            ):
                                continue

                            target_vertex_count = len(write_vertices)
                            target_face_count = len(write_faces)
                            can_write_in_place = (
                                existing_vertex_count == target_vertex_count
                                and existing_face_count == target_face_count
                                and existing_vertex_offset > 0
                                and existing_face_offset > 0
                                and is_valid_offset(
                                    out, existing_vertex_offset, target_vertex_count * 24
                                )
                                and is_valid_offset(
                                    out, existing_face_offset, target_face_count * 2
                                )
                            )

                            if can_write_in_place:
                                for vertex_index, vertex in enumerate(write_vertices):
                                    vx, vy, vz = vertex
                                    nx, ny, nz = write_normals[vertex_index]
                                    vertex_offset = existing_vertex_offset + (vertex_index * 24)
                                    struct.pack_into("<f", out, vertex_offset + 0, float(vx))
                                    struct.pack_into("<f", out, vertex_offset + 4, float(vy))
                                    struct.pack_into("<f", out, vertex_offset + 8, float(vz))
                                    struct.pack_into("<f", out, vertex_offset + 12, float(nx))
                                    struct.pack_into("<f", out, vertex_offset + 16, float(ny))
                                    struct.pack_into("<f", out, vertex_offset + 20, float(nz))

                                for face_index, face_vertex_index in enumerate(write_faces):
                                    struct.pack_into(
                                        "<H",
                                        out,
                                        existing_face_offset + (face_index * 2),
                                        int(face_vertex_index) & 0xFFFF,
                                    )
                            else:
                                appended_vertex_offset = _append_collision_vertices_to_buffer(
                                    out,
                                    write_vertices,
                                    write_normals,
                                )
                                appended_face_offset = _append_collision_faces_to_buffer(
                                    out,
                                    write_faces,
                                )
                                struct.pack_into(
                                    "<i", out, vertex_count_offset, int(target_vertex_count)
                                )
                                struct.pack_into(
                                    "<i",
                                    out,
                                    vertex_data_offset_offset,
                                    int(appended_vertex_offset),
                                )
                                struct.pack_into(
                                    "<i", out, face_count_offset, int(target_face_count)
                                )
                                struct.pack_into(
                                    "<i", out, face_data_offset_offset, int(appended_face_offset)
                                )

                            if (
                                havok_list_count > 0
                                and havok_list_offset > 0
                                and is_valid_offset(out, havok_list_offset, havok_list_count * 8)
                            ):
                                patched_hvk_offsets: dict[int, tuple[int, int]] = {}
                                for havok_group_index in range(havok_list_count):
                                    group_header_offset = havok_list_offset + (
                                        havok_group_index * 8
                                    )
                                    group_entry_count = i32(group_header_offset + 0)
                                    group_entry_offset = i32(group_header_offset + 4)
                                    if (
                                        group_entry_count <= 0
                                        or group_entry_offset <= 0
                                        or not is_valid_offset(
                                            out, group_entry_offset, group_entry_count * 40
                                        )
                                    ):
                                        continue
                                    for group_entry_index in range(group_entry_count):
                                        entry_offset = group_entry_offset + (group_entry_index * 40)
                                        if collision_type is not None:
                                            struct.pack_into(
                                                "<i",
                                                out,
                                                entry_offset + 0,
                                                int(collision_type),
                                            )

                                        hvk_size = i32(entry_offset + 16)
                                        hvk_offset = i32(entry_offset + 20)
                                        if hvk_offset in rewritten_hvk_cache:
                                            cached_offset, cached_size = rewritten_hvk_cache[
                                                hvk_offset
                                            ]
                                            struct.pack_into(
                                                "<i", out, entry_offset + 16, int(cached_size)
                                            )
                                            struct.pack_into(
                                                "<i", out, entry_offset + 20, int(cached_offset)
                                            )
                                            patched_hvk_offsets[hvk_offset] = (
                                                int(cached_offset),
                                                int(cached_size),
                                            )
                                            continue
                                        if hvk_offset in patched_hvk_offsets:
                                            cached_offset, cached_size = patched_hvk_offsets[
                                                hvk_offset
                                            ]
                                            struct.pack_into(
                                                "<i", out, entry_offset + 16, int(cached_size)
                                            )
                                            struct.pack_into(
                                                "<i", out, entry_offset + 20, int(cached_offset)
                                            )
                                            continue
                                        if (
                                            hvk_size <= 0
                                            or hvk_offset <= 0
                                            or not is_valid_offset(out, hvk_offset, hvk_size)
                                        ):
                                            continue

                                        hvk_data = bytearray(
                                            out[hvk_offset : hvk_offset + hvk_size]
                                        )
                                        rebuilt_hvk = _patch_hvk_mesh_data(
                                            hvk_data,
                                            vertices,
                                            faces,
                                        )
                                        if rebuilt_hvk is not None:
                                            new_hvk_offset = hvk_offset
                                            new_hvk_size = len(rebuilt_hvk)
                                            if new_hvk_size == hvk_size:
                                                out[hvk_offset : hvk_offset + hvk_size] = (
                                                    rebuilt_hvk
                                                )
                                            else:
                                                new_hvk_offset = _align_buffer_size(out, 16)
                                                out.extend(rebuilt_hvk)
                                            struct.pack_into(
                                                "<i", out, entry_offset + 16, int(new_hvk_size)
                                            )
                                            struct.pack_into(
                                                "<i", out, entry_offset + 20, int(new_hvk_offset)
                                            )
                                            patched_hvk_offsets[hvk_offset] = (
                                                int(new_hvk_offset),
                                                int(new_hvk_size),
                                            )
                                            rewritten_hvk_cache[hvk_offset] = (
                                                int(new_hvk_offset),
                                                int(new_hvk_size),
                                            )

    return bytes(out)


def _build_depots(fmp: FMPFile) -> tuple[list[str], list[str], list[str], list[str]]:
    depot_nsk: list[str] = []
    depot_emb: list[str] = []
    depot_emm: list[str] = []
    depot_ema: list[str] = []
    seen: set[tuple[int, str]] = set()

    def add(target: list[str], target_id: int, value: str) -> None:
        text = (value or "").strip()
        if not text:
            return
        key = (target_id, text)
        if key in seen:
            return
        seen.add(key)
        target.append(text)

    for obj in fmp.objects:
        for entity in obj.entities:
            visual = entity.visual
            if visual is None:
                continue
            add(depot_emb, 2, visual.emb_file)
            add(depot_ema, 4, visual.ema_file)
            for lod in visual.lods:
                add(depot_nsk, 1, lod.nsk_file)
                add(depot_emm, 3, lod.emm_file)

    return depot_nsk, depot_emb, depot_emm, depot_ema


def _build_default_settings_a(fmp: FMPFile) -> bytes:
    max_abs = 0.0
    for obj in fmp.objects:
        loc = obj.transform.matrix.to_translation()
        max_abs = max(max_abs, abs(float(loc.x)), abs(float(loc.y)), abs(float(loc.z)))
        if obj.instance_data is None:
            continue
        for group in obj.instance_data.groups:
            for transform in group.transforms:
                px, py, pz = transform.position
                max_abs = max(max_abs, abs(float(px)), abs(float(py)), abs(float(pz)))

    width = max(1000.0, (max_abs * 2.0) + 100.0)
    near_distance = 0.0
    far_distance = max(10000.0, width * 4.0)

    out = bytearray(140)
    struct.pack_into("<f", out, 0, float(width))
    struct.pack_into("<f", out, 4, float(width))
    struct.pack_into("<f", out, 8, float(width))
    struct.pack_into("<f", out, 128, float(near_distance))
    struct.pack_into("<f", out, 132, float(far_distance))
    return bytes(out)


def _build_default_settings_b() -> bytes:
    return bytes(508)


def _build_map_bytes(fmp: FMPFile, source_bytes: bytes | None = None) -> bytes:
    has_source_bytes = bool(source_bytes)
    if has_source_bytes:
        base = bytearray(source_bytes or b"")
        if len(base) < 112:
            base.extend(b"\x00" * (112 - len(base)))
    else:
        base = bytearray(112)
        struct.pack_into("<i", base, 16, 112)
        struct.pack_into("<i", base, 20, 252)
        base.extend(_build_default_settings_a(fmp))
        base.extend(_build_default_settings_b())
        if len(base) < 760:
            base.extend(b"\x00" * (760 - len(base)))

    struct.pack_into("<i", base, 0, int(FMP_SIGNATURE))
    struct.pack_into("<i", base, 4, int(fmp.version))
    struct.pack_into("<i", base, 8, int(fmp.i_08))
    struct.pack_into("<i", base, 12, int(fmp.i_12))
    struct.pack_into("<i", base, 96, int(fmp.i_96[0]))
    struct.pack_into("<i", base, 100, int(fmp.i_96[1]))
    struct.pack_into("<i", base, 104, int(fmp.i_96[2]))
    struct.pack_into("<i", base, 108, int(fmp.i_96[3]))
    if not has_source_bytes:
        struct.pack_into("<i", base, 24, 0)
        struct.pack_into("<i", base, 28, 0)
        struct.pack_into("<i", base, 32, 0)
        struct.pack_into("<i", base, 36, 0)
        struct.pack_into("<i", base, 40, 0)
        struct.pack_into("<i", base, 44, 0)
        struct.pack_into("<i", base, 56, 0)
        struct.pack_into("<i", base, 60, 0)

    string_offsets: dict[str, int] = {}

    def append_string(value: str) -> int:
        text = value or ""
        if text in string_offsets:
            return string_offsets[text]
        encoded = text.encode("utf-8", errors="ignore") + b"\x00"
        offset = _append_blob(base, encoded, alignment=4)
        string_offsets[text] = offset
        return offset

    dep_nsk, dep_emb, dep_emm, dep_ema = _build_depots(fmp)
    dep_nsk_idx = {value: index for index, value in enumerate(dep_nsk)}
    dep_emb_idx = {value: index for index, value in enumerate(dep_emb)}
    dep_emm_idx = {value: index for index, value in enumerate(dep_emm)}
    dep_ema_idx = {value: index for index, value in enumerate(dep_ema)}

    def serialize_bvh_node(node: FMPInstanceBVHNode | None) -> int:
        if node is None:
            return 0
        if node.children:
            child_offsets: list[int] = [0] * 8
            used_slots: set[int] = set()
            for child in node.children:
                child_offset = serialize_bvh_node(child)
                slot = child.index if 0 <= child.index < 8 else -1
                if slot < 0 or slot in used_slots:
                    for fallback_slot in range(8):
                        if fallback_slot not in used_slots:
                            slot = fallback_slot
                            break
                if slot < 0 or slot >= 8:
                    continue
                child_offsets[slot] = child_offset
                used_slots.add(slot)

            data = bytearray(48)
            data[0] = 0
            data[1] = int(node.flags) & 0xFF
            struct.pack_into(
                "<3f",
                data,
                4,
                float(node.center[0]),
                float(node.center[1]),
                float(node.center[2]),
            )
            for slot, child_offset in enumerate(child_offsets):
                struct.pack_into("<i", data, 16 + (slot * 4), int(child_offset))
            return _append_blob(base, bytes(data), alignment=4)

        group_indices = [int(idx) for idx in node.group_indices if idx >= 0]
        data = bytearray(4 + (len(group_indices) * 2))
        data[0] = 1
        data[1] = int(node.flags) & 0xFF
        struct.pack_into("<H", data, 2, len(group_indices))
        for index, group_index in enumerate(group_indices):
            struct.pack_into("<H", data, 4 + (index * 2), group_index & 0xFFFF)
        return _append_blob(base, bytes(data), alignment=4)

    def serialize_instance_data(instance_data: FMPInstanceData | None) -> int:
        if instance_data is None:
            return 0

        transform_chunks: list[bytes] = []
        group_chunks: list[bytes] = []
        transform_index = 0
        for group in instance_data.groups:
            transforms = group.transforms or [FMPInstanceTransform()]
            first_index = transform_index
            for transform in transforms:
                rot_x = math.radians(float(transform.rotation[0]))
                rot_y = math.radians(float(transform.rotation[1]))
                rot_z = math.radians(float(transform.rotation[2]))
                transform_chunks.append(
                    struct.pack(
                        "<9f",
                        float(transform.position[0]),
                        float(transform.position[1]),
                        float(transform.position[2]),
                        rot_x,
                        rot_y,
                        rot_z,
                        float(transform.scale[0]),
                        float(transform.scale[1]),
                        float(transform.scale[2]),
                    )
                )
                transform_index += 1

            group_chunks.append(
                struct.pack(
                    "<10f2i",
                    float(group.center[0]),
                    float(group.center[1]),
                    float(group.center[2]),
                    float(group.max_distance),
                    float(group.min_bounds[0]),
                    float(group.min_bounds[1]),
                    float(group.min_bounds[2]),
                    float(group.max_bounds[0]),
                    float(group.max_bounds[1]),
                    float(group.max_bounds[2]),
                    len(transforms),
                    first_index,
                )
            )

        node_transform_offset = (
            _append_blob(base, b"".join(transform_chunks), alignment=4) if transform_chunks else 0
        )
        node_offset = _append_blob(base, b"".join(group_chunks), alignment=4) if group_chunks else 0

        bvh_root = instance_data.bvh_root or _make_leaf_bvh(len(instance_data.groups))
        hierarchy_node_offset = serialize_bvh_node(bvh_root) if bvh_root is not None else 0

        header = struct.pack(
            "<5i",
            len(transform_chunks),
            len(group_chunks),
            node_offset,
            hierarchy_node_offset,
            node_transform_offset,
        )
        return _append_blob(base, header, alignment=4)

    def serialize_visual(visual: FMPVisual | None) -> int:
        if visual is None:
            return 0
        lods = visual.lods or []
        lod_count = len(lods)

        name_offset = append_string(visual.name)
        emb_index = dep_emb_idx.get(visual.emb_file, -1)
        ema_index = dep_ema_idx.get(visual.ema_file, -1)

        nsk_value: int = -1
        emm_value: int = -1
        dist_scalar: float = 0.0
        dist_offset: int = 0
        if lod_count == 1:
            nsk_value = dep_nsk_idx.get(lods[0].nsk_file, -1)
            emm_value = dep_emm_idx.get(lods[0].emm_file, -1)
            dist_scalar = float(lods[0].distance)
        elif lod_count > 1:
            nsk_table = struct.pack(
                f"<{lod_count}i",
                *[dep_nsk_idx.get(lod.nsk_file, -1) for lod in lods],
            )
            emm_table = struct.pack(
                f"<{lod_count}i",
                *[dep_emm_idx.get(lod.emm_file, -1) for lod in lods],
            )
            dist_table = struct.pack(
                f"<{lod_count}f",
                *[float(lod.distance) for lod in lods],
            )
            nsk_value = _append_blob(base, nsk_table, alignment=4)
            emm_value = _append_blob(base, emm_table, alignment=4)
            dist_offset = _append_blob(base, dist_table, alignment=4)

        out = bytearray(52)
        struct.pack_into("<i", out, 0, int(name_offset))
        struct.pack_into("<i", out, 4, int(visual.i_04))
        struct.pack_into("<i", out, 8, int(lod_count))
        struct.pack_into("<i", out, 12, int(nsk_value))
        struct.pack_into("<i", out, 16, int(emb_index))
        struct.pack_into("<i", out, 20, int(emm_value))
        struct.pack_into("<i", out, 24, int(visual.i_24))
        struct.pack_into("<i", out, 28, int(visual.i_28))
        struct.pack_into("<i", out, 32, int(ema_index))
        struct.pack_into("<i", out, 36, int(visual.i_36))
        struct.pack_into("<f", out, 40, float(visual.f_40))
        struct.pack_into("<f", out, 44, float(visual.f_44))
        if lod_count <= 1:
            struct.pack_into("<f", out, 48, float(dist_scalar))
        else:
            struct.pack_into("<i", out, 48, int(dist_offset))

        return _append_blob(base, bytes(out), alignment=4)

    object_records: list[bytes] = []
    for object_index, obj in enumerate(fmp.objects):
        obj.idx = object_index
        visual_offsets: list[int] = [serialize_visual(entity.visual) for entity in obj.entities]

        entity_blob = bytearray()
        for entity_index, entity in enumerate(obj.entities):
            visual_offset = (
                visual_offsets[entity_index] if entity_index < len(visual_offsets) else 0
            )
            entity_blob.extend(struct.pack("<i", int(visual_offset)))
            entity_blob.extend(struct.pack("<i", int(entity.i_04)))
            entity_blob.extend(_pack_transform(entity.transform.matrix))

        entity_offset = _append_blob(base, bytes(entity_blob), alignment=4) if entity_blob else 0
        hierarchy_offset = serialize_instance_data(obj.instance_data)
        name_offset = append_string(obj.name)
        hitbox_group_index = int(obj.hitbox_group_index) if has_source_bytes else 0xFFFF
        hitbox_instances_offset = int(obj.hitbox_instances_offset) if has_source_bytes else 0
        action_offset = int(obj.action_offset) if has_source_bytes else 0

        object_record = bytearray(84)
        struct.pack_into("<i", object_record, 0, int(name_offset))
        struct.pack_into("<H", object_record, 4, int(obj.i_04) & 0xFFFF)
        struct.pack_into("<H", object_record, 6, int(obj.initial_entity_index) & 0xFFFF)
        struct.pack_into("<H", object_record, 8, int(hitbox_group_index) & 0xFFFF)
        struct.pack_into("<i", object_record, 12, int(hitbox_instances_offset))
        struct.pack_into("<i", object_record, 16, int(action_offset))
        struct.pack_into("<H", object_record, 10, int(obj.flags) & 0xFFFF)
        struct.pack_into("<i", object_record, 20, int(len(obj.entities)))
        struct.pack_into("<i", object_record, 24, int(entity_offset))
        struct.pack_into("<i", object_record, 28, int(hierarchy_offset))
        struct.pack_into("<f", object_record, 32, float(obj.f_32))
        object_record[36:84] = _pack_transform(obj.transform.matrix)
        object_records.append(bytes(object_record))

    object_offset = (
        _append_blob(base, b"".join(object_records), alignment=4) if object_records else 0
    )

    def serialize_depot(strings: list[str]) -> int:
        if not strings:
            return 0
        table = struct.pack(f"<{len(strings)}i", *[append_string(value) for value in strings])
        return _append_blob(base, table, alignment=4)

    dep1_offset = serialize_depot(dep_nsk)
    dep2_offset = serialize_depot(dep_emb)
    dep3_offset = serialize_depot(dep_emm)
    dep4_offset = serialize_depot(dep_ema)

    struct.pack_into("<i", base, 48, int(len(fmp.objects)))
    struct.pack_into("<i", base, 52, int(object_offset))
    struct.pack_into("<i", base, 64, int(len(dep_nsk)))
    struct.pack_into("<i", base, 68, int(dep1_offset))
    struct.pack_into("<i", base, 72, int(len(dep_emb)))
    struct.pack_into("<i", base, 76, int(dep2_offset))
    struct.pack_into("<i", base, 80, int(len(dep_emm)))
    struct.pack_into("<i", base, 84, int(dep3_offset))
    struct.pack_into("<i", base, 88, int(len(dep_ema)))
    struct.pack_into("<i", base, 92, int(dep4_offset))

    return bytes(base)


def export_map(
    filepath: str,
    map_root: bpy.types.Object | None = None,
    export_collision_meshes: bool = False,
    export_linked_nsk: bool = False,
    warn: Callable[[str], None] | None = None,
) -> tuple[bool, str | None]:
    resolved_map_root = map_root or _get_map_root()
    plan = (
        collect_map_export_plan(
            resolved_map_root,
            warn=warn,
            include_collision_meshes=export_collision_meshes,
        )
        if resolved_map_root
        else collect_map_export_plan_from_active(
            warn=warn,
            include_collision_meshes=export_collision_meshes,
        )
    )
    if plan is None:
        return False, "Select a MAP root empty (with imported MAP hierarchy) to export."

    source_bytes = b""
    source_fmp = FMPFile(
        version=max(plan.version, 0),
        i_08=int(plan.i_08),
        i_12=int(plan.i_12),
        i_96=(
            int(plan.i_96[0]),
            int(plan.i_96[1]),
            int(plan.i_96[2]),
            int(plan.i_96[3]),
        ),
    )
    source_path = (plan.source_path or "").strip()
    if source_path and os.path.isfile(source_path):
        source_bytes = Path(source_path).read_bytes()
        try:
            source_fmp = parse_fmp_bytes(source_bytes)
        except ValueError:
            if warn:
                warn(
                    "Source MAP parsing failed; continuing with rebuilt layout from current scene."
                )
    elif source_path and warn:
        warn(f"Source MAP file was not found: {source_path}. Exporting a rebuilt MAP layout.")

    merged_fmp = _merge_plan_into_source(plan, source_fmp)
    collision_meshes_by_object: dict[int, dict[int, CollisionMeshData]] = {}
    found_collision_meshes = False
    if export_collision_meshes:
        object_indices_by_name: dict[str, list[int]] = defaultdict(list)
        for merged_object_index, merged_object in enumerate(merged_fmp.objects):
            object_indices_by_name[str(merged_object.name)].append(int(merged_object_index))

        for plan_object in plan.objects:
            if not plan_object.collision_meshes:
                continue
            found_collision_meshes = True

            target_object_index: int | None = None
            if plan_object.index is not None and 0 <= int(plan_object.index) < len(
                merged_fmp.objects
            ):
                target_object_index = int(plan_object.index)
            else:
                name_matches = object_indices_by_name.get(str(plan_object.name), [])
                if len(name_matches) == 1:
                    target_object_index = int(name_matches[0])

            if target_object_index is None:
                continue

            existing = collision_meshes_by_object.get(target_object_index)
            if existing is None:
                collision_meshes_by_object[target_object_index] = dict(plan_object.collision_meshes)
            else:
                existing.update(plan_object.collision_meshes)

        if warn and not found_collision_meshes:
            warn(
                "Collision mesh export is enabled, but no collision mesh objects were found. "
                "Make sure edited meshes still have custom property 'fmp_collision_mesh'."
            )

    out_bytes: bytes
    if source_bytes and _can_reuse_source_layout(source_fmp, merged_fmp):
        try:
            out_bytes = _write_into_source_layout(
                source_bytes,
                merged_fmp,
                collision_meshes_by_object=collision_meshes_by_object,
            )
        except (RuntimeError, OSError, ValueError, TypeError, struct.error) as exc:
            if warn:
                warn(
                    "MAP in-place patch failed; falling back to rebuilt layout "
                    f"(collision/Havok writeback skipped). Reason: {exc}"
                )
            out_bytes = _build_map_bytes(merged_fmp, source_bytes=source_bytes)
            if export_collision_meshes and collision_meshes_by_object:
                try:
                    out_bytes = _write_into_source_layout(
                        out_bytes,
                        merged_fmp,
                        collision_meshes_by_object=collision_meshes_by_object,
                    )
                except (RuntimeError, OSError, ValueError, TypeError, struct.error) as exc:
                    if warn:
                        warn(
                            "Collision/Havok writeback on rebuilt MAP failed after in-place "
                            f"patch error. Reason: {exc}"
                        )
    else:
        if export_collision_meshes and collision_meshes_by_object and warn:
            warn(
                "MAP layout could not be patched in-place from source; rebuilding layout and "
                "attempting collision/Havok writeback on rebuilt bytes."
            )
        out_bytes = _build_map_bytes(merged_fmp, source_bytes=source_bytes)
        if export_collision_meshes and collision_meshes_by_object:
            try:
                out_bytes = _write_into_source_layout(
                    out_bytes,
                    merged_fmp,
                    collision_meshes_by_object=collision_meshes_by_object,
                )
            except (RuntimeError, OSError, ValueError, TypeError, struct.error) as exc:
                if warn:
                    warn(f"Collision/Havok writeback on rebuilt MAP failed. Reason: {exc}")

    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(out_bytes)

    if export_linked_nsk:
        exported_count, failed_count = _export_linked_nsk_files(
            str(output_path),
            resolved_map_root,
            warn=warn,
        )
        if warn:
            warn(f"Linked NSK export: exported={exported_count}, failed={failed_count}.")

    return True, None


__all__ = [
    "FMPExportEntity",
    "FMPExportInstance",
    "FMPExportLOD",
    "FMPExportObject",
    "FMPExportPlan",
    "collect_map_export_plan",
    "collect_map_export_plan_from_active",
    "export_map",
]
