from __future__ import annotations

import contextlib
import math
import re
import struct
from pathlib import Path

import bpy
import mathutils

from ...utils import remove_unused_vertex_groups
from ..EAN.exporter_char import _build_skeleton_from_armature
from ..EMD import EMD_File, EMD_Mesh, EMD_Model
from ..EMD.exporter import _build_emd_bytes, _build_submeshes_from_object
from ..ESK import ESK_SIGNATURE, ESK_Bone
from ..ESK.exporter import (
    _pack_relative_transforms,
    _read_skeleton_layout,
)
from .NSK import NSK_EMD_OFFSET_ADDRESS

_BLENDER_DUPLICATE_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.\d{3}$")


def _strip_blender_duplicate_suffix(name: str) -> str:
    match = _BLENDER_DUPLICATE_SUFFIX_RE.match(name)
    if match is None:
        return name
    return str(match.group("base"))


def _strip_suffix(name: str, suffix: str) -> str:
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _empty_bone_name(name: str) -> str:
    normalized_name = _strip_blender_duplicate_suffix(name)
    if normalized_name.endswith("_model"):
        return _strip_suffix(normalized_name, "_model").strip()
    if normalized_name.endswith("_mesh"):
        return _strip_suffix(normalized_name, "_mesh").strip()
    return ""


def _empty_kind(name: str) -> str | None:
    normalized_name = _strip_blender_duplicate_suffix(name)
    if normalized_name.endswith("_model"):
        return "model"
    if normalized_name.endswith("_mesh"):
        return "mesh"
    return None


def _matrix_in_armature_space(obj: bpy.types.Object, arm_obj: bpy.types.Object) -> mathutils.Matrix:
    return arm_obj.matrix_world.inverted_safe() @ obj.matrix_world


def _empty_matrix_in_nsk_space(
    obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
) -> mathutils.Matrix:
    return _matrix_in_armature_space(obj, arm_obj)


def _direct_mesh_matrix_in_nsk_space(
    obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
) -> mathutils.Matrix:
    matrix = _matrix_in_armature_space(obj, arm_obj)

    # No-empty flow needs to compensate for the display +90deg X armature orientation.
    rx, ry, rz = arm_obj.rotation_euler
    if abs(rx - math.radians(90.0)) < 1e-4 and abs(ry) < 1e-4 and abs(rz) < 1e-4:
        axis_fix = mathutils.Matrix.Rotation(math.radians(90.0), 4, "X")
        return axis_fix @ matrix

    return matrix


def _rebuild_child_sibling_indices(bones: list[ESK_Bone]) -> None:
    for bone in bones:
        bone.child_index = -1
        bone.sibling_index = -1

    children_by_parent: dict[int, list[int]] = {}
    for bone in bones[1:]:
        if bone.parent_index < 0 or bone.parent_index >= len(bones):
            bone.parent_index = 0 if bones else -1
        children_by_parent.setdefault(bone.parent_index, []).append(bone.index)

    for parent_index, child_indices in children_by_parent.items():
        if 0 <= parent_index < len(bones) and child_indices:
            bones[parent_index].child_index = child_indices[0]
            for idx, child_index in enumerate(child_indices):
                bones[child_index].sibling_index = (
                    child_indices[idx + 1] if idx + 1 < len(child_indices) else -1
                )


def _collect_empty_bone_specs(
    arm_obj: bpy.types.Object,
) -> list[tuple[str, str | None, mathutils.Matrix]]:
    specs_with_depth: list[tuple[int, int, str, str | None, mathutils.Matrix]] = []
    for child in arm_obj.children_recursive:
        if child.type != "EMPTY":
            continue

        empty_kind = _empty_kind(child.name)
        if empty_kind is None:
            continue

        bone_name = _empty_bone_name(child.name)
        if not bone_name:
            continue

        depth = 0
        parent_bone_name: str | None = None
        parent = child.parent
        while parent is not None and parent is not arm_obj:
            depth += 1
            if parent.type == "EMPTY":
                empty_parent_bone_name = _empty_bone_name(parent.name)
                if empty_parent_bone_name:
                    parent_bone_name = empty_parent_bone_name
                    break
            parent = parent.parent

        kind_priority = 0 if empty_kind == "model" else 1
        specs_with_depth.append(
            (
                depth,
                kind_priority,
                bone_name,
                parent_bone_name,
                _empty_matrix_in_nsk_space(child, arm_obj),
            )
        )

    specs_with_depth.sort(key=lambda item: (item[0], item[1], item[2]))

    # If both *_model and *_mesh map to the same stripped name, prefer *_model.
    deduped: list[tuple[str, str | None, mathutils.Matrix]] = []
    seen_names: set[str] = set()
    for _depth, _priority, bone_name, parent_name, matrix in specs_with_depth:
        if bone_name in seen_names:
            continue
        seen_names.add(bone_name)
        deduped.append((bone_name, parent_name, matrix))
    return deduped


def _add_missing_bones_from_specs(
    esk,
    specs: list[tuple[str, str | None, mathutils.Matrix]],
) -> None:
    if not specs:
        return

    index_by_name = {bone.name: bone.index for bone in esk.bones if getattr(bone, "name", None)}
    world_cache: dict[int, mathutils.Matrix] = {}

    def compute_world(index: int) -> mathutils.Matrix:
        if index in world_cache:
            return world_cache[index]
        bone = esk.bones[index]
        matrix = bone.matrix.copy()
        if 0 <= bone.parent_index < len(esk.bones) and esk.bones[bone.parent_index] is not bone:
            matrix = compute_world(bone.parent_index) @ matrix
        world_cache[index] = matrix
        return matrix

    for bone_name, parent_name, target_world in specs:
        if bone_name in index_by_name:
            continue

        parent_index = index_by_name.get(parent_name or "", 0 if esk.bones else -1)
        if parent_index >= 0 and parent_index < len(esk.bones):
            parent_world = compute_world(parent_index)
            local_matrix = parent_world.inverted_safe() @ target_world
        else:
            parent_index = -1
            local_matrix = target_world.copy()

        bone_index = len(esk.bones)
        esk.bones.append(
            ESK_Bone(
                bone_name,
                bone_index,
                local_matrix,
                parent_index,
                -1,
                -1,
            )
        )
        index_by_name[bone_name] = bone_index
        world_cache.clear()

    _rebuild_child_sibling_indices(esk.bones)


def _add_missing_empty_bones_to_skeleton(esk, arm_obj: bpy.types.Object) -> None:
    specs = _collect_empty_bone_specs(arm_obj)
    _add_missing_bones_from_specs(esk, specs)


def _collect_direct_mesh_bone_specs(
    arm_obj: bpy.types.Object,
) -> list[tuple[str, str | None, mathutils.Matrix]]:
    specs: list[tuple[str, str | None, mathutils.Matrix]] = []
    seen_names: set[str] = set()

    mesh_sources = [child for child in arm_obj.children_recursive if child.type == "MESH"]

    for mesh_obj in mesh_sources:
        bone_name = (mesh_obj.name or "").strip()
        if not bone_name or bone_name in seen_names:
            continue
        seen_names.add(bone_name)
        specs.append((bone_name, None, _direct_mesh_matrix_in_nsk_space(mesh_obj, arm_obj)))

    return specs


def _add_missing_direct_mesh_bones_to_skeleton(esk, arm_obj: bpy.types.Object) -> None:
    specs = _collect_direct_mesh_bone_specs(arm_obj)
    _add_missing_bones_from_specs(esk, specs)


def _has_structure_empties(arm_obj: bpy.types.Object) -> bool:
    for child in arm_obj.children_recursive:
        if child.type != "EMPTY":
            continue
        if _empty_kind(child.name) is not None:
            return True
    return False


def _update_mesh_bounds(mesh: EMD_Mesh) -> None:
    all_positions = [v.pos for sub in mesh.submeshes for v in sub.vertices]
    if not all_positions:
        return
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


def _collect_model_empty_targets(arm_obj: bpy.types.Object) -> dict[str, mathutils.Matrix]:
    targets: dict[str, mathutils.Matrix] = {}
    for child in arm_obj.children_recursive:
        if child.type != "EMPTY":
            continue

        empty_kind = _empty_kind(child.name)
        if empty_kind is None:
            continue

        bone_name = _empty_bone_name(child.name)
        if not bone_name:
            continue
        if empty_kind == "model":
            # Preferred source for model-bone placement.
            targets[bone_name] = _empty_matrix_in_nsk_space(child, arm_obj)
        elif bone_name not in targets:
            # Fallback if only *_mesh exists.
            targets[bone_name] = _empty_matrix_in_nsk_space(child, arm_obj)
    return targets


def _collect_direct_mesh_targets(arm_obj: bpy.types.Object) -> dict[str, mathutils.Matrix]:
    targets: dict[str, mathutils.Matrix] = {}
    for child in arm_obj.children:
        if child.type != "MESH":
            continue
        bone_name = (child.name or "").strip()
        if not bone_name:
            continue
        targets[bone_name] = _direct_mesh_matrix_in_nsk_space(child, arm_obj)
    return targets


def _apply_model_empty_transforms_to_skeleton(esk, arm_obj: bpy.types.Object) -> None:
    target_world_by_name = _collect_model_empty_targets(arm_obj)
    if not target_world_by_name:
        # Fallback: when no structure empties exist, allow direct mesh transforms
        # to drive matching bone transforms.
        target_world_by_name = _collect_direct_mesh_targets(arm_obj)
    if not target_world_by_name:
        return

    index_by_name = {bone.name: bone.index for bone in esk.bones if getattr(bone, "name", None)}
    target_world_by_index = {
        index_by_name[name]: matrix
        for name, matrix in target_world_by_name.items()
        if name in index_by_name
    }
    if not target_world_by_index:
        return

    depth_cache: dict[int, int] = {}

    def bone_depth(index: int) -> int:
        if index in depth_cache:
            return depth_cache[index]
        depth = 0
        visited = set()
        parent_index = esk.bones[index].parent_index
        while 0 <= parent_index < len(esk.bones) and parent_index not in visited:
            visited.add(parent_index)
            depth += 1
            parent_index = esk.bones[parent_index].parent_index
        depth_cache[index] = depth
        return depth

    world_cache: dict[int, mathutils.Matrix] = {}

    def compute_world(index: int) -> mathutils.Matrix:
        if index in world_cache:
            return world_cache[index]
        bone = esk.bones[index]
        matrix = bone.matrix.copy()
        if 0 <= bone.parent_index < len(esk.bones) and esk.bones[bone.parent_index] is not bone:
            matrix = compute_world(bone.parent_index) @ matrix
        world_cache[index] = matrix
        return matrix

    for bone_index in sorted(target_world_by_index.keys(), key=bone_depth):
        bone = esk.bones[bone_index]
        parent_index = bone.parent_index
        parent_world = (
            compute_world(parent_index)
            if 0 <= parent_index < len(esk.bones)
            else mathutils.Matrix.Identity(4)
        )
        local_scale = bone.matrix.decompose()[2]
        target_world = target_world_by_index[bone_index]
        target_local = parent_world.inverted_safe() @ target_world
        loc, rot, _scale = target_local.decompose()
        bone.matrix = mathutils.Matrix.LocRotScale(loc, rot, local_scale)
        world_cache.clear()


def _build_mesh_from_objects(
    model_name: str,
    mesh_name: str,
    mesh_objects: list[bpy.types.Object],
    arm_obj: bpy.types.Object,
    synthetic_model_bone_names: set[str] | None = None,
) -> EMD_Mesh | None:
    mesh = EMD_Mesh()
    mesh.name = mesh_name
    arm_bone_names = set()
    if arm_obj and getattr(arm_obj, "data", None) and hasattr(arm_obj.data, "bones"):
        arm_bone_names = {bone.name for bone in arm_obj.data.bones}
    model_has_matching_bone = model_name in arm_bone_names or (
        synthetic_model_bone_names is not None and model_name in synthetic_model_bone_names
    )

    for mesh_obj in mesh_objects:
        if mesh_obj.type != "MESH":
            continue
        remove_unused_vertex_groups(mesh_obj)
        has_local_transform = not mesh_obj.matrix_local.is_identity
        is_direct_armature_child = mesh_obj.parent is arm_obj
        is_structure_child = bool(mesh_obj.parent and mesh_obj.parent.type == "EMPTY")
        bake_local_transform = has_local_transform and (
            (is_direct_armature_child and not model_has_matching_bone) or is_structure_child
        )
        if bake_local_transform:
            temp_mesh = mesh_obj.data.copy()
            try:
                temp_mesh.transform(mesh_obj.matrix_local)
                mesh.submeshes.extend(
                    _build_submeshes_from_object(mesh_obj, arm_obj, mesh_data=temp_mesh)
                )
            finally:
                bpy.data.meshes.remove(temp_mesh)
        else:
            mesh.submeshes.extend(_build_submeshes_from_object(mesh_obj, arm_obj))
    if not mesh.submeshes:
        return None
    _update_mesh_bounds(mesh)
    return mesh


def _build_emd_from_armature_hierarchy(arm_obj: bpy.types.Object) -> EMD_File:
    emd = EMD_File()
    emd.version = int(arm_obj.get("emd_file_version", 0x201))
    has_structure_empties = _has_structure_empties(arm_obj)

    used_mesh_ptrs: set[int] = set()
    model_entries: list[tuple[str, list[tuple[str, list[bpy.types.Object]]]]] = []
    synthetic_model_bone_names: set[str] = set()

    for child in arm_obj.children:
        if child.type != "EMPTY":
            continue

        model_name = _strip_suffix(child.name, "_model") or child.name
        mesh_entries: list[tuple[str, list[bpy.types.Object]]] = []

        for model_child in child.children:
            if model_child.type == "EMPTY":
                mesh_name = _strip_suffix(model_child.name, "_mesh") or model_child.name
                mesh_objects = [obj for obj in model_child.children if obj.type == "MESH"]
                if mesh_objects:
                    mesh_entries.append((mesh_name, mesh_objects))
                    used_mesh_ptrs.update(obj.as_pointer() for obj in mesh_objects)
            elif model_child.type == "MESH":
                mesh_entries.append((model_child.name, [model_child]))
                used_mesh_ptrs.add(model_child.as_pointer())

        if mesh_entries:
            model_entries.append((model_name, mesh_entries))

    for child in arm_obj.children:
        if child.type != "MESH":
            continue
        ptr = child.as_pointer()
        if ptr in used_mesh_ptrs:
            continue
        model_entries.append((child.name, [(child.name, [child])]))
        used_mesh_ptrs.add(ptr)
        if not has_structure_empties and child.name:
            synthetic_model_bone_names.add(child.name)

    if not model_entries:
        for child in arm_obj.children_recursive:
            if child.type != "MESH":
                continue
            ptr = child.as_pointer()
            if ptr in used_mesh_ptrs:
                continue
            model_entries.append((child.name, [(child.name, [child])]))
            used_mesh_ptrs.add(ptr)
            if not has_structure_empties and child.name:
                synthetic_model_bone_names.add(child.name)

    first_mesh_obj = None
    for _model_name, mesh_entries in model_entries:
        for _mesh_name, mesh_objects in mesh_entries:
            if mesh_objects:
                first_mesh_obj = mesh_objects[0]
                break
        if first_mesh_obj is not None:
            break
    if first_mesh_obj is not None and "emd_file_version" in first_mesh_obj:
        with contextlib.suppress(TypeError, ValueError):
            emd.version = int(first_mesh_obj.get("emd_file_version"))

    for model_name, mesh_entries in model_entries:
        model = EMD_Model()
        model.name = model_name
        for mesh_name, mesh_objects in mesh_entries:
            built_mesh = _build_mesh_from_objects(
                model_name,
                mesh_name,
                mesh_objects,
                arm_obj,
                synthetic_model_bone_names=synthetic_model_bone_names,
            )
            if built_mesh is not None:
                model.meshes.append(built_mesh)
        if model.meshes:
            emd.models.append(model)

    return emd


def _build_esk_bytes_from_armature(arm_obj: bpy.types.Object) -> bytes:
    esk, _, _ = _build_skeleton_from_armature(arm_obj)

    root_name_original = str(arm_obj.get("esk_root_name_original", "")).strip()
    if root_name_original and esk.bones:
        esk.bones[0].name = root_name_original

    _add_missing_empty_bones_to_skeleton(esk, arm_obj)
    _add_missing_direct_mesh_bones_to_skeleton(esk, arm_obj)
    _apply_model_empty_transforms_to_skeleton(esk, arm_obj)

    source_path = str(arm_obj.get("esk_source_path", "")).strip()
    source_esk_data = _read_source_esk_blob(source_path)
    if source_esk_data:
        templated = _build_esk_bytes_from_template(source_esk_data, esk.bones)
        if templated is not None:
            return templated

    version = int(arm_obj.get("esk_version", 37568)) & 0xFFFF
    i10 = int(arm_obj.get("esk_i10", 0)) & 0xFFFF
    i12 = int(arm_obj.get("esk_i12", 0)) & 0xFFFFFFFF
    i24 = int(arm_obj.get("esk_i24", 0)) & 0xFFFFFFFF
    default_skel_flag = 0 if arm_obj.get("ean_source") else 1
    skeleton_flag = int(arm_obj.get("esk_skeleton_flag", default_skel_flag))
    skeleton_id = int(arm_obj.get("esk_skeleton_id", 0))

    skeleton_bytes = _build_nsk_skeleton_bytes(
        esk.bones,
        skeleton_flag=skeleton_flag,
        skeleton_id=skeleton_id,
    )

    out = bytearray()
    out.extend(struct.pack("<I", ESK_SIGNATURE))
    out.extend(struct.pack("<H", 0xFFFE))
    out.extend(struct.pack("<H", 0x001C))
    out.extend(struct.pack("<H", version))
    out.extend(struct.pack("<H", i10))
    out.extend(struct.pack("<I", i12))
    out.extend(struct.pack("<I", 32))  # Offset to skeleton
    out.extend(struct.pack("<I", 0))  # NSK EMD offset placeholder (filled later)
    out.extend(struct.pack("<I", i24))
    out.extend(struct.pack("<I", 0))
    out.extend(skeleton_bytes)
    return bytes(out)


def _build_nsk_skeleton_bytes(
    bones: list[ESK_Bone],
    skeleton_flag: int = 0,
    skeleton_id: int = 0,
) -> bytes:
    bone_count = len(bones)
    header_size = 36
    index_rel = header_size
    name_rel = index_rel + bone_count * 8
    string_off = name_rel + bone_count * 4

    strings = bytearray()
    name_offsets = []
    for bone in bones:
        name_offsets.append(len(strings))
        strings.extend(bone.name.encode("ascii", "ignore") + b"\x00")

    data = bytearray()
    data.extend(struct.pack("<h", bone_count))
    data.extend(struct.pack("<h", int(skeleton_flag)))
    data.extend(struct.pack("<I", index_rel))
    data.extend(struct.pack("<I", name_rel))
    data.extend(struct.pack("<I", 0))  # relative transforms offset
    data.extend(struct.pack("<I", 0))  # absolute transforms offset (unused for NSK)
    data.extend(struct.pack("<I", 0))  # IK offset
    data.extend(struct.pack("<I", 0))  # extra values offset
    data.extend(struct.pack("<Q", int(skeleton_id) & 0xFFFFFFFFFFFFFFFF))

    for bone in bones:
        data.extend(
            struct.pack("<hhhH", bone.parent_index, bone.child_index, bone.sibling_index, 0)
        )

    for offset in name_offsets:
        data.extend(struct.pack("<I", string_off + offset))
    data.extend(strings)

    rel_off = (len(data) + 15) & ~15
    if rel_off > len(data):
        data.extend(b"\x00" * (rel_off - len(data)))
    struct.pack_into("<I", data, 12, rel_off)
    data.extend(_pack_relative_transforms(bones))

    extra_off = len(data)
    struct.pack_into("<I", data, 24, extra_off)
    # NSK keeps per-bone extra values; default tuple is (0, 0, 65535, 0).
    for _ in bones:
        data.extend(struct.pack("<HHHH", 0, 0, 0xFFFF, 0))

    return bytes(data)


def _read_source_esk_blob(source_path: str) -> bytes | None:
    if not source_path:
        return None
    src = Path(source_path)
    if not src.is_file():
        return None

    data = src.read_bytes()
    if len(data) < 32 or data[:4] != b"#ESK":
        return None

    # Source may be either .esk or .nsk. For .nsk, keep only the ESK section.
    if len(data) >= NSK_EMD_OFFSET_ADDRESS + 4:
        emd_off = struct.unpack_from("<I", data, NSK_EMD_OFFSET_ADDRESS)[0]
        if 0 < emd_off <= len(data) and data[emd_off : emd_off + 4] == b"#EMD":
            return data[:emd_off]

    return data


def _build_esk_bytes_from_template(source_esk_data: bytes, bones) -> bytes | None:
    layout = _read_skeleton_layout(source_esk_data)
    if layout is None:
        return None

    bone_count = int(layout["bone_count"])
    if bone_count != len(bones):
        return None

    source_names = list(layout["names"])
    if source_names != [bone.name for bone in bones]:
        return None

    rel_blob = _pack_relative_transforms(bones)
    rel_off = int(layout["rel_off"])
    if rel_off + len(rel_blob) > len(source_esk_data):
        return None

    out = bytearray(source_esk_data)
    out[rel_off : rel_off + len(rel_blob)] = rel_blob
    # NSK export keeps only relative transforms; clear absolute-transform offset.
    skel_off = struct.unpack_from("<I", out, 16)[0]
    if skel_off + 20 <= len(out):
        struct.pack_into("<I", out, skel_off + 16, 0)
    return bytes(out)


def export_nsk(filepath: str, arm_obj: bpy.types.Object) -> tuple[bool, str | None]:
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return False, "Select an armature to export."

    try:
        emd = _build_emd_from_armature_hierarchy(arm_obj)
        if not emd.models:
            return False, "No mesh models found under this armature."

        esk_bytes = _build_esk_bytes_from_armature(arm_obj)
        emd_bytes = _build_emd_bytes(emd)

        emd_offset = (len(esk_bytes) + 15) & ~15
        out = bytearray(esk_bytes)
        if len(out) < emd_offset:
            out.extend(b"\x00" * (emd_offset - len(out)))
        out.extend(emd_bytes)

        if len(out) < NSK_EMD_OFFSET_ADDRESS + 4:
            return False, "Generated NSK data is invalid."
        struct.pack_into("<I", out, NSK_EMD_OFFSET_ADDRESS, emd_offset)

        Path(filepath).write_bytes(out)
        return True, None
    except (RuntimeError, OSError, ValueError, TypeError, struct.error) as exc:
        import traceback

        traceback.print_exc()
        return False, f"Unexpected error while exporting NSK: {exc}"


__all__ = ["export_nsk"]
