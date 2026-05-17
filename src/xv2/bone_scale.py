from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import bpy
import mathutils

from .consts import (
    BONE_SCALE_EPSILON,
    BONE_SCALE_IDENTITY,
    BONE_SCALE_PROP,
    SCD_LINK_TARGET_ARMATURE_PROP,
    SCD_LINK_TARGET_BONE_PROP,
)


def _as_scale_vector(value) -> mathutils.Vector:
    try:
        scale = mathutils.Vector((float(value[0]), float(value[1]), float(value[2])))
    except (TypeError, ValueError, IndexError):
        return BONE_SCALE_IDENTITY.copy()
    return scale


def is_identity_scale(scale: mathutils.Vector) -> bool:
    return all(abs(float(scale[index]) - 1.0) <= BONE_SCALE_EPSILON for index in range(3))


def get_bone_scale(bone: bpy.types.Bone | None) -> mathutils.Vector:
    if bone is None:
        return BONE_SCALE_IDENTITY.copy()
    return _as_scale_vector(getattr(bone, BONE_SCALE_PROP, BONE_SCALE_IDENTITY))


def get_armature_bone_scales(arm_obj: bpy.types.Object | None) -> dict[str, mathutils.Vector]:
    if arm_obj is None or arm_obj.type != "ARMATURE" or arm_obj.data is None:
        return {}

    scales: dict[str, mathutils.Vector] = {}
    for bone in arm_obj.data.bones:
        scale = get_bone_scale(bone)
        if not is_identity_scale(scale):
            scales[bone.name] = scale
    return scales


def get_scd_target_armature(arm_obj: bpy.types.Object | None) -> bpy.types.Object | None:
    if arm_obj is None:
        return None

    target_name = str(arm_obj.get(SCD_LINK_TARGET_ARMATURE_PROP, "")).strip()
    target_armature = bpy.data.objects.get(target_name) if target_name else None
    if target_armature is None or target_armature.type != "ARMATURE":
        return None
    return target_armature


def get_effective_armature_bone_scales(
    arm_obj: bpy.types.Object | None,
) -> dict[str, mathutils.Vector]:
    scales = get_armature_bone_scales(arm_obj)
    if arm_obj is None or arm_obj.type != "ARMATURE" or arm_obj.data is None:
        return scales

    target_armature = get_scd_target_armature(arm_obj)
    if target_armature is not None:
        for bone in arm_obj.data.bones:
            if bone.name in scales:
                continue
            target_bone_name = str(bone.get(SCD_LINK_TARGET_BONE_PROP, "")).strip()
            if not target_bone_name:
                continue
            target_bone = target_armature.data.bones.get(target_bone_name)
            target_scale = get_bone_scale(target_bone)
            if not is_identity_scale(target_scale):
                scales[bone.name] = target_scale

    return scales


def scale_vector(vector: mathutils.Vector, scale: mathutils.Vector) -> mathutils.Vector:
    return mathutils.Vector((vector.x * scale.x, vector.y * scale.y, vector.z * scale.z))


def get_bone_inherited_scale(bone: bpy.types.Bone | None) -> mathutils.Vector:
    scale = BONE_SCALE_IDENTITY.copy()
    while bone is not None:
        scale = scale_vector(scale, get_bone_scale(bone))
        bone = bone.parent
    return scale


def get_scd_link_scale(arm_obj: bpy.types.Object | None) -> mathutils.Vector:
    if arm_obj is None or arm_obj.type != "ARMATURE" or arm_obj.data is None:
        return BONE_SCALE_IDENTITY.copy()

    target_armature = get_scd_target_armature(arm_obj)
    if target_armature is None or target_armature.data is None:
        return BONE_SCALE_IDENTITY.copy()

    for bone in arm_obj.data.bones:
        target_bone_name = str(bone.get(SCD_LINK_TARGET_BONE_PROP, "")).strip()
        if not target_bone_name:
            continue
        target_bone = target_armature.data.bones.get(target_bone_name)
        scale = get_bone_inherited_scale(target_bone)
        if not is_identity_scale(scale):
            return scale

    return BONE_SCALE_IDENTITY.copy()


def get_inherited_parent_scale(
    bones: list,
    bone_index: int,
    bone_scales: dict[str, mathutils.Vector],
    cache: dict[int, mathutils.Vector] | None = None,
) -> mathutils.Vector:
    if not bone_scales or not (0 <= bone_index < len(bones)):
        return BONE_SCALE_IDENTITY.copy()

    cache = cache if cache is not None else {}
    cached_scale = cache.get(bone_index)
    if cached_scale is not None:
        return cached_scale.copy()

    bone = bones[bone_index]
    parent_index = int(getattr(bone, "parent_index", -1))
    if not (0 <= parent_index < len(bones)) or parent_index == bone_index:
        inherited_scale = BONE_SCALE_IDENTITY.copy()
    else:
        inherited_scale = get_inherited_parent_scale(
            bones,
            parent_index,
            bone_scales,
            cache,
        )
        parent_scale = bone_scales.get(getattr(bones[parent_index], "name", ""))
        if parent_scale is not None:
            inherited_scale = scale_vector(inherited_scale, parent_scale)

    cache[bone_index] = inherited_scale.copy()
    return inherited_scale


def bake_parent_scale_into_matrix(
    matrix: mathutils.Matrix,
    parent_scale: mathutils.Vector,
) -> mathutils.Matrix:
    if is_identity_scale(parent_scale):
        return matrix

    loc, rot, scale = matrix.decompose()
    loc = scale_vector(loc, parent_scale)
    return mathutils.Matrix.LocRotScale(loc, rot, scale)


def apply_bone_scale_preview(arm_obj: bpy.types.Object | None) -> int:
    if (
        arm_obj is None
        or arm_obj.type != "ARMATURE"
        or arm_obj.data is None
        or arm_obj.pose is None
    ):
        return 0

    changed = 0
    for bone in arm_obj.data.bones:
        pose_bone = arm_obj.pose.bones.get(bone.name)
        if pose_bone is None:
            continue
        pose_bone.scale = get_bone_scale(bone)
        changed += 1

    if hasattr(bpy.context.view_layer, "update"):
        bpy.context.view_layer.update()
    return changed


def _set_pose_scales(
    targets: list[tuple[bpy.types.Object, str, mathutils.Vector]],
) -> dict[tuple[int, str], tuple[bpy.types.Object, mathutils.Vector]]:
    previous_scales: dict[tuple[int, str], tuple[bpy.types.Object, mathutils.Vector]] = {}

    for arm_obj, bone_name, bone_scale in targets:
        if arm_obj is None or arm_obj.pose is None:
            continue
        pose_bone = arm_obj.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        key = (arm_obj.as_pointer(), bone_name)
        if key not in previous_scales:
            previous_scales[key] = (arm_obj, pose_bone.scale.copy())
        pose_bone.scale = bone_scale

    if previous_scales and hasattr(bpy.context.view_layer, "update"):
        bpy.context.view_layer.update()
    return previous_scales


def _restore_pose_scales(
    previous_scales: dict[tuple[int, str], tuple[bpy.types.Object, mathutils.Vector]],
) -> None:
    for (_arm_pointer, bone_name), (arm_obj, previous_scale) in previous_scales.items():
        if arm_obj is None or arm_obj.pose is None:
            continue
        pose_bone = arm_obj.pose.bones.get(bone_name)
        if pose_bone is not None:
            pose_bone.scale = previous_scale

    if previous_scales and hasattr(bpy.context.view_layer, "update"):
        bpy.context.view_layer.update()


@contextmanager
def applied_bone_scale_pose(
    arm_obj: bpy.types.Object | None,
    enabled: bool,
) -> Iterator[dict[str, mathutils.Vector]]:
    bone_scales = get_armature_bone_scales(arm_obj) if enabled else {}
    if not bone_scales or arm_obj is None or arm_obj.pose is None:
        yield bone_scales
        return

    targets = [(arm_obj, bone_name, bone_scale) for bone_name, bone_scale in bone_scales.items()]
    previous_scales = _set_pose_scales(targets)
    try:
        yield bone_scales
    finally:
        _restore_pose_scales(previous_scales)


@contextmanager
def applied_effective_bone_scale_pose(
    arm_obj: bpy.types.Object | None,
    enabled: bool,
) -> Iterator[dict[str, mathutils.Vector]]:
    bone_scales = get_effective_armature_bone_scales(arm_obj) if enabled else {}
    if not bone_scales or arm_obj is None or arm_obj.pose is None:
        yield bone_scales
        return

    direct_scales = get_armature_bone_scales(arm_obj)
    targets = [(arm_obj, bone_name, bone_scale) for bone_name, bone_scale in direct_scales.items()]

    target_armature = get_scd_target_armature(arm_obj)
    if target_armature is not None and arm_obj.data is not None:
        for bone in arm_obj.data.bones:
            target_bone_name = str(bone.get(SCD_LINK_TARGET_BONE_PROP, "")).strip()
            if not target_bone_name:
                continue
            target_bone = target_armature.data.bones.get(target_bone_name)
            target_scale = get_bone_scale(target_bone)
            if not is_identity_scale(target_scale):
                targets.append((target_armature, target_bone_name, target_scale))
    else:
        targets = [
            (arm_obj, bone_name, bone_scale) for bone_name, bone_scale in bone_scales.items()
        ]

    previous_scales = _set_pose_scales(targets)
    try:
        yield bone_scales
    finally:
        _restore_pose_scales(previous_scales)


@contextmanager
def identity_bone_scale_pose(
    arm_obj: bpy.types.Object | None,
    bone_scales: dict[str, mathutils.Vector],
) -> Iterator[None]:
    if not bone_scales or arm_obj is None or arm_obj.pose is None:
        yield
        return

    previous_scales = _set_pose_scales(
        [(arm_obj, bone_name, BONE_SCALE_IDENTITY.copy()) for bone_name in bone_scales]
    )
    try:
        yield
    finally:
        _restore_pose_scales(previous_scales)
