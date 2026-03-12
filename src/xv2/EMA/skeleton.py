from __future__ import annotations

import mathutils

from ...utils import read_cstring
from ...utils.binary import f32, u16, u32, u64
from ..ESK import ESK_Bone, ESK_File


def _to_parent_index(value: int) -> int:
    return -1 if value == 0xFFFF else int(value)


def _read_matrix(data: bytes, offset: int) -> mathutils.Matrix:
    return mathutils.Matrix(
        (
            (
                f32(data, offset + 0),
                f32(data, offset + 4),
                f32(data, offset + 8),
                f32(data, offset + 12),
            ),
            (
                f32(data, offset + 16),
                f32(data, offset + 20),
                f32(data, offset + 24),
                f32(data, offset + 28),
            ),
            (
                f32(data, offset + 32),
                f32(data, offset + 36),
                f32(data, offset + 40),
                f32(data, offset + 44),
            ),
            (
                f32(data, offset + 48),
                f32(data, offset + 52),
                f32(data, offset + 56),
                f32(data, offset + 60),
            ),
        )
    )


def parse_ema_skeleton_as_esk(data: bytes, skeleton_offset: int) -> tuple[ESK_File, list[int]]:
    if skeleton_offset <= 0:
        raise ValueError("EMA/EMO skeleton offset is invalid.")
    if skeleton_offset + 64 > len(data):
        raise ValueError("EMA/EMO skeleton header is out of range.")

    ik2_count = u16(data, skeleton_offset + 2)
    if ik2_count != 0:
        raise ValueError("EMA/EMO skeleton IK2 data is not supported.")

    bone_count = u16(data, skeleton_offset + 0)
    bone_offset_rel = u32(data, skeleton_offset + 8)
    names_offset_rel = u32(data, skeleton_offset + 12)
    abs_matrix_offset_rel = u32(data, skeleton_offset + 28)
    skeleton_flag = u16(data, skeleton_offset + 54)
    skeleton_id = u64(data, skeleton_offset + 56)

    bone_offset = skeleton_offset + bone_offset_rel
    names_offset = skeleton_offset + names_offset_rel
    abs_matrix_offset = skeleton_offset + abs_matrix_offset_rel if abs_matrix_offset_rel else 0

    esk = ESK_File()
    esk.skeleton_flag = int(skeleton_flag)
    esk.skeleton_id = int(skeleton_id)

    bone_part_indices: list[int] = []
    for bone_index in range(int(bone_count)):
        offset = bone_offset + (bone_index * 80)
        parent_idx = _to_parent_index(u16(data, offset + 0))
        child_idx = _to_parent_index(u16(data, offset + 2))
        sibling_idx = _to_parent_index(u16(data, offset + 4))
        part_idx = int(u16(data, offset + 6))

        # EMA/EMO stores this matrix transposed compared to Blender's column-vector style.
        relative_matrix = _read_matrix(data, offset + 16).transposed()
        bone = ESK_Bone(
            name=f"bone_{bone_index}",
            index=bone_index,
            matrix=relative_matrix,
            parent_index=parent_idx,
            child_index=child_idx,
            sibling_index=sibling_idx,
        )

        if abs_matrix_offset:
            bone.absolute_matrix = _read_matrix(
                data,
                abs_matrix_offset + (bone_index * 64),
            ).transposed()

        esk.bones.append(bone)
        bone_part_indices.append(part_idx)

    for bone_index in range(int(bone_count)):
        name_rel = u32(data, names_offset + (bone_index * 4))
        if name_rel:
            name = read_cstring(data, skeleton_offset + name_rel)
            if name:
                esk.bones[bone_index].name = name

    return esk, bone_part_indices


__all__ = ["parse_ema_skeleton_as_esk"]
