import math
import struct

import bpy
import mathutils

from ...utils import read_cstring
from ...utils.binary import i16, u16, u32, u64
from ..consts import (
    ESK_SIGNATURE,
    SOURCE_ROOT_MATRIX_PROP,
    SOURCE_ROOT_NAME_PROP,
)


class ESK_Bone:
    def __init__(
        self,
        name: str,
        index: int,
        matrix: mathutils.Matrix,
        parent_index: int = -1,
        child_index: int = -1,
        sibling_index: int = -1,
    ):
        self.name = name
        self.index = index
        self.matrix = matrix
        self.parent_index = parent_index
        self.child_index = child_index
        self.sibling_index = sibling_index
        self.absolute_matrix: mathutils.Matrix | None = None


class ESK_File:
    def __init__(self):
        self.bones: list[ESK_Bone] = []
        self.version: int = 37568
        self.i_10: int = 0
        self.i_12: int = 0
        self.i_24: int = 0
        self.skeleton_flag: int = 0
        self.skeleton_id: int = 0


def _flatten_matrix(matrix: mathutils.Matrix) -> list[float]:
    return [float(matrix[row][col]) for row in range(4) for col in range(4)]


def _matrix_from_prop(value) -> mathutils.Matrix | None:
    if value is None:
        return None
    try:
        items = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(items) != 16:
        return None
    return mathutils.Matrix(
        (
            (items[0], items[1], items[2], items[3]),
            (items[4], items[5], items[6], items[7]),
            (items[8], items[9], items[10], items[11]),
            (items[12], items[13], items[14], items[15]),
        )
    )


def get_source_root_matrix(arm_obj: bpy.types.Object) -> mathutils.Matrix | None:
    arm_data = getattr(arm_obj, "data", None)
    if arm_data is None:
        return None
    return _matrix_from_prop(arm_data.get(SOURCE_ROOT_MATRIX_PROP))


def get_source_root_name(arm_obj: bpy.types.Object) -> str | None:
    arm_data = getattr(arm_obj, "data", None)
    if arm_data is None:
        return None
    root_name = str(arm_data.get(SOURCE_ROOT_NAME_PROP, "")).strip()
    return root_name or None


def store_source_skeleton(arm_obj: bpy.types.Object, esk: ESK_File) -> None:
    arm_data = getattr(arm_obj, "data", None)
    if arm_data is None or not esk.bones:
        return

    arm_data[SOURCE_ROOT_NAME_PROP] = str(esk.bones[0].name or arm_obj.name)
    arm_data[SOURCE_ROOT_MATRIX_PROP] = _flatten_matrix(esk.bones[0].matrix)


def parse_esk_bytes(data: bytes) -> ESK_File:
    if u32(data, 0) != ESK_SIGNATURE:
        raise ValueError("Invalid ESK signature")

    esk = ESK_File()

    esk.version = u16(data, 8)
    esk.i_10 = u16(data, 10)
    esk.i_12 = u32(data, 12)
    skeleton_offset = u32(data, 16)
    esk.i_24 = u32(data, 24)
    offs = skeleton_offset

    bone_count = i16(data, offs + 0)
    esk.skeleton_flag = i16(data, offs + 2)
    bone_index_table_offset = u32(data, offs + 4) + offs
    name_table_offset = u32(data, offs + 8) + offs
    relative_transform_offset = u32(data, offs + 12) + offs
    absolute_matrix_offset = u32(data, offs + 16)
    if absolute_matrix_offset:
        absolute_matrix_offset += offs
    esk.skeleton_id = u64(data, offs + 28)

    for bone_index in range(bone_count):
        bone_index_offset = bone_index_table_offset + 8 * bone_index
        parent_idx = i16(data, bone_index_offset + 0)
        child_idx = i16(data, bone_index_offset + 2)
        sibling_idx = i16(data, bone_index_offset + 4)

        name_rel = u32(data, name_table_offset + 4 * bone_index)
        name_off = offs + name_rel
        name = read_cstring(data, name_off)

        t_off = relative_transform_offset + 48 * bone_index
        px, py, pz, pw, rx, ry, rz, rw, sx, sy, sz, sw = struct.unpack_from("<12f", data, t_off)

        pos = mathutils.Vector((px, py, pz)) * pw
        rot = mathutils.Quaternion((rw, rx, ry, rz))
        scl = mathutils.Vector((sx, sy, sz)) * sw

        local_mat = mathutils.Matrix.LocRotScale(pos, rot, scl)

        esk_bone = ESK_Bone(name, bone_index, local_mat, parent_idx, child_idx, sibling_idx)
        if absolute_matrix_offset:
            m_off = absolute_matrix_offset + 64 * bone_index
            m_vals = struct.unpack_from("<16f", data, m_off)
            esk_bone.absolute_matrix = mathutils.Matrix(
                (
                    (m_vals[0], m_vals[1], m_vals[2], m_vals[3]),
                    (m_vals[4], m_vals[5], m_vals[6], m_vals[7]),
                    (m_vals[8], m_vals[9], m_vals[10], m_vals[11]),
                    (m_vals[12], m_vals[13], m_vals[14], m_vals[15]),
                )
            )
        esk.bones.append(esk_bone)

    return esk


def parse_esk(path: str) -> ESK_File:
    with open(path, "rb") as file_handle:
        data = file_handle.read()
    return parse_esk_bytes(data)


def build_armature(
    esk: ESK_File,
    armature_name: str = "ESK_Armature",
    preserve_bone_axes: bool = False,
) -> bpy.types.Object:
    bpy.ops.object.add(type="ARMATURE", enter_editmode=True)
    arm_obj = bpy.context.object
    arm = arm_obj.data

    if arm.edit_bones:
        arm.edit_bones.remove(arm.edit_bones[0])

    arm.name = armature_name

    bones = esk.bones[1:]

    ebones_by_index: dict[int, bpy.types.EditBone] = {}

    for bone in bones:
        edit_bone = arm.edit_bones.new(bone.name or f"bone_{bone.index}")
        ebones_by_index[bone.index] = edit_bone

    world_mats: dict[int, mathutils.Matrix] = {}
    world_abs_mats: dict[int, mathutils.Matrix] = {}

    def compute_world(bone_data: ESK_Bone) -> mathutils.Matrix:
        if bone_data.index in world_mats:
            return world_mats[bone_data.index]
        matrix = bone_data.matrix.copy()
        loc, rot, scale = matrix.decompose()
        if abs(scale.x - 1.0) > 1e-5 or abs(scale.y - 1.0) > 1e-5 or abs(scale.z - 1.0) > 1e-5:
            matrix = mathutils.Matrix.LocRotScale(loc, rot, mathutils.Vector((1.0, 1.0, 1.0)))
        if (
            0 <= bone_data.parent_index < len(esk.bones)
            and esk.bones[bone_data.parent_index] is not bone_data
        ):
            parent_bone = esk.bones[bone_data.parent_index]
            matrix = compute_world(parent_bone) @ matrix
        world_mats[bone_data.index] = matrix
        return matrix

    def compute_world_abs(bone_data: ESK_Bone) -> mathutils.Matrix | None:
        abs_mat = bone_data.absolute_matrix
        if abs_mat is None:
            return None
        if bone_data.index in world_abs_mats:
            return world_abs_mats[bone_data.index]
        # ESK absolute matrices are stored with translation in the last row.
        matrix = abs_mat.transposed().inverted()
        world_abs_mats[bone_data.index] = matrix
        return matrix

    def get_bone_length(bone_data: ESK_Bone, head: mathutils.Vector) -> float:
        length = 0.0
        if 0 < bone_data.child_index < len(esk.bones):
            child_bone = esk.bones[bone_data.child_index]
            child_world = compute_world_abs(child_bone) or compute_world(child_bone)
            length = (child_world.to_translation() - head).length
        elif bone_data.matrix is not None:
            length = bone_data.matrix.to_translation().length
        if length <= 1e-6:
            return 0.1
        return length

    def orient_edit_bone(
        edit_bone: bpy.types.EditBone,
        world_matrix: mathutils.Matrix,
        bone_length: float,
    ) -> None:
        head = world_matrix.to_translation()
        rotation_matrix = world_matrix.to_quaternion().to_matrix()

        direction = rotation_matrix @ mathutils.Vector((0.0, 1.0, 0.0))
        if direction.length <= 1e-6:
            direction = mathutils.Vector((0.0, 1.0, 0.0))
        else:
            direction.normalize()

        edit_bone.head = head
        edit_bone.tail = head + (direction * bone_length)

        # Keep the source z-axis as the roll reference so mirrored chains share orientation.
        roll_ref = rotation_matrix @ mathutils.Vector((0.0, 0.0, 1.0))
        if roll_ref.length <= 1e-6:
            roll_ref = rotation_matrix @ mathutils.Vector((1.0, 0.0, 0.0))
        if roll_ref.length > 1e-6:
            roll_ref.normalize()
            if abs(direction.dot(roll_ref)) > 0.999:
                fallback_ref = rotation_matrix @ mathutils.Vector((1.0, 0.0, 0.0))
                if fallback_ref.length > 1e-6:
                    fallback_ref.normalize()
                    if abs(direction.dot(fallback_ref)) <= 0.999:
                        roll_ref = fallback_ref
            if abs(direction.dot(roll_ref)) <= 0.999:
                edit_bone.align_roll(roll_ref)

    for bone in bones:
        edit_bone = ebones_by_index[bone.index]

        is_thumb = "thumb" in (bone.name or "").lower()
        if preserve_bone_axes:
            if is_thumb:
                world_matrix = compute_world_abs(bone) or compute_world(bone)
                bone_length = get_bone_length(bone, world_matrix.to_translation())
                orient_edit_bone(edit_bone, world_matrix, bone_length)
                edit_bone.roll -= math.radians(90.0)
            else:
                world_matrix = compute_world(bone)
                bone_length = get_bone_length(bone, world_matrix.to_translation())
                orient_edit_bone(edit_bone, world_matrix, bone_length)
        else:
            if is_thumb:
                world_matrix = compute_world_abs(bone) or compute_world(bone)
                head = world_matrix.to_translation()
                rotation_matrix = world_matrix.to_quaternion().to_matrix()
                bone_length = get_bone_length(bone, head)
                direction = rotation_matrix @ mathutils.Vector((0.0, 1.0, 0.0))
                if direction.length <= 1e-6:
                    direction = mathutils.Vector((0.0, 1.0, 0.0))
                direction.normalize()

                tail = head + (direction * bone_length)
                edit_bone.head = head
                edit_bone.tail = tail

                ref_axis = rotation_matrix @ mathutils.Vector((1.0, 0.0, 0.0))
                if ref_axis.length <= 1e-6:
                    ref_axis = mathutils.Vector((0.0, 0.0, 1.0))
                edit_bone.align_roll(ref_axis)
                edit_bone.roll -= math.radians(90.0)
            else:
                world_matrix = compute_world(bone)
                head = world_matrix.to_translation()
                rotation_matrix = world_matrix.to_3x3()
                tail = head + (rotation_matrix @ mathutils.Vector((0.0, 0.1, 0.0)))
                edit_bone.head = head
                edit_bone.tail = tail

        if bone.parent_index > 0 and bone.parent_index in ebones_by_index:
            edit_bone.parent = ebones_by_index[bone.parent_index]
            edit_bone.use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")
    arm_obj.location = (0.0, 0.0, 0.0)
    store_source_skeleton(arm_obj, esk)
    return arm_obj
