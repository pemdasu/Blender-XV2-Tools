import math
import struct

import bpy
import mathutils

from ...utils import read_cstring

ESK_SIGNATURE = 1263748387


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


def parse_esk(path: str) -> ESK_File:
    with open(path, "rb") as file_handle:
        data = file_handle.read()

    if struct.unpack_from("<I", data, 0)[0] != ESK_SIGNATURE:
        raise ValueError("Invalid ESK signature")

    esk = ESK_File()

    version = struct.unpack_from("<H", data, 8)[0]
    _ = version  # version currently unused
    skeleton_offset = struct.unpack_from("<I", data, 16)[0]
    offs = skeleton_offset

    bone_count = struct.unpack_from("<h", data, offs + 0)[0]
    _ = struct.unpack_from("<h", data, offs + 2)[0]
    bone_index_table_offset = struct.unpack_from("<I", data, offs + 4)[0] + offs
    name_table_offset = struct.unpack_from("<I", data, offs + 8)[0] + offs
    relative_transform_offset = struct.unpack_from("<I", data, offs + 12)[0] + offs
    absolute_matrix_offset = struct.unpack_from("<I", data, offs + 16)[0]
    if absolute_matrix_offset:
        absolute_matrix_offset += offs

    for bone_index in range(bone_count):
        bone_index_offset = bone_index_table_offset + 8 * bone_index
        parent_idx = struct.unpack_from("<h", data, bone_index_offset + 0)[0]
        child_idx = struct.unpack_from("<h", data, bone_index_offset + 2)[0]
        sibling_idx = struct.unpack_from("<h", data, bone_index_offset + 4)[0]

        name_rel = struct.unpack_from("<I", data, name_table_offset + 4 * bone_index)[0]
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


def build_armature(esk: ESK_File, armature_name: str = "ESK_Armature") -> bpy.types.Object:
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

    for bone in bones:
        edit_bone = ebones_by_index[bone.index]

        is_thumb = "thumb" in (bone.name or "").lower()
        if is_thumb:
            world_matrix = compute_world_abs(bone) or compute_world(bone)
            head = world_matrix.to_translation()
            rotation_matrix = world_matrix.to_quaternion().to_matrix()
            bone_length = 0.0
            if 0 < bone.child_index < len(esk.bones):
                child_bone = esk.bones[bone.child_index]
                child_world = compute_world_abs(child_bone) or compute_world(child_bone)
                bone_length = (child_world.to_translation() - head).length
            elif bone.matrix is not None:
                bone_length = bone.matrix.to_translation().length
            if bone_length <= 1e-6:
                bone_length = 0.1
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
    return arm_obj
