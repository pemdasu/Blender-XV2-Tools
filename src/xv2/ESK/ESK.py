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
    bone_index_table_offset = struct.unpack_from("<I", data, offs + 4)[0] + offs
    name_table_offset = struct.unpack_from("<I", data, offs + 8)[0] + offs
    skinning_table_offset = struct.unpack_from("<I", data, offs + 12)[0] + offs

    for bone_index in range(bone_count):
        bone_index_offset = bone_index_table_offset + 8 * bone_index
        parent_idx = struct.unpack_from("<h", data, bone_index_offset + 0)[0]
        child_idx = struct.unpack_from("<h", data, bone_index_offset + 2)[0]
        sibling_idx = struct.unpack_from("<h", data, bone_index_offset + 4)[0]

        name_rel = struct.unpack_from("<I", data, name_table_offset + 4 * bone_index)[0]
        name_off = offs + name_rel
        name = read_cstring(data, name_off)

        t_off = skinning_table_offset + 48 * bone_index
        px, py, pz, pw, rx, ry, rz, rw, sx, sy, sz, sw = struct.unpack_from("<12f", data, t_off)

        pos = mathutils.Vector((px, py, pz)) * pw
        rot = mathutils.Quaternion((rw, rx, ry, rz))
        scl = mathutils.Vector((sx, sy, sz)) * sw

        local_mat = mathutils.Matrix.LocRotScale(pos, rot, scl)

        esk.bones.append(ESK_Bone(name, bone_index, local_mat, parent_idx, child_idx, sibling_idx))

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

    for bone in bones:
        world_matrix = compute_world(bone)
        edit_bone = ebones_by_index[bone.index]

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
