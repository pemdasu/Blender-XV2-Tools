import secrets
import struct
from collections import defaultdict

import bpy
import mathutils

from ..ESK.ESK import ESK_Bone, ESK_File
from .EAN import ComponentType


def _align16(buf: bytearray) -> None:
    pad = (-len(buf)) % 16
    if pad:
        buf.extend(b"\x00" * pad)


def _bone_child_sibling_indices(bones: list[ESK_Bone]) -> None:
    for bone in bones:
        bone.child_index = -1
        bone.sibling_index = -1

    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for bone in bones:
        children_by_parent[bone.parent_index].append(bone.index)

    for parent_idx, child_indices in children_by_parent.items():
        if 0 <= parent_idx < len(bones) and child_indices:
            bones[parent_idx].child_index = child_indices[0]
        for i, child_idx in enumerate(child_indices):
            bones[child_idx].sibling_index = (
                child_indices[i + 1] if i + 1 < len(child_indices) else -1
            )


def _build_skeleton_bytes(bones: list[ESK_Bone]) -> bytes:
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

    pad = (-len(strings)) % 4
    skin_rel = string_off + len(strings) + pad
    skeleton_len = skin_rel + 48 * bone_count

    data = bytearray()
    data.extend(struct.pack("<h", bone_count))
    data.extend(struct.pack("<H", 0))
    data.extend(struct.pack("<I", index_rel))
    data.extend(struct.pack("<I", name_rel))
    data.extend(struct.pack("<I", skin_rel))
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<I", skeleton_len))
    data.extend(struct.pack("<Q", secrets.randbits(64) or 1))

    for bone in bones:
        data.extend(
            struct.pack("<hhhH", bone.parent_index, bone.child_index, bone.sibling_index, 0)
        )

    for offset in name_offsets:
        data.extend(struct.pack("<I", string_off + offset))
    data.extend(strings)
    if pad:
        data.extend(b"\x00" * pad)

    for bone in bones:
        loc, rot, scale = bone.matrix.decompose()
        data.extend(struct.pack("<4f", loc.x, loc.y, loc.z, 1.0))
        data.extend(struct.pack("<4f", rot.x, rot.y, rot.z, rot.w))
        data.extend(struct.pack("<4f", scale.x, scale.y, scale.z, 1.0))

    return bytes(data)


def _build_skeleton_from_armature(
    arm_obj: bpy.types.Object,
) -> tuple[ESK_File, bytes, dict[str, mathutils.Matrix]]:
    esk = ESK_File()
    bones: list[ESK_Bone] = []

    rest_locals: dict[str, mathutils.Matrix] = {}
    arm_bones = list(arm_obj.data.bones)

    root_bone = ESK_Bone(arm_obj.name, 0, mathutils.Matrix.Identity(4), -1, -1, -1)
    bones.append(root_bone)

    bone_indices = {bone.name: idx + 1 for idx, bone in enumerate(arm_bones)}

    for idx, bone in enumerate(arm_bones):
        parent_idx = bone_indices.get(bone.parent.name, 0) if bone.parent else 0
        local_mat = (
            bone.parent.matrix_local.inverted() @ bone.matrix_local
            if bone.parent
            else bone.matrix_local.copy()
        )
        rest_locals[bone.name] = local_mat.copy()
        esk_bone = ESK_Bone(bone.name, idx + 1, local_mat, parent_idx, -1, -1)
        bones.append(esk_bone)

    _bone_child_sibling_indices(bones)
    esk.bones = bones

    skeleton_bytes = _build_skeleton_bytes(bones)
    return esk, skeleton_bytes, rest_locals


def _collect_actions(arm_obj: bpy.types.Object, bone_names: set[str]) -> list[bpy.types.Action]:
    actions: list[bpy.types.Action] = []
    seen = set()
    for act in bpy.data.actions:
        if act.name in seen:
            continue
        for fc in act.fcurves:
            if not fc.data_path.startswith('pose.bones["'):
                continue
            name = fc.data_path.split('"')[1]
            if name in bone_names:
                actions.append(act)
                seen.add(act.name)
                break
    return actions


def _parse_anim_index(action_name: str, fallback: int) -> int:
    parts = action_name.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return fallback


def _parse_anim_meta(action_name: str, fallback_idx: int) -> tuple[int, str]:
    parts = action_name.split("|")
    anim_idx = fallback_idx
    anim_name = action_name
    if len(parts) >= 2:
        try:
            anim_idx = int(parts[1])
        except ValueError:
            anim_idx = fallback_idx
    if len(parts) >= 3:
        anim_name = parts[2]
    return anim_idx, anim_name


def _pack_half(val: float) -> bytes:
    return struct.pack("<e", val)


def _build_animation_bytes(
    action: bpy.types.Action,
    arm_obj: bpy.types.Object,
    esk: ESK_File,
    rest_locals: dict[str, mathutils.Matrix],
) -> tuple[bytes, int]:
    scene = bpy.context.scene
    original_frame = scene.frame_current

    frames_by_bone: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: {"pos": set(), "rot": set(), "scl": set()}
    )
    global_frames: set[int] = set()

    for fc in action.fcurves:
        if not fc.data_path.startswith('pose.bones["'):
            continue
        bone_name = fc.data_path.split('"')[1]
        target = None
        if fc.data_path.endswith("location"):
            target = "pos"
        elif fc.data_path.endswith("rotation_quaternion"):
            target = "rot"
        elif fc.data_path.endswith("scale"):
            target = "scl"
        if target is None:
            continue
        frames = {int(round(pt.co.x)) for pt in fc.keyframe_points}
        frames_by_bone[bone_name][target].update(frames)
        global_frames.update(frames)

    if not global_frames:
        return b"", 0

    frame_count = max(global_frames) + 1
    index_size = 1 if frame_count > 255 else 0
    float_size = 2  # 32-bit floats

    # Map frames to bones to minimize frame_set calls.
    frames_to_bones: dict[int, list[str]] = defaultdict(list)
    sample_frames_by_bone: dict[str, set[int]] = {}
    for bone_name, comp_frames in frames_by_bone.items():
        sample_frames = comp_frames["pos"] | comp_frames["rot"] | comp_frames["scl"]
        if not sample_frames:
            continue
        sample_frames_by_bone[bone_name] = sample_frames
        for f in sample_frames:
            frames_to_bones[f].append(bone_name)

    samples: dict[
        str, dict[int, tuple[mathutils.Vector, mathutils.Quaternion, mathutils.Vector]]
    ] = defaultdict(dict)
    for frame in sorted(frames_to_bones.keys()):
        scene.frame_set(frame)
        for bone_name in frames_to_bones[frame]:
            pbone = arm_obj.pose.bones.get(bone_name)
            if pbone is None:
                continue
            delta = mathutils.Matrix.LocRotScale(
                pbone.location.copy(),
                pbone.rotation_quaternion.copy(),
                pbone.scale.copy(),
            )
            rest_local = rest_locals.get(bone_name, mathutils.Matrix.Identity(4))
            baked_local = rest_local @ delta
            loc, rot, scl = baked_local.decompose()
            samples[bone_name][frame] = (loc, rot, scl)

    nodes: list[
        tuple[int, list[tuple[int, int, int, list[tuple[int, float, float, float, float]]]]]
    ] = []

    for bone in esk.bones:
        if bone.index == 0:
            continue  # skip the dummy root
        pbone = arm_obj.pose.bones.get(bone.name)
        if pbone is None:
            continue

        pos_frames = frames_by_bone[bone.name]["pos"]
        rot_frames = frames_by_bone[bone.name]["rot"]
        scale_frames = frames_by_bone[bone.name]["scl"]
        sample_frames = sample_frames_by_bone.get(bone.name)
        if not sample_frames:
            continue

        comps: list[tuple[int, int, int, list[tuple[int, float, float, float, float]]]] = []

        def build_component(comp_type: ComponentType, frames: set[int]):
            if not frames:
                return None
            keyframes = []
            for f in sorted(frames):
                loc, rot, scl = samples[bone.name][f]
                if comp_type == ComponentType.Position:
                    vals = (loc.x, loc.y, loc.z, 1.0)
                elif comp_type == ComponentType.Rotation:
                    vals = (rot.x, rot.y, rot.z, rot.w)
                else:
                    vals = (scl.x, scl.y, scl.z, 1.0)
                keyframes.append((f, *vals))

            end_frame = frame_count - 1
            if keyframes[0][0] != 0:
                first_vals = keyframes[0][1:]
                keyframes.insert(0, (0, *first_vals))
            if keyframes[-1][0] != end_frame:
                last_vals = keyframes[-1][1:]
                keyframes.append((end_frame, *last_vals))

            return comp_type, 7, 0, keyframes

        for comp_type, frames in (
            (ComponentType.Position, pos_frames),
            (ComponentType.Rotation, rot_frames),
            (ComponentType.Scale, scale_frames),
        ):
            comp = build_component(comp_type, frames)
            if comp:
                comps.append(comp)

        if comps:
            nodes.append((bone.index, comps))

    scene.frame_set(original_frame)

    if not nodes:
        return b"", 0

    anim = bytearray()
    anim.extend(b"\x00\x00")
    anim.append(index_size)
    anim.append(float_size)
    anim.extend(struct.pack("<I", frame_count))
    anim.extend(struct.pack("<I", len(nodes)))
    anim.extend(struct.pack("<I", 16 if nodes else 0))

    node_table_offset = len(anim)
    for _ in nodes:
        anim.extend(b"\x00\x00\x00\x00")

    for node_idx, (bone_idx, comps) in enumerate(nodes):
        node_start = len(anim)
        anim[node_table_offset + 4 * node_idx : node_table_offset + 4 * node_idx + 4] = struct.pack(
            "<I", node_start
        )
        anim.extend(struct.pack("<h", bone_idx))
        anim.extend(struct.pack("<h", len(comps)))
        anim.extend(struct.pack("<I", 8 if comps else 0))

        comp_table_offset = len(anim)
        for _ in comps:
            anim.extend(b"\x00\x00\x00\x00")

        for comp_idx, (ctype, i01, i02, keyframes) in enumerate(comps):
            comp_start = len(anim)
            anim[comp_table_offset + 4 * comp_idx : comp_table_offset + 4 * comp_idx + 4] = (
                struct.pack("<I", comp_start - node_start)
            )

            anim.extend(struct.pack("<BBhI", ctype, i01, i02, len(keyframes)))
            anim.extend(struct.pack("<II", 0, 0))

            idx_offset = len(anim)
            for frame_idx, *_vals in keyframes:
                if index_size == 0:
                    anim.extend(struct.pack("<B", frame_idx))
                else:
                    anim.extend(struct.pack("<H", frame_idx))

            _align16(anim)

            float_offset = len(anim)
            for _, x, y, z, w in keyframes:
                if float_size == 1:
                    anim.extend(_pack_half(x))
                    anim.extend(_pack_half(y))
                    anim.extend(_pack_half(z))
                    anim.extend(_pack_half(w))
                else:
                    anim.extend(struct.pack("<4f", x, y, z, w))

            anim[comp_start + 8 : comp_start + 12] = struct.pack("<I", idx_offset - comp_start)
            anim[comp_start + 12 : comp_start + 16] = struct.pack("<I", float_offset - comp_start)

    anim.extend(b"\x00" * 12)
    return bytes(anim), frame_count


def export_ean(filepath: str, arm_obj: bpy.types.Object) -> bool:
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return False

    esk, skeleton_bytes, rest_locals = _build_skeleton_from_armature(arm_obj)
    actual_bone_names = {b.name for b in esk.bones if b.index != 0}

    actions = _collect_actions(arm_obj, actual_bone_names)
    if not actions:
        return False

    animations_by_index: dict[int, tuple[bytes, str]] = {}
    max_index = -1
    for fallback_idx, act in enumerate(sorted(actions, key=lambda a: a.name)):
        anim_index, anim_label = _parse_anim_meta(act.name, fallback_idx)
        anim_bytes, frame_count = _build_animation_bytes(act, arm_obj, esk, rest_locals)
        if not anim_bytes:
            continue
        animations_by_index[anim_index] = (anim_bytes, anim_label)
        max_index = max(max_index, anim_index)

    if max_index < 0:
        return False

    animation_count = max_index + 1

    out = bytearray([35, 69, 65, 78, 254, 255, 32, 0])
    out.extend(struct.pack("<I", 37568))
    out.extend(b"\x00\x00\x00\x00")
    out.append(0)  # is_camera
    out.append(4)
    out.extend(struct.pack("<H", animation_count))
    out.extend(b"\x00" * 12)

    skeleton_offset = len(out)
    out[20:24] = struct.pack("<I", skeleton_offset)
    out.extend(skeleton_bytes)

    if animation_count > 0:
        out[24:28] = struct.pack("<I", len(out))
        animation_table_offset = len(out)
        for _ in range(animation_count):
            out.extend(b"\x00\x00\x00\x00")

        for idx in range(animation_count):
            entry = animations_by_index.get(idx)
            if not entry:
                continue
            anim_bytes, _ = entry
            _align16(out)
            out[animation_table_offset + 4 * idx : animation_table_offset + 4 * idx + 4] = (
                struct.pack("<I", len(out))
            )
            out.extend(anim_bytes)

        out[28:32] = struct.pack("<I", len(out))
        name_table_offset = len(out)
        for _ in range(animation_count):
            out.extend(b"\x00\x00\x00\x00")

        for idx in range(animation_count):
            entry = animations_by_index.get(idx)
            if not entry:
                continue
            _, anim_label = entry
            out[name_table_offset + 4 * idx : name_table_offset + 4 * idx + 4] = struct.pack(
                "<I", len(out)
            )
            out.extend(anim_label.encode("ascii", "ignore"))
            out.append(0)

    with open(filepath, "wb") as f:
        f.write(out)
    return True


__all__ = ["export_ean"]
