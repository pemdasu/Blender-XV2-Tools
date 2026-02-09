import contextlib
import secrets
import struct
from collections import defaultdict
from pathlib import Path

import bpy
import mathutils

from ...utils import read_cstring
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


def _collect_actions(bone_names: set[str]) -> list[bpy.types.Action]:
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


def _load_source_ean_template(path_value: object) -> dict[str, object] | None:
    try:
        source_path = str(path_value or "").strip()
        if not source_path:
            return None
        source_file = Path(source_path)
        if not source_file.is_file():
            return None

        data = source_file.read_bytes()
        if len(data) < 32 or data[0:4] != b"#EAN":
            return None

        version = struct.unpack_from("<I", data, 8)[0]
        i17 = int(data[17])
        animation_count = struct.unpack_from("<H", data, 18)[0]
        skeleton_offset = struct.unpack_from("<I", data, 20)[0]
        animation_table_offset = struct.unpack_from("<I", data, 24)[0]
        if skeleton_offset <= 0 or animation_table_offset <= skeleton_offset:
            return None
        if animation_table_offset > len(data):
            return None

        bone_count = struct.unpack_from("<h", data, skeleton_offset + 0)[0]
        name_table_offset = struct.unpack_from("<I", data, skeleton_offset + 8)[0] + skeleton_offset
        if bone_count < 0 or name_table_offset <= 0:
            return None
        if name_table_offset + bone_count * 4 > len(data):
            return None

        source_esk = ESK_File()
        source_rest_locals: dict[str, mathutils.Matrix] = {}
        index_table_offset = struct.unpack_from("<I", data, skeleton_offset + 4)[0] + skeleton_offset
        skinning_table_offset = struct.unpack_from("<I", data, skeleton_offset + 12)[0] + skeleton_offset
        for bone_index in range(bone_count):
            bone_index_offset = index_table_offset + 8 * bone_index
            parent_idx = struct.unpack_from("<h", data, bone_index_offset + 0)[0]
            child_idx = struct.unpack_from("<h", data, bone_index_offset + 2)[0]
            sibling_idx = struct.unpack_from("<h", data, bone_index_offset + 4)[0]
            name_rel = struct.unpack_from("<I", data, name_table_offset + 4 * bone_index)[0]
            name_off = skeleton_offset + name_rel
            if not (0 <= name_off < len(data)):
                return None
            bone_name = read_cstring(data, name_off)

            t_off = skinning_table_offset + 48 * bone_index
            px, py, pz, pw, rx, ry, rz, rw, sx, sy, sz, sw = struct.unpack_from("<12f", data, t_off)
            pos = mathutils.Vector((px, py, pz)) * pw
            rot = mathutils.Quaternion((rw, rx, ry, rz))
            scl = mathutils.Vector((sx, sy, sz)) * sw
            local_mat = mathutils.Matrix.LocRotScale(pos, rot, scl)
            source_rest_locals[bone_name] = local_mat.copy()
            source_esk.bones.append(
                ESK_Bone(
                    bone_name,
                    bone_index,
                    local_mat,
                    parent_idx,
                    child_idx,
                    sibling_idx,
                )
            )

        float_size = 1
        if animation_count > 0:
            for anim_idx in range(animation_count):
                anim_ptr = struct.unpack_from("<I", data, animation_table_offset + 4 * anim_idx)[0]
                if anim_ptr <= 0 or anim_ptr + 4 > len(data):
                    continue
                source_float_size = int(data[anim_ptr + 3])
                if source_float_size in (1, 2):
                    float_size = source_float_size
                break

        return {
            "version": version,
            "i17": i17,
            "skeleton_bytes": data[skeleton_offset:animation_table_offset],
            "float_size": float_size,
            "esk": source_esk,
            "rest_locals": source_rest_locals,
        }
    except Exception:
        return None


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


def _is_character_action_name(action_name: str) -> bool:
    parts = action_name.split("|")
    if len(parts) < 3:
        return False
    try:
        int(parts[1])
    except ValueError:
        return False
    return True


def _parse_action_index(action_name: str) -> int | None:
    parts = action_name.split("|")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _read_ean_index(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _action_anim_index(action: bpy.types.Action, fallback_idx: int) -> int:
    idx = _read_ean_index(action.get("ean_index"))
    if idx is not None:
        return idx
    anim_idx, _ = _parse_anim_meta(action.name, fallback_idx)
    return anim_idx


def _action_sort_key(action: bpy.types.Action) -> tuple:
    idx = _read_ean_index(action.get("ean_index"))
    if idx is not None:
        return (0, idx, action.name)
    name_idx = _parse_action_index(action.name)
    if name_idx is not None:
        return (1, name_idx, action.name)
    return (2, action.name)


def _pack_half(val: float) -> bytes:
    return struct.pack("<e", val)


def _patch_skeleton_rest_transforms(
    skeleton_bytes: bytes,
    rest_locals: dict[str, mathutils.Matrix],
) -> bytes:
    try:
        data = bytearray(skeleton_bytes)
        if len(data) < 36:
            return skeleton_bytes

        bone_count = struct.unpack_from("<h", data, 0)[0]
        name_rel = struct.unpack_from("<I", data, 8)[0]
        skin_rel = struct.unpack_from("<I", data, 12)[0]
        if bone_count <= 0:
            return skeleton_bytes
        if name_rel <= 0 or skin_rel <= 0:
            return skeleton_bytes
        if name_rel + bone_count * 4 > len(data):
            return skeleton_bytes
        if skin_rel + bone_count * 48 > len(data):
            return skeleton_bytes

        for bone_index in range(bone_count):
            name_off_rel = struct.unpack_from("<I", data, name_rel + 4 * bone_index)[0]
            if not (0 <= name_off_rel < len(data)):
                continue
            bone_name = read_cstring(data, name_off_rel)
            local_mat = rest_locals.get(bone_name)
            if local_mat is None:
                continue

            loc, rot, scale = local_mat.decompose()
            t_off = skin_rel + 48 * bone_index
            struct.pack_into("<4f", data, t_off + 0, loc.x, loc.y, loc.z, 1.0)
            struct.pack_into("<4f", data, t_off + 16, rot.x, rot.y, rot.z, rot.w)
            struct.pack_into("<4f", data, t_off + 32, scale.x, scale.y, scale.z, 1.0)

        return bytes(data)
    except Exception:
        return skeleton_bytes


def _build_animation_bytes(
    action: bpy.types.Action,
    arm_obj: bpy.types.Object,
    esk: ESK_File,
    rest_locals: dict[str, mathutils.Matrix],
    depsgraph: bpy.types.Depsgraph,
    add_dummy_rest: bool = False,
    float_size: int = 1,
) -> bytes:
    scene = bpy.context.scene
    original_frame = scene.frame_current
    original_action = arm_obj.animation_data.action if arm_obj.animation_data else None

    frames_by_bone: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: {"pos": set(), "rot": set(), "scl": set()}
    )
    global_frames: set[int] = set()

    for fc in action.fcurves:
        if not fc.data_path.startswith('pose.bones["'):
            continue
        bone_name = fc.data_path.split('"')[1]
        target = None
        path_tail = fc.data_path.rsplit(".", 1)[-1]
        match path_tail:
            case "location":
                target = "pos"
            case "rotation_quaternion":
                target = "rot"
            case "scale":
                target = "scl"
        if target is None:
            continue
        frames = {int(round(pt.co.x)) for pt in fc.keyframe_points}
        frames_by_bone[bone_name][target].update(frames)
        global_frames.update(frames)

    dummy_bones: set[str] = set()
    if add_dummy_rest:
        for bone in esk.bones:
            if bone.index == 0:
                continue
            bone_frames = frames_by_bone[bone.name]
            if bone_frames["pos"] or bone_frames["rot"] or bone_frames["scl"]:
                continue
            dummy_bones.add(bone.name)
            bone_frames["pos"].add(0)
            bone_frames["rot"].add(0)
            bone_frames["scl"].add(0)
            global_frames.add(0)

    if not global_frames:
        scene.frame_set(original_frame)
        if arm_obj.animation_data:
            arm_obj.animation_data.action = original_action
        return b""

    frame_count = max(global_frames) + 1
    index_size = 1 if frame_count > 255 else 0
    float_size = 2 if int(float_size) == 2 else 1

    # Map frames to bones to minimize frame_set calls.
    frames_to_bones: dict[int, list[str]] = defaultdict(list)
    sample_frames_by_bone: dict[str, set[int]] = {}
    for bone_name, comp_frames in frames_by_bone.items():
        sample_frames = comp_frames["pos"] | comp_frames["rot"] | comp_frames["scl"]
        if not sample_frames:
            continue
        sample_frames_by_bone[bone_name] = sample_frames
        if bone_name in dummy_bones:
            continue
        for f in sample_frames:
            frames_to_bones[f].append(bone_name)

    samples: dict[
        str, dict[int, tuple[mathutils.Vector, mathutils.Quaternion, mathutils.Vector]]
    ] = defaultdict(dict)
    for frame in sorted(frames_to_bones.keys()):
        scene.frame_set(frame)
        with contextlib.suppress(Exception):
            bpy.context.view_layer.update()
        arm_eval = arm_obj.evaluated_get(depsgraph)
        for bone_name in frames_to_bones[frame]:
            pbone = arm_eval.pose.bones.get(bone_name)
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

    if dummy_bones:
        for bone_name in dummy_bones:
            rest_local = rest_locals.get(bone_name, mathutils.Matrix.Identity(4))
            loc, rot, scl = rest_local.decompose()
            samples[bone_name][0] = (loc, rot, scl)

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

        def build_component(comp_type: ComponentType, frames: set[int], pad_last: bool = True):
            if not frames:
                return None
            keyframes = []
            for f in sorted(frames):
                loc, rot, scl = samples[bone.name][f]
                match comp_type:
                    case ComponentType.Position:
                        vals = (loc.x, loc.y, loc.z, 1.0)
                    case ComponentType.Rotation:
                        vals = (rot.x, rot.y, rot.z, rot.w)
                    case _:
                        vals = (scl.x, scl.y, scl.z, 1.0)
                keyframes.append((f, *vals))

            end_frame = frame_count - 1
            if keyframes[0][0] != 0:
                first_vals = keyframes[0][1:]
                keyframes.insert(0, (0, *first_vals))
            if pad_last and keyframes[-1][0] != end_frame:
                last_vals = keyframes[-1][1:]
                keyframes.append((end_frame, *last_vals))

            return comp_type, 7, 0, keyframes

        dummy_only = bone.name in dummy_bones
        for comp_type, frames in (
            (ComponentType.Position, pos_frames),
            (ComponentType.Rotation, rot_frames),
            (ComponentType.Scale, scale_frames),
        ):
            comp = build_component(comp_type, frames, pad_last=not dummy_only)
            if comp:
                comps.append(comp)

        if comps:
            nodes.append((bone.index, comps))

    scene.frame_set(original_frame)

    if not nodes:
        return b""

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

    # Restore original action and frame
    scene.frame_set(original_frame)
    if arm_obj.animation_data:
        arm_obj.animation_data.action = original_action
    return bytes(anim)


def export_ean(
    filepath: str, arm_obj: bpy.types.Object, add_dummy_rest: bool = False
) -> tuple[bool, str | None]:
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return False, "Select an armature to export."

    original_action = None
    restore_action = False

    try:
        esk, skeleton_bytes, rest_locals = _build_skeleton_from_armature(arm_obj)
        try:
            header_version = int(arm_obj.get("ean_i08", 37505))
        except Exception:
            header_version = 37505
        try:
            header_i17 = int(arm_obj.get("ean_i17", 4))
        except Exception:
            header_i17 = 4
        preferred_float_size = 1
        source_template = _load_source_ean_template(arm_obj.get("ean_source"))
        if source_template:
            template_skeleton = source_template.get("skeleton_bytes")
            source_esk = source_template.get("esk")
            source_rest_locals = source_template.get("rest_locals")
            if isinstance(template_skeleton, (bytes, bytearray)):
                skeleton_bytes = bytes(template_skeleton)
            if isinstance(source_esk, ESK_File) and source_esk.bones:
                esk = source_esk
            if isinstance(source_rest_locals, dict):
                merged_rest_locals = dict(source_rest_locals)
                merged_rest_locals.update(rest_locals)
                rest_locals = merged_rest_locals
            skeleton_bytes = _patch_skeleton_rest_transforms(skeleton_bytes, rest_locals)
            header_version = int(source_template.get("version", header_version))
            header_i17 = int(source_template.get("i17", header_i17))
            preferred_float_size = int(source_template.get("float_size", preferred_float_size))
        actual_bone_names = {b.name for b in esk.bones if b.index != 0}

        collected_actions = _collect_actions(actual_bone_names)
        actions = [act for act in collected_actions if _is_character_action_name(act.name)]
        if not collected_actions:
            return False, f"No actions with pose bone keyframes found on armature {arm_obj.name}."
        if not actions:
            arm_name = arm_obj.name or "Armature"
            example_name = f"{arm_name}|0|Idle"
            return False, (
                f"Actions must be named '<prefix>|<index>|<label>' (e.g. {example_name})."
            )
        armature_prefixed_actions = [a for a in actions if a.name.startswith(f"{arm_obj.name}|")]
        if armature_prefixed_actions:
            actions = armature_prefixed_actions

        if arm_obj.animation_data is None:
            arm_obj.animation_data_create()
        original_action = arm_obj.animation_data.action
        restore_action = True

        animations_by_index: dict[int, tuple[bytes, str]] = {}
        max_index = -1
        for fallback_idx, act in enumerate(sorted(actions, key=_action_sort_key)):
            anim_index = _action_anim_index(act, fallback_idx)
            _, anim_label = _parse_anim_meta(act.name, fallback_idx)

            arm_obj.animation_data.action = act
            with contextlib.suppress(Exception):
                bpy.context.view_layer.update()

            anim_bytes = _build_animation_bytes(
                act,
                arm_obj,
                esk,
                rest_locals,
                bpy.context.view_layer.depsgraph,
                add_dummy_rest=add_dummy_rest,
                float_size=preferred_float_size,
            )
            if not anim_bytes:
                continue
            animations_by_index[anim_index] = (anim_bytes, anim_label)
            max_index = max(max_index, anim_index)

        if max_index < 0:
            return (
                False,
                "Animations were found but none had exportable keyframes on pose bones (location/rotation/scale).",
            )

        animation_count = max_index + 1

        out = bytearray([35, 69, 65, 78, 254, 255, 32, 0])
        out.extend(struct.pack("<I", header_version))
        out.extend(b"\x00\x00\x00\x00")
        out.append(0)  # is_camera
        out.append(header_i17 & 0xFF)
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
        return True, None
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return False, f"Unexpected error while exporting: {exc}"
    finally:
        if restore_action and getattr(arm_obj, "animation_data", None):
            arm_obj.animation_data.action = original_action


__all__ = ["export_ean"]
