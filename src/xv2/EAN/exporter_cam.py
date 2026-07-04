import math
import secrets
import struct
from collections.abc import Sequence
from pathlib import Path

import bpy

from ...utils.blender_compat import find_action_fcurve, iter_action_fcurves
from .EAN import ComponentType


def _map_vec_to_xv2(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (x, z, -y)


def _collect_frames_from_action(action: bpy.types.Action, data_paths: Sequence[str]) -> set[int]:
    frames: set[int] = set()
    if action is None:
        return frames
    for fcurve in iter_action_fcurves(action):
        if fcurve.data_path in data_paths:
            frames.update(int(round(point.co.x)) for point in fcurve.keyframe_points)
    return frames


def _eval_scalar(
    action: bpy.types.Action, data_path: str, frame: int, default: float = 0.0, index: int = 0
) -> float:
    if action is None:
        return default
    fcurve = find_action_fcurve(action, data_path, index=index)
    return fcurve.evaluate(frame) if fcurve else default


def _calc_edge_frames(
    keyframes: list[tuple[int, float, float, float, float]], frame_count: int
) -> list[tuple[int, float, float, float, float]]:
    if not keyframes:
        return keyframes
    frames = {keyframe[0] for keyframe in keyframes}
    first = min(frames)
    last = max(frames)
    keyframe_by_frame = {keyframe[0]: keyframe for keyframe in keyframes}

    if 0 not in frames:
        keyframe_by_frame[0] = (0, *keyframe_by_frame[first][1:])
    end_frame = max(frame_count - 1, last)
    if end_frame not in frames:
        keyframe_by_frame[end_frame] = (end_frame, *keyframe_by_frame[last][1:])

    return [keyframe_by_frame[f] for f in sorted(keyframe_by_frame.keys())]


def _pack_half(value: float) -> bytes:
    try:
        return struct.pack("<e", value)
    except struct.error:
        import numpy as np  # type: ignore

        return np.float16(value).tobytes()


def _write_skeleton_single_node() -> bytes:
    bone_name = "Node"
    bone_count = 1
    skeleton_id = secrets.randbits(64) or 1

    header_size = 36
    index_table_rel = header_size
    name_table_rel = index_table_rel + bone_count * 8
    string_off = name_table_rel + bone_count * 4
    name_bytes = bone_name.encode("ascii") + b"\x00"
    pad = (-len(name_bytes)) % 4
    skinning_rel = string_off + len(name_bytes) + pad
    skeleton_len = skinning_rel + 48 * bone_count

    data = bytearray()
    data.extend(struct.pack("<h", bone_count))
    data.extend(struct.pack("<H", 0))  # flag
    data.extend(struct.pack("<I", index_table_rel))
    data.extend(struct.pack("<I", name_table_rel))
    data.extend(struct.pack("<I", skinning_rel))
    data.extend(struct.pack("<I", 0))  # extra1
    data.extend(struct.pack("<I", 0))  # extra2
    data.extend(struct.pack("<I", skeleton_len))  # extra3 (size)
    data.extend(struct.pack("<Q", skeleton_id))  # skeleton ID / extra4

    # Index table (parent, child, sibling, padding)
    data.extend(struct.pack("<hhhH", -1, -1, -1, 0))

    # Name table
    data.extend(struct.pack("<I", string_off))
    data.extend(name_bytes)
    if pad:
        data.extend(b"\x00" * pad)

    # Skinning (pos, rot, scale) using same defaults as camera EANs
    data.extend(struct.pack("<12f", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0))

    return bytes(data)


def _pack_animation(components: list[dict], frame_count: int, use_16bit_indices: bool) -> bytes:
    data = bytearray()

    index_size = 1 if use_16bit_indices else 0  # IntPrecision: 0=_8bit,1=_16bit
    float_size = 1  # FloatPrecision: 1=_16bit (half)
    node_count = 1 if components else 0

    data.extend(b"\x00\x00")  # padding/flags
    data.append(index_size)
    data.append(float_size)
    data.extend(struct.pack("<I", frame_count))
    data.extend(struct.pack("<I", node_count))
    data.extend(struct.pack("<I", 0))  # node table rel placeholder

    if node_count:
        node_table_rel = len(data)
        data[12:16] = struct.pack("<I", node_table_rel)

        # Node table (one entry)
        node_rel = 20  # match vanilla camera EANs (header16 + table4 -> node starts at +20)
        data.extend(struct.pack("<I", node_rel))

        node_start = len(data)
        # Node header
        data.extend(struct.pack("<h", 0))  # bone index
        data.extend(struct.pack("<h", len(components)))  # component count
        data.extend(struct.pack("<I", 8))  # component table rel

        comp_table_start = len(data)
        # placeholder comp offsets
        for _ in components:
            data.extend(b"\x00\x00\x00\x00")

        for idx, comp in enumerate(components):
            comp_start = len(data)
            comp_rel = comp_start - node_start
            data[comp_table_start + idx * 4 : comp_table_start + idx * 4 + 4] = struct.pack(
                "<I", comp_rel
            )

            keyframes = comp["keyframes"]
            data.extend(
                struct.pack("<BBhI", comp["type"], comp["i01"], comp["i02"], len(keyframes))
            )
            data.extend(struct.pack("<I", 16))  # index list rel

            idx_bytes = bytearray()
            for keyframe in keyframes:
                frame = keyframe[0]
                if use_16bit_indices:
                    idx_bytes.extend(struct.pack("<H", frame))
                else:
                    idx_bytes.extend(struct.pack("<B", frame))

            float_bytes = bytearray()
            for _, x, y, z, w in keyframes:
                float_bytes.extend(_pack_half(x))
                float_bytes.extend(_pack_half(y))
                float_bytes.extend(_pack_half(z))
                float_bytes.extend(_pack_half(w))

            float_rel = 16 + len(idx_bytes)
            data.extend(struct.pack("<I", float_rel))

            data.extend(idx_bytes)
            data.extend(float_bytes)

    return bytes(data)


def _has_node_actions() -> bool:
    return any(
        action.name.startswith("Node_") and not action.name.endswith("_data")
        for action in bpy.data.actions
    )


def _has_legacy_pairs() -> bool:
    plus = {action.name[1:] for action in bpy.data.actions if action.name.startswith("+")}
    minus = {action.name[1:] for action in bpy.data.actions if action.name.startswith("-")}
    return bool(plus & minus)


def _detect_cam_mode(active: bpy.types.Object | None) -> str | None:
    """Pick the export path.

    A file can hold both conventions at once (e.g. an importer-built camera later re-rigged into
    the legacy ``+Name`` / ``-Name`` control rig). The active object's current action tells us
    which rig is wired up, so it wins over a scan of every action in the file.
    """
    if active is not None:
        if active.type == "CAMERA":
            action = active.animation_data.action if active.animation_data else None
            if action and action.name.startswith("+"):
                return "legacy"
        elif active.type == "EMPTY" and any(child.type == "CAMERA" for child in active.children):
            return "importer"

    if _has_node_actions():
        return "importer"
    if _has_legacy_pairs():
        return "legacy"
    return None


CollectedAnimations = tuple[list[bytes], list[int], bytearray]


def _collect_importer_rig_animations(
    rig_obj: bpy.types.Object | None,
    bake_visual_keying: bool = True,
) -> CollectedAnimations | None:
    cam_obj = None
    target_obj = None

    if rig_obj:
        for child in rig_obj.children:
            if child.type == "CAMERA":
                cam_obj = child
            elif child.type == "EMPTY" and child.name.lower().startswith("cameratarget"):
                target_obj = child
    if cam_obj is None:
        cam_obj = (
            bpy.context.object
            if bpy.context.object and bpy.context.object.type == "CAMERA"
            else None
        )
    if cam_obj is None:
        return None

    if target_obj is None:
        for constraint in cam_obj.constraints:
            if constraint.type == "TRACK_TO" and getattr(constraint, "target", None):
                target_obj = constraint.target
                break

    base_entries: list[tuple[int | None, str]] = []
    for action in bpy.data.actions:
        name = action.name
        if name.startswith("Node_") and not name.endswith("_data"):
            base = name[len("Node_") :]
            idx_val = action.get("ean_index")
            idx_int = idx_val if isinstance(idx_val, int) else None
            base_entries.append((idx_int, base))
    # Deduplicate by base name, preferring the first occurrence (which keeps index if present).
    seen = set()
    deduped: list[tuple[int | None, str]] = []
    for entry in base_entries:
        if entry[1] in seen:
            continue
        seen.add(entry[1])
        deduped.append(entry)
    # Sort by explicit index if available; fall back to name.
    base_entries_sorted = sorted(
        deduped, key=lambda e: (e[0] is None, e[0] if e[0] is not None else e[1])
    )
    base_names = [entry[1] for entry in base_entries_sorted]
    if not base_names:
        return None

    animations_bytes: list[bytes] = []
    name_offsets: list[int] = []
    names_blob = bytearray()

    scene = bpy.context.scene
    depsgraph = bpy.context.view_layer.depsgraph
    original_frame = scene.frame_current

    for base in base_names:
        cam_action = bpy.data.actions.get(f"Node_{base}")
        target_action = bpy.data.actions.get(f"Target_{base}")
        data_action = bpy.data.actions.get(f"Node_{base}_data")

        # Collect frames from the source actions
        frames = set()
        frames.update(_collect_frames_from_action(cam_action, ("location",)))
        frames.update(_collect_frames_from_action(target_action, ("location",)))
        frames.update(_collect_frames_from_action(data_action, ("xv2_roll", "xv2_fov")))
        if not frames:
            frames.add(0)
        if bake_visual_keying:
            # Sample every frame to keep constraint and driver motion that sits between sparse keys.
            frames = set(range(min(frames), max(frames) + 1))
        frame_count = max(frames) + 1
        use_16bit_indices = frame_count > 255

        # Temporarily assign actions so we can sample evaluated transforms (visual keying)
        # without baking.
        cam_anim_data = cam_obj.animation_data or cam_obj.animation_data_create()
        orig_cam_action = cam_anim_data.action
        cam_anim_data.action = cam_action

        target_anim_created = False
        target_anim_data = target_obj.animation_data if target_obj else None
        orig_target_action = target_anim_data.action if target_anim_data else None
        if target_obj:
            if target_obj.animation_data is None:
                target_obj.animation_data_create()
                target_anim_created = True
            target_obj.animation_data.action = target_action

        data_anim_created = False
        data_anim_data = (
            cam_obj.data.animation_data if hasattr(cam_obj.data, "animation_data") else None
        )
        orig_data_action = data_anim_data.action if data_anim_data else None
        if data_action:
            if cam_obj.data.animation_data is None:
                cam_obj.data.animation_data_create()
                data_anim_created = True
            cam_obj.data.animation_data.action = data_action

        components: list[dict] = []

        pos_keyframes: list[tuple[int, float, float, float, float]] = []
        scale_keyframes: list[tuple[int, float, float, float, float]] = []
        target_keyframes: list[tuple[int, float, float, float, float]] = []

        for frame in sorted(frames):
            scene.frame_set(frame)
            if hasattr(bpy.context.view_layer, "update"):
                bpy.context.view_layer.update()

            cam_eval = cam_obj.evaluated_get(depsgraph)
            cam_loc = cam_eval.matrix_world.translation
            pos_keyframes.append((frame, *_map_vec_to_xv2(cam_loc.x, cam_loc.y, cam_loc.z), 1.0))

            if bake_visual_keying:
                # Read the evaluated camera so drivers and constraints on roll and FOV get baked.
                # FOV comes from the final lens rather than the raw xv2_fov property.
                cam_data_eval = cam_eval.data
                roll_val = getattr(
                    cam_data_eval, "xv2_roll", getattr(cam_obj.data, "xv2_roll", 0.0)
                )
                lens = getattr(cam_data_eval, "lens", 0.0)
                sensor = getattr(cam_data_eval, "sensor_height", 0.0) or getattr(
                    cam_obj.data, "sensor_height", 32.0
                )
                if lens > 1e-6 and sensor > 1e-6:
                    fov_rad = 2.0 * math.atan(sensor / (2.0 * lens))
                else:
                    fov_rad = math.radians(getattr(cam_data_eval, "xv2_fov", 40.0))
                scale_keyframes.append((frame, -math.radians(roll_val), fov_rad, 0.0, 0.0))
            else:
                roll_val = _eval_scalar(
                    data_action, "xv2_roll", frame, getattr(cam_obj.data, "xv2_roll", 0.0)
                )
                fov_val = _eval_scalar(
                    data_action, "xv2_fov", frame, getattr(cam_obj.data, "xv2_fov", 40.0)
                )
                scale_keyframes.append(
                    (frame, -math.radians(roll_val), math.radians(fov_val), 0.0, 0.0)
                )

            if target_obj:
                targ_eval = target_obj.evaluated_get(depsgraph)
                targ_loc = targ_eval.matrix_world.translation
                target_keyframes.append(
                    (frame, *_map_vec_to_xv2(targ_loc.x, targ_loc.y, targ_loc.z), 1.0)
                )

        pos_keyframes = _calc_edge_frames(pos_keyframes, frame_count)
        components.append(
            {
                "type": ComponentType.Position,
                "i01": 3,
                "i02": 0,
                "keyframes": pos_keyframes,
            }
        )

        scale_keyframes = _calc_edge_frames(scale_keyframes, frame_count)
        components.append(
            {
                "type": ComponentType.Scale,
                "i01": 3,
                "i02": 0,
                "keyframes": scale_keyframes,
            }
        )

        if target_obj and target_keyframes:
            target_keyframes = _calc_edge_frames(target_keyframes, frame_count)
            components.append(
                {
                    "type": ComponentType.Rotation,
                    "i01": 3,
                    "i02": 0,
                    "keyframes": target_keyframes,
                }
            )

        # Restore actions and frame to avoid baking/persisting changes.
        cam_anim_data.action = orig_cam_action
        if target_obj and target_obj.animation_data:
            target_obj.animation_data.action = orig_target_action
            if target_anim_created and target_obj.animation_data is not None:
                # clear if we created it solely for sampling
                target_obj.animation_data.action = None
        if data_action and cam_obj.data.animation_data:
            cam_obj.data.animation_data.action = orig_data_action
            if data_anim_created and cam_obj.data.animation_data is not None:
                cam_obj.data.animation_data.action = None
        scene.frame_set(original_frame)

        anim_bytes = _pack_animation(components, frame_count, use_16bit_indices=use_16bit_indices)
        animations_bytes.append(anim_bytes)
        name_offsets.append(len(names_blob))
        names_blob.extend(base.encode("ascii", "ignore") + b"\x00")

    return animations_bytes, name_offsets, names_blob


def _find_legacy_target(
    cam_obj: bpy.types.Object, target_action: bpy.types.Action
) -> bpy.types.Object | None:
    owner = next(
        (
            obj
            for obj in bpy.data.objects
            if obj.animation_data and obj.animation_data.action is target_action
        ),
        None,
    )
    if owner is not None:
        return owner
    for constraint in cam_obj.constraints:
        if constraint.type == "TRACK_TO" and getattr(constraint, "target", None):
            return constraint.target
    return None


def _collect_legacy_animations() -> CollectedAnimations | None:
    cam_obj = (
        bpy.context.object
        if bpy.context.object and bpy.context.object.type == "CAMERA"
        else None
    )
    if cam_obj is None:
        cam_obj = next(
            (obj for obj in bpy.data.objects if obj.type == "CAMERA" and obj.name == "Node"),
            None,
        )
    if cam_obj is None:
        cam_obj = next((obj for obj in bpy.data.objects if obj.type == "CAMERA"), None)
    if cam_obj is None:
        return None

    # Pair +Name / -Name actions by the shared suffix.
    plus_actions: dict[str, bpy.types.Action] = {}
    minus_actions: dict[str, bpy.types.Action] = {}
    for action in bpy.data.actions:
        name = action.name
        if name.startswith("+"):
            plus_actions.setdefault(name[1:], action)
        elif name.startswith("-"):
            minus_actions.setdefault(name[1:], action)

    base_names = sorted(base for base in plus_actions if base in minus_actions)
    if not base_names:
        return None

    scene = bpy.context.scene
    depsgraph = bpy.context.view_layer.depsgraph
    original_frame = scene.frame_current

    animations_bytes: list[bytes] = []
    name_offsets: list[int] = []
    names_blob = bytearray()

    for base in base_names:
        cam_action = plus_actions[base]
        target_action = minus_actions[base]
        target_obj = _find_legacy_target(cam_obj, target_action)

        # Motion comes from the constraint stack, so sample every frame across the pair.
        frame_count = max(int(cam_action.frame_range[1]), int(target_action.frame_range[1])) + 1
        use_16bit_indices = frame_count > 255
        sensor = getattr(cam_obj.data, "sensor_width", 32.0)

        # Temporarily assign the paired actions so evaluated sampling bakes the rig. The control
        # empties keep their own actions so their constraints still drive the camera.
        cam_anim_data = cam_obj.animation_data or cam_obj.animation_data_create()
        orig_cam_action = cam_anim_data.action
        cam_anim_data.action = cam_action

        target_anim_created = False
        orig_target_action = None
        if target_obj:
            if target_obj.animation_data is None:
                target_obj.animation_data_create()
                target_anim_created = True
            orig_target_action = target_obj.animation_data.action
            target_obj.animation_data.action = target_action

        pos_keyframes: list[tuple[int, float, float, float, float]] = []
        scale_keyframes: list[tuple[int, float, float, float, float]] = []
        target_keyframes: list[tuple[int, float, float, float, float]] = []

        for frame in range(frame_count):
            scene.frame_set(frame)
            if hasattr(bpy.context.view_layer, "update"):
                bpy.context.view_layer.update()

            cam_eval = cam_obj.evaluated_get(depsgraph)
            cam_loc = cam_eval.matrix_world.translation
            pos_keyframes.append((frame, *_map_vec_to_xv2(cam_loc.x, cam_loc.y, cam_loc.z), 1.0))

            scale_y = cam_eval.matrix_world.to_scale().y
            fov = 2.0 * math.atan(sensor / (2.0 * scale_y)) if abs(scale_y) > 1e-6 else 0.0
            roll = _eval_scalar(target_action, "rotation_euler", frame, 0.0, index=1)
            # Legacy quirk: flip roll when the camera's raw + action location Y is non-negative.
            if _eval_scalar(cam_action, "location", frame, 0.0, index=1) >= 0:
                roll = -roll
            scale_keyframes.append((frame, roll, fov, 0.0, 0.0))

            if target_obj:
                targ_eval = target_obj.evaluated_get(depsgraph)
                targ_loc = targ_eval.matrix_world.translation
                target_keyframes.append(
                    (frame, *_map_vec_to_xv2(targ_loc.x, targ_loc.y, targ_loc.z), 1.0)
                )

        components: list[dict] = []
        pos_keyframes = _calc_edge_frames(pos_keyframes, frame_count)
        components.append(
            {"type": ComponentType.Position, "i01": 3, "i02": 0, "keyframes": pos_keyframes}
        )

        scale_keyframes = _calc_edge_frames(scale_keyframes, frame_count)
        components.append(
            {"type": ComponentType.Scale, "i01": 3, "i02": 0, "keyframes": scale_keyframes}
        )

        if target_obj and target_keyframes:
            target_keyframes = _calc_edge_frames(target_keyframes, frame_count)
            components.append(
                {"type": ComponentType.Rotation, "i01": 3, "i02": 0, "keyframes": target_keyframes}
            )

        cam_anim_data.action = orig_cam_action
        if target_obj and target_obj.animation_data:
            target_obj.animation_data.action = orig_target_action
            if target_anim_created:
                target_obj.animation_data.action = None
        scene.frame_set(original_frame)

        anim_bytes = _pack_animation(components, frame_count, use_16bit_indices=use_16bit_indices)
        animations_bytes.append(anim_bytes)
        name_offsets.append(len(names_blob))
        names_blob.extend(base.encode("ascii", "ignore") + b"\x00")

    return animations_bytes, name_offsets, names_blob


def _assemble_cam_ean(
    animations_bytes: list[bytes], name_offsets: list[int], names_blob: bytearray
) -> bytes:
    out = bytearray([35, 69, 65, 78, 254, 255, 32, 0])
    out.extend(struct.pack("<I", 37568))
    out.extend(b"\x00\x00\x00\x00")
    out.append(1)  # is_camera true
    out.append(4)
    out.extend(struct.pack("<H", len(animations_bytes)))
    out.extend(b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

    skeleton_bytes = _write_skeleton_single_node()
    skeleton_offset = len(out)
    out[20:24] = struct.pack("<I", skeleton_offset)
    out.extend(skeleton_bytes)

    if animations_bytes:
        anim_table_off = len(out)
        out[24:28] = struct.pack("<I", anim_table_off)
        for _ in animations_bytes:
            out.extend(b"\x00\x00\x00\x00")
        for i, anim_bytes in enumerate(animations_bytes):
            out[anim_table_off + i * 4 : anim_table_off + i * 4 + 4] = struct.pack("<I", len(out))
            out.extend(anim_bytes)
        name_table_off = len(out)
        out[28:32] = struct.pack("<I", name_table_off)
        for _ in animations_bytes:
            out.extend(b"\x00\x00\x00\x00")
        for i, off in enumerate(name_offsets):
            out[name_table_off + i * 4 : name_table_off + i * 4 + 4] = struct.pack("<I", len(out))
            end = names_blob.find(b"\x00", off)
            out.extend(names_blob[off : end + 1])

    return bytes(out)


def export_cam_ean(
    filepath: str,
    rig_obj: bpy.types.Object | None = None,
    bake_visual_keying: bool = True,
) -> bool:
    if rig_obj is None:
        rig_obj = (
            bpy.context.object
            if bpy.context.object and bpy.context.object.type == "EMPTY"
            else None
        )

    mode = _detect_cam_mode(bpy.context.object)
    if mode == "importer":
        collected = _collect_importer_rig_animations(rig_obj, bake_visual_keying)
    elif mode == "legacy":
        collected = _collect_legacy_animations()
    else:
        return False

    if not collected:
        return False
    animations_bytes, name_offsets, names_blob = collected
    if not animations_bytes:
        return False

    Path(filepath).write_bytes(_assemble_cam_ean(animations_bytes, name_offsets, names_blob))
    return True


__all__ = ["export_cam_ean"]
