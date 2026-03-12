from __future__ import annotations

import contextlib
import os
import struct
import tempfile
from pathlib import Path

import bpy

from ..EAN import ComponentType, read_ean
from ..EAN.exporter import export_ean
from ..ESK import ESK_File

EMA_SIGNATURE = 1095583011  # "#EMA"
EMA_TYPE_OBJ = 3
EMA_ANIM_TYPE_OBJ = 0
EMA_FLOAT32 = 2


def _pad_data(data: bytearray, alignment: int) -> None:
    padding = (-len(data)) % alignment
    if padding:
        data.extend(b"\x00" * padding)


def _to_u16_index(value: int) -> int:
    if value < 0:
        return 0xFFFF
    return value & 0xFFFF


def _component_parameter(component_type: ComponentType) -> int | None:
    if component_type == ComponentType.Position:
        return 0
    if component_type == ComponentType.Rotation:
        return 1
    if component_type == ComponentType.Scale:
        return 2
    return None


def _build_ema_skeleton_bytes(skeleton: ESK_File) -> bytes:
    bones = list(skeleton.bones)
    bone_count = len(bones)

    out = bytearray()
    out.extend(struct.pack("<H", bone_count & 0xFFFF))
    out.extend(struct.pack("<H", 0))  # IK2 count
    out.extend(struct.pack("<I", 0))  # IK count
    out.extend(struct.pack("<I", 64))  # bones offset
    names_offset_field = len(out)
    out.extend(struct.pack("<I", 0))  # names offset
    out.extend(struct.pack("<I", 0))  # IK2 offset
    out.extend(struct.pack("<I", 0))  # IK2 names offset
    out.extend(struct.pack("<I", 0))  # extra values offset
    out.extend(struct.pack("<I", 0))  # abs matrix offset
    out.extend(struct.pack("<I", 0))  # IK offset
    out.extend(struct.pack("<i", 0))  # I_36
    out.extend(struct.pack("<i", 0))  # I_40
    out.extend(struct.pack("<i", 0))  # I_44
    out.extend(struct.pack("<i", 0))  # I_48
    out.extend(struct.pack("<H", 0))  # I_52
    out.extend(struct.pack("<H", int(skeleton.skeleton_flag) & 0xFFFF))
    out.extend(struct.pack("<Q", int(skeleton.skeleton_id) & 0xFFFFFFFFFFFFFFFF))

    for bone in bones:
        out.extend(struct.pack("<H", _to_u16_index(int(bone.parent_index))))
        out.extend(struct.pack("<H", _to_u16_index(int(bone.child_index))))
        out.extend(struct.pack("<H", _to_u16_index(int(bone.sibling_index))))
        out.extend(struct.pack("<H", 0xFFFF))  # EmoPartIndex
        out.extend(struct.pack("<H", 0xFFFF))  # I_08
        out.extend(struct.pack("<H", 0))  # IK flag
        out.extend(struct.pack("<f", 0.0))  # F_12
        matrix = bone.matrix.transposed()
        for row in range(4):
            for col in range(4):
                out.extend(struct.pack("<f", float(matrix[row][col])))

    names_offset = len(out)
    struct.pack_into("<I", out, names_offset_field, names_offset)
    names_ptr_start = len(out)
    out.extend(b"\x00" * (4 * bone_count))
    for bone_index, bone in enumerate(bones):
        name_offset = len(out)
        struct.pack_into("<I", out, names_ptr_start + (bone_index * 4), name_offset)
        out.extend(str(bone.name).encode("utf8", errors="ignore") + b"\x00")

    _pad_data(out, 16)
    return bytes(out)


def _build_ema_animation_bytes(animation, bone_index_by_name: dict[str, int]) -> bytes:
    commands: list[tuple[int, int, int, list[tuple[int, float]]]] = []
    end_frame = 0

    for node in animation.nodes:
        bone_index = bone_index_by_name.get(node.bone_name)
        if bone_index is None:
            continue
        for component in node.components:
            parameter = _component_parameter(component.type)
            if parameter is None:
                continue
            keyframes = sorted(component.keyframes, key=lambda key: key.frame_index)
            if not keyframes:
                continue
            axis_values = (
                [(int(key.frame_index), float(key.x)) for key in keyframes],
                [(int(key.frame_index), float(key.y)) for key in keyframes],
                [(int(key.frame_index), float(key.z)) for key in keyframes],
                [(int(key.frame_index), float(key.w)) for key in keyframes],
            )
            for axis, values in enumerate(axis_values):
                commands.append((int(bone_index), int(parameter), axis, values))
                end_frame = max(end_frame, max(frame for frame, _value in values))

    if not commands:
        return b""

    out = bytearray()
    out.extend(struct.pack("<H", end_frame & 0xFFFF))
    out.extend(struct.pack("<H", len(commands) & 0xFFFF))
    values_count_field = len(out)
    out.extend(struct.pack("<I", 0))  # value count
    out.extend(struct.pack("<B", EMA_ANIM_TYPE_OBJ))
    out.extend(struct.pack("<B", 0))  # light unknown
    out.extend(struct.pack("<H", EMA_FLOAT32))
    name_offset_field = len(out)
    out.extend(struct.pack("<I", 0))
    values_offset_field = len(out)
    out.extend(struct.pack("<I", 0))

    command_ptrs_start = len(out)
    out.extend(b"\x00" * (4 * len(commands)))
    values: list[float] = []

    for command_index, (bone_index, parameter, axis, keyframes) in enumerate(commands):
        command_start = len(out)
        struct.pack_into("<I", out, command_ptrs_start + (command_index * 4), command_start)

        int16_for_time = any(frame > 255 for frame, _value in keyframes)
        flags_a = axis & 0x03
        flags_b = 0x04  # Int16ForValueIndex = true
        if int16_for_time:
            flags_b |= 0x02
        flags = (flags_a & 0x0F) | ((flags_b & 0x0F) << 4)

        out.extend(struct.pack("<H", bone_index & 0xFFFF))
        out.extend(struct.pack("<B", parameter & 0xFF))
        out.extend(struct.pack("<B", flags & 0xFF))
        out.extend(struct.pack("<H", len(keyframes) & 0xFFFF))
        index_offset_field = len(out)
        out.extend(struct.pack("<H", 0))

        for frame, _value in keyframes:
            if frame < 0 or frame > 0xFFFF:
                raise ValueError("EMA export supports keyframes in 0..65535 range.")
            if int16_for_time:
                out.extend(struct.pack("<H", frame & 0xFFFF))
            else:
                out.extend(struct.pack("<B", frame & 0xFF))

        while ((len(out) - command_start) % 4) != 0:
            out.extend(b"\x00")

        index_offset = len(out) - command_start
        struct.pack_into("<H", out, index_offset_field, index_offset & 0xFFFF)

        for _frame, value in keyframes:
            value_index = len(values)
            if value_index > 0xFFFF:
                raise ValueError("EMA export value table exceeded 65535 entries.")
            values.append(float(value))
            out.extend(struct.pack("<H", value_index & 0xFFFF))
            out.extend(struct.pack("<B", 0))  # padding
            out.extend(struct.pack("<B", 0))  # linear interpolation

        while ((len(out) - command_start) % 4) != 0:
            out.extend(b"\x00")

    struct.pack_into("<I", out, values_count_field, len(values))
    struct.pack_into("<I", out, values_offset_field, len(out))
    for value in values:
        out.extend(struct.pack("<f", value))
    _pad_data(out, 4)

    animation_name = str(animation.name or animation.index)
    if animation_name:
        name_blob = animation_name.encode("utf8", errors="ignore")
        if len(name_blob) > 255:
            name_blob = name_blob[:255]
        struct.pack_into("<I", out, name_offset_field, len(out))
        out.extend(b"\x00" * 10)
        out.extend(struct.pack("<B", len(name_blob)))
        out.extend(name_blob)
        out.extend(b"\x00")

    return bytes(out)


def ean_to_ema_bytes(
    ean_file,
    *,
    version: int = 0x92C0,
    i_20: int = 0,
    i_24: int = 0,
    i_28: int = 0,
) -> bytes:
    if not ean_file.skeleton:
        raise ValueError("EAN data has no skeleton; EMA export requires a skeleton.")

    bone_index_by_name = {
        bone.name: bone.index for bone in ean_file.skeleton.bones if getattr(bone, "name", None)
    }
    animations_by_index: dict[int, bytes] = {}
    max_index = -1
    for animation in ean_file.animations:
        animation_blob = _build_ema_animation_bytes(animation, bone_index_by_name)
        if not animation_blob:
            continue
        animations_by_index[int(animation.index)] = animation_blob
        max_index = max(max_index, int(animation.index))

    if max_index < 0:
        raise ValueError("No exportable animation data was found.")

    animation_count = max_index + 1
    out = bytearray()
    out.extend(struct.pack("<I", EMA_SIGNATURE))
    out.extend(struct.pack("<H", 0xFFFE))
    out.extend(struct.pack("<H", 32))
    out.extend(struct.pack("<I", int(version)))
    skeleton_offset_field = len(out)
    out.extend(struct.pack("<I", 0))
    out.extend(struct.pack("<H", animation_count & 0xFFFF))
    out.extend(struct.pack("<H", EMA_TYPE_OBJ))
    out.extend(struct.pack("<I", int(i_20)))
    out.extend(struct.pack("<I", int(i_24)))
    out.extend(struct.pack("<I", int(i_28)))

    animation_ptrs_start = len(out)
    out.extend(b"\x00" * (4 * animation_count))
    for animation_index in range(animation_count):
        animation_blob = animations_by_index.get(animation_index)
        if animation_blob is None:
            continue
        struct.pack_into("<I", out, animation_ptrs_start + (animation_index * 4), len(out))
        out.extend(animation_blob)

    _pad_data(out, 16)
    struct.pack_into("<I", out, skeleton_offset_field, len(out))
    out.extend(_build_ema_skeleton_bytes(ean_file.skeleton))
    return bytes(out)


def export_ema(
    filepath: str,
    arm_obj: bpy.types.Object,
    *,
    add_dummy_rest: bool = False,
) -> tuple[bool, str | None]:
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return False, "Select an armature to export."

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".ean", delete=False) as temp_file:
            temp_path = temp_file.name

        ok, error = export_ean(temp_path, arm_obj, add_dummy_rest=add_dummy_rest)
        if not ok:
            return False, error or "Failed to export intermediate EAN data."

        ean_file = read_ean(temp_path, link_skeleton=True)
        version = int(arm_obj.get("ema_version", 0x92C0))
        i_20 = int(arm_obj.get("ema_i20", 0))
        i_24 = int(arm_obj.get("ema_i24", 0))
        i_28 = int(arm_obj.get("ema_i28", 0))
        ema_bytes = ean_to_ema_bytes(
            ean_file,
            version=version,
            i_20=i_20,
            i_24=i_24,
            i_28=i_28,
        )
        Path(filepath).write_bytes(ema_bytes)
        return True, None
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        return False, f"Unexpected error while exporting EMA: {exc}"
    finally:
        if temp_path and os.path.isfile(temp_path):
            with contextlib.suppress(OSError):
                os.remove(temp_path)


__all__ = ["ean_to_ema_bytes", "export_ema"]
