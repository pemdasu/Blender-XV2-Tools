from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from ...utils import half_to_float, read_cstring
from ...utils.binary import u16, u32
from ..consts import (
    EMA_ANIM_TYPE_OBJ,
    EMA_DEFAULT_POSITION,
    EMA_DEFAULT_ROTATION_EULER,
    EMA_DEFAULT_SCALE,
    EMA_SIGNATURE,
    EMA_TYPE_OBJ,
    INTERP_CUBIC,
    INTERP_LINEAR,
    INTERP_QUADRATIC,
    PARAM_POSITION,
    PARAM_ROTATION,
    PARAM_SCALE,
)
from ..EAN import (
    ComponentType,
    EANAnimation,
    EANAnimationComponent,
    EANFile,
    EANKeyframe,
    EANNode,
    FloatPrecision,
)
from ..ESK import ESK_File
from .skeleton import parse_ema_skeleton_as_esk

RestTuple = tuple[float, float, float, float]


@dataclass
class EMAKeyframe:
    time: int
    value: float
    interpolation: int = INTERP_LINEAR
    control_point1: float = 0.0
    control_point2: float = 0.0


@dataclass
class EMACommand:
    parameter: int
    component: int
    keyframes: list[EMAKeyframe] = field(default_factory=list)
    no_interpolation: bool = False


@dataclass
class EMANode:
    bone_name: str
    commands: list[EMACommand] = field(default_factory=list)


@dataclass
class EMAAnimation:
    name: str
    index: int
    ema_type: int
    float_precision: int
    end_frame: int
    nodes: list[EMANode] = field(default_factory=list)


@dataclass
class EMAFile:
    version: int
    ema_type: int
    i_20: int
    i_24: int
    i_28: int
    skeleton: ESK_File | None
    animations: list[EMAAnimation] = field(default_factory=list)


def parse_ema(path: str) -> EMAFile:
    with open(path, "rb") as file_handle:
        data = file_handle.read()
    return parse_ema_bytes(data)


def parse_ema_bytes(data: bytes) -> EMAFile:
    if u32(data, 0) != EMA_SIGNATURE:
        raise ValueError('EMA signature "#EMA" not found at offset 0.')

    version = int(u32(data, 8))
    skeleton_offset = int(u32(data, 12))
    animation_count = int(u16(data, 16))
    ema_type = int(u16(data, 18))
    i_20 = int(u32(data, 20))
    i_24 = int(u32(data, 24))
    i_28 = int(u32(data, 28))

    skeleton: ESK_File | None = None
    if skeleton_offset > 0:
        skeleton, _ = parse_ema_skeleton_as_esk(data, skeleton_offset)

    animations: list[EMAAnimation] = []
    for animation_index in range(animation_count):
        animation_offset = int(u32(data, 32 + (animation_index * 4)))
        if animation_offset == 0:
            continue
        animations.append(
            _parse_animation(
                data=data,
                animation_offset=animation_offset,
                animation_index=animation_index,
                skeleton=skeleton,
            )
        )

    return EMAFile(
        version=version,
        ema_type=ema_type,
        i_20=i_20,
        i_24=i_24,
        i_28=i_28,
        skeleton=skeleton,
        animations=animations,
    )


def _parse_animation(
    *,
    data: bytes,
    animation_offset: int,
    animation_index: int,
    skeleton: ESK_File | None,
) -> EMAAnimation:
    end_frame = int(u16(data, animation_offset + 0))
    command_count = int(u16(data, animation_offset + 2))
    value_count = int(u32(data, animation_offset + 4))
    animation_type = int(data[animation_offset + 8])
    float_precision = int(u16(data, animation_offset + 10))
    name_rel = int(u32(data, animation_offset + 12))
    values_rel = int(u32(data, animation_offset + 16))

    name = str(animation_index)
    if name_rel:
        # EMA stores an 11-byte name prefix before the null-terminated text.
        parsed_name = read_cstring(data, animation_offset + name_rel + 11)
        if parsed_name:
            name = parsed_name

    values_offset = animation_offset + values_rel
    values: list[float] = []
    for value_index in range(value_count):
        if float_precision == 1:
            values.append(half_to_float(u16(data, values_offset + (value_index * 2))))
        elif float_precision in {0, 2}:
            values.append(struct.unpack_from("<f", data, values_offset + (value_index * 4))[0])
        else:
            raise ValueError(f"Unsupported EMA float precision: {float_precision}")

    nodes_by_name: dict[str, EMANode] = {}
    for command_index in range(command_count):
        command_rel = int(u32(data, animation_offset + 20 + (command_index * 4)))
        if command_rel == 0:
            continue
        command_offset = animation_offset + command_rel
        bone_index = int(u16(data, command_offset + 0))

        if skeleton is not None and 0 <= bone_index < len(skeleton.bones):
            bone_name = skeleton.bones[bone_index].name
        else:
            bone_name = f"bone_{bone_index}"

        node = nodes_by_name.setdefault(bone_name, EMANode(bone_name=bone_name))
        node.commands.append(_parse_command(data, command_offset, values))

    return EMAAnimation(
        name=name,
        index=animation_index,
        ema_type=animation_type,
        float_precision=float_precision,
        end_frame=end_frame,
        nodes=list(nodes_by_name.values()),
    )


def _parse_command(data: bytes, command_offset: int, values: list[float]) -> EMACommand:
    parameter = int(data[command_offset + 2])
    flags = int(data[command_offset + 3])
    flags_a = flags & 0x0F
    flags_b = (flags >> 4) & 0x0F
    component = flags_a & 0x03
    no_interpolation = bool(flags_a & 0x04)
    int16_for_time = bool(flags_b & 0x02)
    int16_for_value_index = bool(flags_b & 0x04)
    keyframe_count = int(u16(data, command_offset + 4))
    index_offset = int(u16(data, command_offset + 6))

    keyframes: list[EMAKeyframe] = []
    for keyframe_index in range(keyframe_count):
        if int16_for_time:
            time = int(u16(data, command_offset + 8 + (keyframe_index * 2)))
        else:
            time = int(data[command_offset + 8 + keyframe_index])

        base_index = 0
        interpolation = INTERP_LINEAR
        if int16_for_value_index:
            base_index = int(u16(data, command_offset + index_offset + (keyframe_index * 4)))
            interpolation = (
                int(data[command_offset + index_offset + 3 + (keyframe_index * 4)]) & 0xC0
            )
        else:
            packed = int(u16(data, command_offset + index_offset + (keyframe_index * 2)))
            base_index = packed & 0x3FFF
            interpolation_bits = packed & 0xC000
            if interpolation_bits == 0x4000:
                interpolation = INTERP_QUADRATIC
            elif interpolation_bits == 0x8000:
                interpolation = INTERP_CUBIC

        value = values[base_index] if 0 <= base_index < len(values) else 0.0
        control_point1 = 0.0
        control_point2 = 0.0
        if interpolation == INTERP_QUADRATIC:
            cp1_index = base_index + 1
            if cp1_index < len(values):
                control_point1 = values[cp1_index]
        elif interpolation == INTERP_CUBIC:
            cp1_index = base_index + 1
            cp2_index = base_index + 2
            if cp1_index < len(values):
                control_point1 = values[cp1_index]
            if cp2_index < len(values):
                control_point2 = values[cp2_index]

        keyframes.append(
            EMAKeyframe(
                time=time,
                value=value,
                interpolation=interpolation,
                control_point1=control_point1,
                control_point2=control_point2,
            )
        )

    keyframes.sort(key=lambda keyframe: keyframe.time)
    return EMACommand(
        parameter=parameter,
        component=component,
        keyframes=keyframes,
        no_interpolation=no_interpolation,
    )


def _quadratic_bezier(t: float, p0: float, p1: float, p2: float) -> float:
    one_minus_t = 1.0 - t
    return (one_minus_t * one_minus_t * p0) + (2.0 * one_minus_t * t * p1) + (t * t * p2)


def _cubic_bezier(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    one_minus_t = 1.0 - t
    return (
        (one_minus_t * one_minus_t * one_minus_t * p0)
        + (3.0 * one_minus_t * one_minus_t * t * p1)
        + (3.0 * one_minus_t * t * t * p2)
        + (t * t * t * p3)
    )


def _evaluate_command(command: EMACommand | None, frame: int, default_value: float) -> float:
    if command is None or not command.keyframes:
        return default_value

    previous_keyframe: EMAKeyframe | None = None
    next_keyframe: EMAKeyframe | None = None
    for keyframe in command.keyframes:
        if keyframe.time == frame:
            return keyframe.value
        if keyframe.time < frame:
            previous_keyframe = keyframe
            continue
        next_keyframe = keyframe
        break

    if previous_keyframe is None and next_keyframe is None:
        return default_value
    if previous_keyframe is None and next_keyframe is not None:
        return next_keyframe.value
    if previous_keyframe is not None and next_keyframe is None:
        return previous_keyframe.value

    assert previous_keyframe is not None
    assert next_keyframe is not None
    if next_keyframe.time <= previous_keyframe.time:
        return previous_keyframe.value

    factor = (frame - previous_keyframe.time) / float(next_keyframe.time - previous_keyframe.time)
    if previous_keyframe.interpolation == INTERP_CUBIC:
        control_point1 = previous_keyframe.value + previous_keyframe.control_point1
        control_point2 = next_keyframe.value - previous_keyframe.control_point2
        return _cubic_bezier(
            factor,
            previous_keyframe.value,
            control_point1,
            control_point2,
            next_keyframe.value,
        )
    if previous_keyframe.interpolation == INTERP_QUADRATIC:
        control_point1 = previous_keyframe.value + previous_keyframe.control_point1
        return _quadratic_bezier(
            factor,
            previous_keyframe.value,
            control_point1,
            next_keyframe.value,
        )
    return previous_keyframe.value + ((next_keyframe.value - previous_keyframe.value) * factor)


def _euler_degrees_to_quaternion(x_deg: float, y_deg: float, z_deg: float) -> RestTuple:
    # Match Xv2CoreLib MathHelpers.EulerToQuaternion (roll=X, pitch=Y, yaw=Z in degrees).
    roll = math.radians(x_deg)
    pitch = math.radians(y_deg)
    yaw = math.radians(z_deg)

    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5
    cos_yaw = math.cos(half_yaw)
    sin_yaw = math.sin(half_yaw)
    cos_pitch = math.cos(half_pitch)
    sin_pitch = math.sin(half_pitch)
    cos_roll = math.cos(half_roll)
    sin_roll = math.sin(half_roll)

    return (
        (sin_roll * cos_pitch * cos_yaw) - (cos_roll * sin_pitch * sin_yaw),
        (cos_roll * sin_pitch * cos_yaw) + (sin_roll * cos_pitch * sin_yaw),
        (cos_roll * cos_pitch * sin_yaw) - (sin_roll * sin_pitch * cos_yaw),
        (cos_roll * cos_pitch * cos_yaw) + (sin_roll * sin_pitch * sin_yaw),
    )


def _to_ean_component_type(parameter: int) -> ComponentType | None:
    if parameter == PARAM_POSITION:
        return ComponentType.Position
    if parameter == PARAM_ROTATION:
        return ComponentType.Rotation
    if parameter == PARAM_SCALE:
        return ComponentType.Scale
    return None


def _build_ean_component(
    parameter: int,
    commands: list[EMACommand],
    default_values: RestTuple,
    end_frame: int,
) -> EANAnimationComponent | None:
    relevant_commands = [command for command in commands if command.parameter == parameter]
    if not relevant_commands:
        return None

    by_component = {command.component: command for command in relevant_commands}
    keyframes: list[EANKeyframe] = []
    for frame in range(max(0, end_frame) + 1):
        x = _evaluate_command(by_component.get(0), frame, default_values[0])
        y = _evaluate_command(by_component.get(1), frame, default_values[1])
        z = _evaluate_command(by_component.get(2), frame, default_values[2])
        w = _evaluate_command(by_component.get(3), frame, default_values[3])
        if parameter == PARAM_ROTATION:
            x, y, z, w = _euler_degrees_to_quaternion(x * w, y * w, z * w)
        keyframes.append(EANKeyframe(frame_index=frame, x=x, y=y, z=z, w=w))

    component_type = _to_ean_component_type(parameter)
    if component_type is None:
        return None
    return EANAnimationComponent(type=component_type, i_01=0, i_02=0, keyframes=keyframes)


def ema_obj_to_ean(ema_file: EMAFile) -> EANFile:
    if ema_file.ema_type != EMA_TYPE_OBJ:
        raise ValueError(
            "Only object EMA files are supported "
            f"(expected type {EMA_TYPE_OBJ}, got {ema_file.ema_type})."
        )
    if ema_file.skeleton is None:
        raise ValueError("EMA file has no skeleton; object animation import requires a skeleton.")

    skeleton = ema_file.skeleton
    animations: list[EANAnimation] = []

    for animation in sorted(ema_file.animations, key=lambda item: item.index):
        if animation.ema_type != EMA_ANIM_TYPE_OBJ:
            continue

        ean_nodes: list[EANNode] = []
        for node in animation.nodes:
            components: list[EANAnimationComponent] = []
            position_component = _build_ean_component(
                PARAM_POSITION,
                node.commands,
                EMA_DEFAULT_POSITION,
                animation.end_frame,
            )
            rotation_component = _build_ean_component(
                PARAM_ROTATION,
                node.commands,
                EMA_DEFAULT_ROTATION_EULER,
                animation.end_frame,
            )
            scale_component = _build_ean_component(
                PARAM_SCALE,
                node.commands,
                EMA_DEFAULT_SCALE,
                animation.end_frame,
            )
            if position_component is not None:
                components.append(position_component)
            if rotation_component is not None:
                components.append(rotation_component)
            if scale_component is not None:
                components.append(scale_component)
            if components:
                ean_nodes.append(EANNode(bone_name=node.bone_name, components=components))

        animations.append(
            EANAnimation(
                name=animation.name or str(animation.index),
                index=animation.index,
                float_precision=FloatPrecision._32BIT,
                nodes=ean_nodes,
            )
        )

    if not animations:
        raise ValueError("No object animations were found in the EMA file.")

    return EANFile(
        is_camera=False,
        i_08=ema_file.i_20,
        i_17=0,
        skeleton=skeleton,
        animations=animations,
    )


__all__ = [
    "EMA_ANIM_TYPE_OBJ",
    "EMA_TYPE_OBJ",
    "EMAAnimation",
    "EMACommand",
    "EMAFile",
    "EMAKeyframe",
    "EMANode",
    "ema_obj_to_ean",
    "parse_ema",
    "parse_ema_bytes",
]
