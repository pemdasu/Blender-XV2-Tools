from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from pathlib import Path

from ...utils import half_to_float, read_cstring
from ..ESK.ESK import ESK_Bone, ESK_File


class IntPrecision(enum.IntEnum):
    _8BIT = 0
    _16BIT = 1


class FloatPrecision(enum.IntEnum):
    _16BIT = 1
    _32BIT = 2


class ComponentType(enum.IntEnum):
    Position = 0
    Rotation = 1
    Scale = 2
    Unknown = 255


@dataclass
class EANKeyframe:
    frame_index: int
    x: float
    y: float
    z: float
    w: float


@dataclass
class EANAnimationComponent:
    type: ComponentType
    i_01: int
    i_02: int
    keyframes: list[EANKeyframe] = field(default_factory=list)


@dataclass
class EANNode:
    bone_name: str
    components: list[EANAnimationComponent] = field(default_factory=list)


@dataclass
class EANAnimation:
    name: str
    index: int
    float_precision: FloatPrecision
    nodes: list[EANNode] = field(default_factory=list)


@dataclass
class EANFile:
    is_camera: bool
    i_08: int
    i_17: int
    skeleton: ESK_File
    animations: list[EANAnimation] = field(default_factory=list)


def read_ean(path: str | Path, link_skeleton: bool = True) -> EANFile:
    with open(path, "rb") as f:
        data = f.read()
    return read_ean_bytes(data, link_skeleton=link_skeleton)


def read_ean_bytes(data: bytes, link_skeleton: bool = True) -> EANFile:
    parser = _EANParser(data, link_skeleton=link_skeleton)
    return parser.parse()


class _EANParser:
    def __init__(self, data: bytes, link_skeleton: bool):
        self.data = data
        self.link_skeleton = link_skeleton

    def parse(self) -> EANFile:
        animation_count = self._u16(18)
        skeleton_offset = self._i32(20)
        animation_table_offset = self._i32(24)
        animation_names_offset = self._i32(28)

        i_08 = self._i32(8)
        is_camera = self.data[16] != 0
        i_17 = self.data[17]

        skeleton = self._parse_skeleton(skeleton_offset)

        animations: list[EANAnimation] = []
        for i in range(animation_count):
            anim_ptr = self._i32(animation_table_offset + 4 * i)
            name_ptr = self._i32(animation_names_offset + 4 * i)
            if anim_ptr == 0:
                continue
            animation = self._parse_animation(anim_ptr, name_ptr, i, skeleton)
            animations.append(animation)

        return EANFile(
            is_camera=is_camera,
            i_08=i_08,
            i_17=i_17,
            skeleton=skeleton,
            animations=animations,
        )

    def _parse_skeleton(self, offset: int) -> ESK_File:
        skeleton = ESK_File()
        if offset <= 0:
            return skeleton

        bone_count = self._i16(offset + 0)
        bone_index_table_offset = self._i32(offset + 4) + offset
        name_table_offset = self._i32(offset + 8) + offset
        skinning_table_offset = self._i32(offset + 12) + offset

        for bone_index in range(bone_count):
            bone_index_offset = bone_index_table_offset + 8 * bone_index
            parent_idx = self._i16(bone_index_offset + 0)
            child_idx = self._i16(bone_index_offset + 2)
            sibling_idx = self._i16(bone_index_offset + 4)

            name_rel = self._i32(name_table_offset + 4 * bone_index)
            name_off = offset + name_rel
            name = read_cstring(self.data, name_off)

            t_off = skinning_table_offset + 48 * bone_index
            px, py, pz, pw, rx, ry, rz, rw, sx, sy, sz, sw = struct.unpack_from(
                "<12f", self.data, t_off
            )

            bone = ESK_Bone(
                name=name,
                index=bone_index,
                matrix=None,  # Not building matrices here
                parent_index=parent_idx,
                child_index=child_idx,
                sibling_index=sibling_idx,
            )
            # Store raw transform values for reference
            bone.position = (px * pw, py * pw, pz * pw)
            bone.rotation = (rw, rx, ry, rz)
            bone.scale = (sx * sw, sy * sw, sz * sw)
            skeleton.bones.append(bone)

        return skeleton

    def _parse_animation(
        self, offset: int, name_offset: int, anim_index: int, skeleton: ESK_File
    ) -> EANAnimation:
        index_size = self.data[offset + 2]
        float_size = self.data[offset + 3]
        node_count = self._i32(offset + 8)
        node_table_offset = self._i32(offset + 12) + offset

        float_precision = FloatPrecision(float_size)
        nodes: list[EANNode] = []

        for _ in range(node_count):
            node_ptr = self._i32(node_table_offset) + offset
            bone_index = self._i16(node_ptr)
            bone_name = (
                skeleton.bones[bone_index].name
                if 0 <= bone_index < len(skeleton.bones)
                else f"bone_{bone_index}"
            )
            keyframed_count = self._i16(node_ptr + 2)
            keyframed_offset = self._i32(node_ptr + 4) + node_ptr

            components: list[EANAnimationComponent] = []
            for _ in range(keyframed_count):
                comp_ptr = self._i32(keyframed_offset) + node_ptr
                comp_type = (
                    ComponentType(self.data[comp_ptr + 0])
                    if self.data[comp_ptr + 0] in ComponentType._value2member_map_
                    else ComponentType.Unknown
                )
                i_01 = self.data[comp_ptr + 1]
                i_02 = self._i16(comp_ptr + 2)

                keyframe_count = self._i32(comp_ptr + 4)
                index_list_offset = self._i32(comp_ptr + 8) + comp_ptr
                matrix_offset = self._i32(comp_ptr + 12) + comp_ptr

                keyframes = self._parse_keyframes(
                    index_list_offset,
                    matrix_offset,
                    keyframe_count,
                    IntPrecision(index_size),
                    float_precision,
                )

                components.append(
                    EANAnimationComponent(
                        type=comp_type,
                        i_01=i_01,
                        i_02=i_02,
                        keyframes=keyframes,
                    )
                )

                keyframed_offset += 4

            nodes.append(EANNode(bone_name=bone_name, components=components))
            node_table_offset += 4

        name = read_cstring(self.data, name_offset) or str(anim_index)

        return EANAnimation(
            name=name,
            index=anim_index,
            float_precision=float_precision,
            nodes=nodes,
        )

    def _parse_keyframes(
        self,
        index_offset: int,
        float_offset: int,
        count: int,
        index_size: IntPrecision,
        float_size: FloatPrecision,
    ) -> list[EANKeyframe]:
        keyframes: list[EANKeyframe] = []
        idx_off = index_offset
        flt_off = float_offset

        for _ in range(count):
            if index_size == IntPrecision._8BIT:
                frame = self.data[idx_off]
                idx_off += 1
            else:
                frame = self._u16(idx_off)
                idx_off += 2

            if float_size == FloatPrecision._16BIT:
                x = half_to_float(self._u16(flt_off + 0))
                y = half_to_float(self._u16(flt_off + 2))
                z = half_to_float(self._u16(flt_off + 4))
                w = half_to_float(self._u16(flt_off + 6))
                flt_off += 8
            else:
                x = self._f32(flt_off + 0)
                y = self._f32(flt_off + 4)
                z = self._f32(flt_off + 8)
                w = self._f32(flt_off + 12)
                flt_off += 16

            keyframes.append(EANKeyframe(frame_index=frame, x=x, y=y, z=z, w=w))

        return keyframes

    def _u16(self, offset: int) -> int:
        return struct.unpack_from("<H", self.data, offset)[0]

    def _i16(self, offset: int) -> int:
        return struct.unpack_from("<h", self.data, offset)[0]

    def _i32(self, offset: int) -> int:
        return struct.unpack_from("<i", self.data, offset)[0]

    def _f32(self, offset: int) -> float:
        return struct.unpack_from("<f", self.data, offset)[0]


__all__ = [
    "ComponentType",
    "EANAnimation",
    "EANAnimationComponent",
    "EANFile",
    "EANKeyframe",
    "EANNode",
    "FloatPrecision",
    "IntPrecision",
    "read_ean",
    "read_ean_bytes",
]
