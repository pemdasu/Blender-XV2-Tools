from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from pathlib import Path


class Version(enum.IntEnum):
    XV1 = 1
    XV2 = 0


class PartType(enum.IntEnum):
    FaceBase = 0
    FaceForehead = 1
    FaceEye = 2
    FaceNose = 3
    FaceEar = 4
    Hair = 5
    Bust = 6
    Pants = 7
    Rist = 8
    Boots = 9


class PartTypeFlags(enum.IntFlag):
    None_ = 0x0
    FaceBase = 0x1
    FaceForehead = 0x2
    FaceEye = 0x4
    FaceNose = 0x8
    FaceEar = 0x10
    Hair = 0x20
    Bust = 0x40
    Pants = 0x80
    Rist = 0x100
    Boots = 0x200
    AllParts = (
        FaceBase | FaceForehead | FaceEye | FaceNose | FaceEar | Hair | Bust | Pants | Rist | Boots
    )


class Race(enum.IntEnum):
    Human = 0
    Saiyan = 1
    Namekian = 2
    FriezaRace = 3
    Majin = 4
    Other = 5


class Gender(enum.IntEnum):
    Male = 0
    Female = 1


class PartFlags(enum.IntFlag):
    Unk1 = 0x1
    DytFromTextureEmb = 0x2
    DytRampsFromTextureEmb = 0x4
    GreenScouterOverlay = 0x8
    RedScouterOverlay = 0x10
    BlueScouterOverlay = 0x20
    PurpleScouterOverlay = 0x40
    Unk8 = 0x80
    Unk9 = 0x100
    OrangeScouterOverlay = 0x200


@dataclass
class ColorSelector:
    part_color_group: int
    color_index: int


@dataclass
class CustomColor:
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    a: float = 0.0

    def is_black(self) -> bool:
        return self.r == 0.0 and self.g == 0.0 and self.b == 0.0 and self.a == 0.0


@dataclass
class Colors:
    id: int
    color1: CustomColor
    color2: CustomColor
    color3: CustomColor
    color4: CustomColor

    def is_null(self) -> bool:
        return all(c.is_black() for c in (self.color1, self.color2, self.color3, self.color4))


@dataclass
class PhysicsPart:
    model1: int
    model2: int
    texture: int
    flags: PartFlags
    hide_flags: PartTypeFlags
    hide_mat_flags: PartTypeFlags
    chara_code: str
    emd_path: str
    emm_path: str
    emb_path: str
    ean_path: str
    bone_to_attach: str
    scd_path: str


@dataclass
class Unk3:
    values: tuple[int, ...]


@dataclass
class Part:
    part_type: PartType | None
    model: int
    model2: int
    texture: int
    shader: int
    flags: PartFlags
    hide_flags: PartTypeFlags
    hide_mat_flags: PartTypeFlags
    f_36: float
    f_40: float
    i_44: int
    i_48: int
    chara_code: str
    emd_path: str
    emm_path: str
    emb_path: str
    ean_path: str
    color_selectors: list[ColorSelector] = field(default_factory=list)
    physics_parts: list[PhysicsPart] = field(default_factory=list)
    unk3: list[Unk3] | None = None


@dataclass
class PartColor:
    id: int
    name: str
    colors: list[Colors] = field(default_factory=list)


@dataclass
class BoneScale:
    bone_name: str
    scale_x: float
    scale_y: float
    scale_z: float


@dataclass
class Body:
    id: int
    body_scales: list[BoneScale] = field(default_factory=list)


@dataclass
class Bone:
    bone_name: str
    i_00: int
    i_04: int
    f_12: float
    f_16: float
    f_20: float
    f_24: float
    f_28: float
    f_32: float
    f_36: float
    f_40: float
    f_44: float


@dataclass
class SkeletonData:
    i_00: int
    bones: list[Bone] = field(default_factory=list)


@dataclass
class PartSet:
    id: int
    parts: dict[PartType, Part] = field(default_factory=dict)

    def get(self, part_type: PartType) -> Part | None:
        return self.parts.get(part_type)


@dataclass
class BCSFile:
    version: Version = Version.XV2
    race: Race | int = Race.Human
    gender: Gender | int = Gender.Male
    f_48: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 7)
    part_sets: list[PartSet] = field(default_factory=list)
    part_colors: list[PartColor] = field(default_factory=list)
    bodies: list[Body] = field(default_factory=list)
    skeleton1: SkeletonData | None = None
    skeleton2: SkeletonData | None = None


def read_bcs(path: str | Path) -> BCSFile:
    with open(path, "rb") as f:
        data = f.read()
    return read_bcs_bytes(data)


def read_bcs_bytes(data: bytes) -> BCSFile:
    return _BCSParser(data).parse()


def _enum_value(enum_cls: type[enum.IntEnum], value: int) -> enum.IntEnum | int:
    try:
        return enum_cls(value)
    except ValueError:
        return value


class _BCSParser:
    _PART_ORDER = [
        PartType.FaceBase,
        PartType.FaceForehead,
        PartType.FaceEye,
        PartType.FaceNose,
        PartType.FaceEar,
        PartType.Hair,
        PartType.Bust,
        PartType.Pants,
        PartType.Rist,
        PartType.Boots,
    ]

    def __init__(self, data: bytes):
        self.data = data

    def parse(self) -> BCSFile:
        version = self._read_version()

        partset_count = self._u16(12)
        partcolors_count = self._u16(14)
        body_count = self._u16(16)

        if version == Version.XV1:
            partset_offset = self._i32(20)
            partcolors_offset = self._i32(24)
            body_offset = self._i32(28)
            skeleton2_offset = 0
            skeleton1_offset = 64
            f_48 = self._float_array(36, 7)
            race_val = self.data[32] if len(self.data) > 32 else 0
            gender_val = self.data[33] if len(self.data) > 33 else 0
        else:
            partset_offset = self._i32(24)
            partcolors_offset = self._i32(28)
            body_offset = self._i32(32)
            skeleton2_offset = self._i32(36)
            skeleton1_offset = self._i32(40)
            f_48 = self._float_array(48, 7)
            race_val = self.data[44] if len(self.data) > 44 else 0
            gender_val = self.data[45] if len(self.data) > 45 else 0

        bcs = BCSFile(
            version=version,
            race=_enum_value(Race, race_val),
            gender=_enum_value(Gender, gender_val),
            f_48=tuple(f_48),
        )

        self._parse_part_sets(bcs, partset_offset, partset_count)
        self._parse_part_colors(bcs, partcolors_offset, partcolors_count)
        self._parse_bodies(bcs, body_offset, body_count)

        if skeleton1_offset != 0:
            skeleton_ptr = (
                skeleton1_offset if version == Version.XV1 else self._i32(skeleton1_offset)
            )
            if skeleton_ptr:
                bcs.skeleton1 = self._parse_skeleton(skeleton_ptr, version)

        if skeleton2_offset != 0:
            skeleton_ptr = self._i32(skeleton2_offset)
            if skeleton_ptr:
                bcs.skeleton2 = self._parse_skeleton(skeleton_ptr, version)

        return bcs

    def _parse_part_sets(self, bcs: BCSFile, table_offset: int, count: int) -> None:
        offset = table_offset
        for idx in range(count):
            partset_ptr = self._i32(offset)
            if partset_ptr != 0:
                part_set = PartSet(id=idx)

                part_count = self._i32(partset_ptr + 20)
                if part_count != 10:
                    raise ValueError(
                        f"Part count mismatch on PartSet {idx}: expected 10, found {part_count}"
                    )

                table = partset_ptr + self._i32(partset_ptr + 24)
                for i, part_type in enumerate(self._PART_ORDER):
                    part_offset = self._i32(table + i * 4)
                    part = self._parse_part(part_offset, partset_ptr, part_type, bcs.version)
                    if part:
                        part_set.parts[part_type] = part

                bcs.part_sets.append(part_set)
            offset += 4

    def _parse_part(
        self,
        relative_offset: int,
        part_base: int,
        part_type: PartType,
        version: Version,
    ) -> Part | None:
        if relative_offset == 0:
            return None

        offset = part_base + relative_offset
        hide_flags = self._i32(offset + 28)
        hide_mat_flags = self._i32(offset + 32)

        if hide_flags > 0x3FF or hide_mat_flags > 0x3FF:
            raise ValueError(
                f"Unexpected hide flags on part {part_type}: {hide_flags}, {hide_mat_flags}"
            )

        color_count = self._u16(offset + 18)
        color_offset = self._i32(offset + 20) + offset if color_count > 0 else 0
        physics_count = self._u16(offset + 74)
        physics_offset = self._i32(offset + 76) + offset if physics_count > 0 else 0

        part = Part(
            part_type=part_type,
            model=self._i16(offset + 0),
            model2=self._i16(offset + 2),
            texture=self._i16(offset + 4),
            shader=self._i16(offset + 16),
            flags=PartFlags(self._u32(offset + 24)),
            hide_flags=PartTypeFlags(hide_flags),
            hide_mat_flags=PartTypeFlags(hide_mat_flags),
            f_36=self._f32(offset + 36),
            f_40=self._f32(offset + 40),
            i_44=self._i32(offset + 44),
            i_48=self._i32(offset + 48),
            chara_code=self._read_fixed_string(offset + 52, 4, encoding="ascii"),
            emd_path=self._string_rel(self._i32(offset + 56), offset),
            emm_path=self._string_rel(self._i32(offset + 60), offset),
            emb_path=self._string_rel(self._i32(offset + 64), offset),
            ean_path=self._string_rel(self._i32(offset + 68), offset),
        )

        part.color_selectors = self._parse_color_selectors(color_offset, color_count)
        part.physics_parts = self._parse_physics_parts(physics_offset, physics_count)

        if version == Version.XV2:
            unk3_count = self._u16(offset + 82)
            unk3_offset = self._i32(offset + 84) + offset if unk3_count > 0 else 0
            part.unk3 = self._parse_unk3(unk3_offset, unk3_count)

        return part

    def _parse_color_selectors(self, offset: int, count: int) -> list[ColorSelector]:
        if count <= 0 or offset == 0:
            return []

        selectors: list[ColorSelector] = []
        for _ in range(count):
            selectors.append(
                ColorSelector(
                    part_color_group=self._u16(offset + 0), color_index=self._u16(offset + 2)
                )
            )
            offset += 4
        return selectors

    def _parse_physics_parts(self, offset: int, count: int) -> list[PhysicsPart]:
        if count <= 0 or offset == 0:
            return []

        parts: list[PhysicsPart] = []
        for _ in range(count):
            hide_flags = self._i32(offset + 28)
            hide_mat_flags = self._i32(offset + 32)

            if hide_flags > 0x200 or hide_mat_flags > 0x200:
                raise ValueError(f"Unexpected physics hide flags: {hide_flags}, {hide_mat_flags}")

            parts.append(
                PhysicsPart(
                    model1=self._i16(offset + 0),
                    model2=self._i16(offset + 2),
                    texture=self._i16(offset + 4),
                    flags=PartFlags(self._u32(offset + 24)),
                    hide_flags=PartTypeFlags(hide_flags),
                    hide_mat_flags=PartTypeFlags(hide_mat_flags),
                    chara_code=self._read_fixed_string(offset + 36, 4, encoding="ascii"),
                    emd_path=self._string_rel(self._i32(offset + 40), offset),
                    emm_path=self._string_rel(self._i32(offset + 44), offset),
                    emb_path=self._string_rel(self._i32(offset + 48), offset),
                    ean_path=self._string_rel(self._i32(offset + 52), offset),
                    bone_to_attach=self._string_rel(self._i32(offset + 56), offset),
                    scd_path=self._string_rel(self._i32(offset + 60), offset),
                )
            )
            offset += 72

        return parts

    def _parse_unk3(self, offset: int, count: int) -> list[Unk3]:
        if count <= 0 or offset == 0:
            return []

        entries: list[Unk3] = []
        for _ in range(count):
            entries.append(Unk3(values=tuple(self._int16_array(offset, 6))))
            offset += 12
        return entries

    def _parse_part_colors(self, bcs: BCSFile, table_offset: int, count: int) -> None:
        offset = table_offset
        for idx in range(count):
            color_ptr = self._i32(offset)
            if color_ptr != 0:
                name_offset = self._i32(color_ptr + 0) + color_ptr
                name = self._read_cstring(name_offset)

                color_count = self._u16(color_ptr + 10)
                colors_offset = self._i32(color_ptr + 12) + color_ptr if color_count > 0 else 0
                colors = self._parse_colors(colors_offset, color_count)

                bcs.part_colors.append(PartColor(id=idx, name=name, colors=colors))
            offset += 4

    def _parse_colors(self, offset: int, count: int) -> list[Colors]:
        entries: list[Colors] = []
        if count <= 0 or offset == 0:
            return entries

        for idx in range(count):
            color1 = CustomColor(*self._float_array(offset + 0, 4))
            color2 = CustomColor(*self._float_array(offset + 16, 4))
            color3 = CustomColor(*self._float_array(offset + 32, 4))
            color4 = CustomColor(*self._float_array(offset + 48, 4))
            entry = Colors(id=idx, color1=color1, color2=color2, color3=color3, color4=color4)
            if not entry.is_null():
                entries.append(entry)
            offset += 80

        return entries

    def _parse_bodies(self, bcs: BCSFile, table_offset: int, count: int) -> None:
        offset = table_offset
        for idx in range(count):
            body_ptr = self._i32(offset)
            if body_ptr != 0:
                body = self._parse_body(body_ptr, idx)
                if body.body_scales:
                    bcs.bodies.append(body)
            offset += 4

    def _parse_body(self, offset: int, index: int) -> Body:
        body_scale_count = self._u16(offset + 2)
        body_scale_offset = self._i32(offset + 4) + offset

        body_scales: list[BoneScale] = []
        for _ in range(body_scale_count):
            bone_name = self._read_cstring(self._i32(body_scale_offset + 12) + body_scale_offset)
            body_scales.append(
                BoneScale(
                    bone_name=bone_name,
                    scale_x=self._f32(body_scale_offset + 0),
                    scale_y=self._f32(body_scale_offset + 4),
                    scale_z=self._f32(body_scale_offset + 8),
                )
            )
            body_scale_offset += 16

        return Body(id=index, body_scales=body_scales)

    def _parse_skeleton(self, offset: int, version: Version) -> SkeletonData:
        relative_to = 32 if version == Version.XV1 else offset

        i_00 = self._i16(offset)
        bone_count = self._u16(offset + 2)
        bone_offset = self._i32(offset + 4) + relative_to

        bones: list[Bone] = []
        for _ in range(bone_count):
            if version == Version.XV1:
                bone_name = self._read_cstring(self._i32(bone_offset + 12) + bone_offset)
                bones.append(
                    Bone(
                        bone_name=bone_name,
                        i_00=self._i32(bone_offset + 0),
                        i_04=self._i32(bone_offset + 4),
                        f_12=self._f32(bone_offset + 16),
                        f_16=self._f32(bone_offset + 20),
                        f_20=self._f32(bone_offset + 24),
                        f_24=self._f32(bone_offset + 28),
                        f_28=self._f32(bone_offset + 32),
                        f_32=self._f32(bone_offset + 36),
                        f_36=self._f32(bone_offset + 40),
                        f_40=self._f32(bone_offset + 44),
                        f_44=self._f32(bone_offset + 48),
                    )
                )
            else:
                bone_name = self._read_cstring(self._i32(bone_offset + 48) + bone_offset)
                bones.append(
                    Bone(
                        bone_name=bone_name,
                        i_00=self._i32(bone_offset + 0),
                        i_04=self._i32(bone_offset + 4),
                        f_12=self._f32(bone_offset + 12),
                        f_16=self._f32(bone_offset + 16),
                        f_20=self._f32(bone_offset + 20),
                        f_24=self._f32(bone_offset + 24),
                        f_28=self._f32(bone_offset + 28),
                        f_32=self._f32(bone_offset + 32),
                        f_36=self._f32(bone_offset + 36),
                        f_40=self._f32(bone_offset + 40),
                        f_44=self._f32(bone_offset + 44),
                    )
                )
            bone_offset += 52

        return SkeletonData(i_00=i_00, bones=bones)

    def _read_version(self) -> Version:
        signature = self._i16(6)
        if signature == 72:
            return Version.XV1
        if signature in (0, 76):
            return Version.XV2
        raise ValueError(f"Unknown BCS version flag: {signature}")

    def _u16(self, offset: int) -> int:
        return struct.unpack_from("<H", self.data, offset)[0]

    def _i16(self, offset: int) -> int:
        return struct.unpack_from("<h", self.data, offset)[0]

    def _u32(self, offset: int) -> int:
        return struct.unpack_from("<I", self.data, offset)[0]

    def _i32(self, offset: int) -> int:
        return struct.unpack_from("<i", self.data, offset)[0]

    def _f32(self, offset: int) -> float:
        return struct.unpack_from("<f", self.data, offset)[0]

    def _float_array(self, offset: int, count: int) -> list[float]:
        return list(struct.unpack_from(f"<{count}f", self.data, offset))

    def _int16_array(self, offset: int, count: int) -> list[int]:
        return list(struct.unpack_from(f"<{count}h", self.data, offset))

    def _read_cstring(self, offset: int, encoding: str = "utf-8") -> str:
        if offset <= 0 or offset >= len(self.data):
            return ""
        end = offset
        while end < len(self.data) and self.data[end] != 0:
            end += 1
        return self.data[offset:end].decode(encoding, errors="ignore")

    def _read_fixed_string(self, offset: int, length: int, encoding: str = "utf-8") -> str:
        if offset < 0 or offset >= len(self.data):
            return ""
        raw = self.data[offset : offset + length]
        nul = raw.find(b"\x00")
        if nul != -1:
            raw = raw[:nul]
        return raw.decode(encoding, errors="ignore")

    def _string_rel(self, relative_offset: int, base: int) -> str:
        if relative_offset == 0:
            return ""
        return self._read_cstring(base + relative_offset)


__all__ = [
    "BCSFile",
    "Body",
    "Bone",
    "BoneScale",
    "ColorSelector",
    "Colors",
    "CustomColor",
    "Gender",
    "Part",
    "PartColor",
    "PartFlags",
    "PartSet",
    "PartType",
    "PartTypeFlags",
    "PhysicsPart",
    "Race",
    "SkeletonData",
    "Unk3",
    "Version",
    "read_bcs",
    "read_bcs_bytes",
]
