import struct
from pathlib import Path

import bpy
import mathutils

from ..EAN.exporter_char import _build_skeleton_from_armature
from .ESK import ESK_SIGNATURE, ESK_Bone


def _align16_size(size: int) -> int:
    return (size + 15) & ~15


def _read_arm_int_prop(arm_obj: bpy.types.Object, key: str, default: int) -> int:
    try:
        return int(arm_obj.get(key, default))
    except Exception:
        return int(default)


def _read_arm_u64_prop(arm_obj: bpy.types.Object, key: str, default: int) -> int:
    raw = arm_obj.get(key, default)
    try:
        return int(raw)
    except Exception:
        try:
            return int(str(raw).strip())
        except Exception:
            return int(default)


def _pack_relative_transforms(bones: list[ESK_Bone]) -> bytes:
    out = bytearray()
    for bone in bones:
        loc, rot, scale = bone.matrix.decompose()
        out.extend(struct.pack("<4f", loc.x, loc.y, loc.z, 1.0))
        out.extend(struct.pack("<4f", rot.x, rot.y, rot.z, rot.w))
        out.extend(struct.pack("<4f", scale.x, scale.y, scale.z, 1.0))
    return bytes(out)


def _pack_absolute_transforms(bones: list[ESK_Bone]) -> bytes:
    out = bytearray()
    world_mats: dict[int, mathutils.Matrix] = {}

    def compute_world(bone_data: ESK_Bone) -> mathutils.Matrix:
        if bone_data.index in world_mats:
            return world_mats[bone_data.index]
        matrix = bone_data.matrix.copy()
        if 0 <= bone_data.parent_index < len(bones) and bones[bone_data.parent_index] is not bone_data:
            matrix = compute_world(bones[bone_data.parent_index]) @ matrix
        world_mats[bone_data.index] = matrix
        return matrix

    for bone in bones:
        world = compute_world(bone)
        abs_mat = world.inverted_safe().transposed()
        for row in abs_mat:
            out.extend(struct.pack("<4f", *row))
    return bytes(out)


def _build_esk_skeleton_bytes(
    bones: list[ESK_Bone],
    skeleton_flag: int = 0,
    skeleton_id: int = 0,
) -> bytes:
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

    data = bytearray()
    data.extend(struct.pack("<h", bone_count))
    data.extend(struct.pack("<h", int(skeleton_flag)))
    data.extend(struct.pack("<I", index_rel))
    data.extend(struct.pack("<I", name_rel))
    data.extend(struct.pack("<I", 0))  # relative transforms offset
    data.extend(struct.pack("<I", 0))  # absolute transforms offset
    data.extend(struct.pack("<I", 0))  # IK offset
    data.extend(struct.pack("<I", 0))  # extra values offset
    data.extend(struct.pack("<Q", int(skeleton_id) & 0xFFFFFFFFFFFFFFFF))

    for bone in bones:
        data.extend(
            struct.pack("<hhhH", bone.parent_index, bone.child_index, bone.sibling_index, 0)
        )

    for offset in name_offsets:
        data.extend(struct.pack("<I", string_off + offset))
    data.extend(strings)

    rel_off = _align16_size(len(data))
    pad = rel_off - len(data)
    if pad > 0:
        data.extend(b"\x00" * pad)
    struct.pack_into("<I", data, 12, rel_off)

    data.extend(_pack_relative_transforms(bones))

    abs_off = _align16_size(len(data))
    if abs_off > len(data):
        data.extend(b"\x00" * (abs_off - len(data)))
    struct.pack_into("<I", data, 16, abs_off)
    data.extend(_pack_absolute_transforms(bones))

    return bytes(data)


def _read_skeleton_layout(data: bytes) -> dict[str, object] | None:
    if len(data) < 32 or struct.unpack_from("<I", data, 0)[0] != ESK_SIGNATURE:
        return None
    skel_off = struct.unpack_from("<I", data, 16)[0]
    if skel_off <= 0 or skel_off + 36 > len(data):
        return None
    bone_count = struct.unpack_from("<h", data, skel_off)[0]
    idx_rel, name_rel, rel_rel, abs_rel = struct.unpack_from("<IIII", data, skel_off + 4)
    idx_off = skel_off + idx_rel
    name_off = skel_off + name_rel
    rel_off = skel_off + rel_rel
    abs_off = (skel_off + abs_rel) if abs_rel else 0
    if bone_count < 0 or idx_off < 0 or name_off < 0 or rel_off < 0:
        return None
    if idx_off + bone_count * 8 > len(data) or name_off + bone_count * 4 > len(data):
        return None

    names: list[str] = []
    for i in range(bone_count):
        name_rel_i = struct.unpack_from("<I", data, name_off + i * 4)[0]
        name_abs = skel_off + name_rel_i
        if not (0 <= name_abs < len(data)):
            return None
        end = data.find(b"\0", name_abs)
        if end == -1:
            return None
        names.append(data[name_abs:end].decode("ascii", "ignore"))

    return {
        "bone_count": bone_count,
        "rel_off": rel_off,
        "abs_off": abs_off,
        "names": names,
    }


def _export_using_source_template(
    filepath: str,
    source_path: str,
    bones: list[ESK_Bone],
) -> bool:
    try:
        src = Path(source_path)
        if not src.is_file():
            return False
        data = src.read_bytes()
        layout = _read_skeleton_layout(data)
        if layout is None:
            return False

        bone_count = int(layout["bone_count"])
        if bone_count != len(bones):
            return False

        source_names = list(layout["names"])
        if source_names != [bone.name for bone in bones]:
            return False

        rel_blob = _pack_relative_transforms(bones)
        abs_blob = _pack_absolute_transforms(bones)
        rel_off = int(layout["rel_off"])
        abs_off = int(layout["abs_off"])
        if rel_off + len(rel_blob) > len(data):
            return False
        if abs_off and abs_off + len(abs_blob) > len(data):
            return False

        out = bytearray(data)
        out[rel_off : rel_off + len(rel_blob)] = rel_blob
        if abs_off:
            out[abs_off : abs_off + len(abs_blob)] = abs_blob

        with open(filepath, "wb") as f:
            f.write(out)
        return True
    except Exception:
        return False


def export_esk(filepath: str, arm_obj: bpy.types.Object) -> tuple[bool, str | None]:
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return False, "Select an armature to export."

    try:
        esk, _skeleton_bytes, _rest_locals = _build_skeleton_from_armature(arm_obj)
        source_path = str(arm_obj.get("esk_source_path", "")).strip()
        if source_path and _export_using_source_template(filepath, source_path, esk.bones):
            return True, None

        version = _read_arm_int_prop(arm_obj, "esk_version", 37568) & 0xFFFF
        i10 = _read_arm_int_prop(arm_obj, "esk_i10", 0) & 0xFFFF
        i12 = _read_arm_int_prop(arm_obj, "esk_i12", 0) & 0xFFFFFFFF
        i24 = _read_arm_int_prop(arm_obj, "esk_i24", 0) & 0xFFFFFFFF
        default_skel_flag = 0 if arm_obj.get("ean_source") else 1
        skeleton_flag = _read_arm_int_prop(arm_obj, "esk_skeleton_flag", default_skel_flag)
        skeleton_id = _read_arm_u64_prop(arm_obj, "esk_skeleton_id", 0)

        skeleton_bytes = _build_esk_skeleton_bytes(
            esk.bones,
            skeleton_flag=skeleton_flag,
            skeleton_id=skeleton_id,
        )

        out = bytearray()
        out.extend(struct.pack("<I", ESK_SIGNATURE))
        out.extend(struct.pack("<H", 0xFFFE))
        out.extend(struct.pack("<H", 0x001C))
        out.extend(struct.pack("<H", version))
        out.extend(struct.pack("<H", i10))
        out.extend(struct.pack("<I", i12))
        out.extend(struct.pack("<I", 32))  # Offset to skeleton
        out.extend(struct.pack("<I", 0))  # NSK offset (unused)
        out.extend(struct.pack("<I", i24))
        out.extend(struct.pack("<I", 0))
        out.extend(skeleton_bytes)

        with open(filepath, "wb") as f:
            f.write(out)
        return True, None
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return False, f"Unexpected error while exporting: {exc}"


__all__ = ["export_esk"]
