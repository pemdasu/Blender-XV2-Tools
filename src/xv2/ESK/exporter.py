import struct

import bpy

from ..EAN.exporter_char import _build_skeleton_from_armature
from .ESK import ESK_SIGNATURE


def export_esk(filepath: str, arm_obj: bpy.types.Object) -> tuple[bool, str | None]:
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return False, "Select an armature to export."

    try:
        _esk, skeleton_bytes, _rest_locals = _build_skeleton_from_armature(arm_obj)

        header_size = 32
        file_size = header_size + len(skeleton_bytes)

        out = bytearray()
        out.extend(struct.pack("<I", ESK_SIGNATURE))
        out.extend(struct.pack("<H", 0xFFFE))
        out.extend(struct.pack("<H", header_size))
        out.extend(struct.pack("<H", 1))
        out.extend(struct.pack("<H", 0))
        out.extend(struct.pack("<I", file_size))
        out.extend(struct.pack("<I", header_size))
        out.extend(struct.pack("<I", len(skeleton_bytes)))
        out.extend(struct.pack("<I", 0))
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
