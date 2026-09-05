from __future__ import annotations

from pathlib import Path

import bpy

from ..EAN.exporter_char import _build_skeleton_from_armature
from ..ESK.exporter import get_or_create_skeleton_id
from ..NSK.exporter import _build_emd_from_armature_hierarchy
from .EMO import build_emo_bytes_from_emd_esk


def export_emo(filepath: str, arm_obj: bpy.types.Object) -> tuple[bool, str | None]:
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return False, "Select an armature to export."

    try:
        emd_file = _build_emd_from_armature_hierarchy(arm_obj)
        if not emd_file.models:
            return False, "No mesh models found under this armature."
        esk_file, _skeleton_bytes, _rest_locals = _build_skeleton_from_armature(arm_obj)
        esk_file.skeleton_id = get_or_create_skeleton_id(arm_obj)
        emo_version = int(arm_obj.get("emo_version", 0x92C0))
        emo_i24_raw = arm_obj.get("emo_i24", 0)
        try:
            emo_i24 = int(emo_i24_raw)
        except (TypeError, ValueError):
            emo_i24 = 0

        emo_bytes = build_emo_bytes_from_emd_esk(
            emd_file,
            esk_file,
            version=emo_version,
            i_24=emo_i24,
        )
        Path(filepath).write_bytes(emo_bytes)
        return True, None
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        return False, f"Unexpected error while exporting EMO: {exc}"


__all__ = ["export_emo"]
