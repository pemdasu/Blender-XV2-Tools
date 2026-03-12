from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import bpy

from ..NSK import parse_nsk
from ..NSK.exporter import export_nsk
from .EMO import build_emo_bytes_from_emd_esk


def export_emo(filepath: str, arm_obj: bpy.types.Object) -> tuple[bool, str | None]:
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return False, "Select an armature to export."

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".nsk", delete=False) as temp_file:
            temp_path = temp_file.name

        ok, error = export_nsk(temp_path, arm_obj)
        if not ok:
            return False, error or "Failed to export intermediate NSK data."

        nsk_file = parse_nsk(temp_path)
        emo_version = int(arm_obj.get("emo_version", 0x92C0))
        emo_i24_raw = arm_obj.get("emo_i24", 0)
        try:
            emo_i24 = int(emo_i24_raw)
        except (TypeError, ValueError):
            emo_i24 = 0

        emo_bytes = build_emo_bytes_from_emd_esk(
            nsk_file.emd_file,
            nsk_file.esk_file,
            version=emo_version,
            i_24=emo_i24,
        )
        Path(filepath).write_bytes(emo_bytes)
        return True, None
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        return False, f"Unexpected error while exporting EMO: {exc}"
    finally:
        if temp_path and os.path.isfile(temp_path):
            with contextlib.suppress(OSError):
                os.remove(temp_path)


__all__ = ["export_emo"]
