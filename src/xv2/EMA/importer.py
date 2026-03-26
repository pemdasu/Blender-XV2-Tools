from __future__ import annotations

import bpy

from ..EAN.importer import import_ean_data
from .EMA import ema_obj_to_ean, parse_ema


def import_ema_animations(
    path: str,
    target_armature: bpy.types.Object | None = None,
    replace_armature: bool = False,
    preserve_bone_axes: bool = True,
) -> bpy.types.Object | None:
    ema_file = parse_ema(path)
    ean_file = ema_obj_to_ean(ema_file)
    armature = import_ean_data(
        ean_file,
        source_path=path,
        target_armature=target_armature,
        replace_armature=replace_armature,
        preserve_bone_axes=preserve_bone_axes,
    )
    if armature is not None:
        armature["ema_source_path"] = path
        armature["ema_version"] = int(ema_file.version)
        armature["ema_i20"] = int(ema_file.i_20)
        armature["ema_i24"] = int(ema_file.i_24)
        armature["ema_i28"] = int(ema_file.i_28)
    return armature


__all__ = ["import_ema_animations"]
