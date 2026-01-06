import math

import bpy

from .ESK import build_armature, parse_esk


def import_esk(path: str) -> bpy.types.Object | None:
    try:
        esk = parse_esk(path)
    except Exception as exc:
        print(f"Failed to read ESK: {exc}")
        return None

    arm_name = esk.bones[0].name if esk.bones else "Armature"
    arm_obj = build_armature(esk, arm_name)
    arm_obj.name = arm_name
    arm_obj.rotation_euler[0] = math.radians(90.0)
    if arm_obj.data:
        arm_obj.data.display_type = "STICK"
    return arm_obj


__all__ = ["import_esk"]
