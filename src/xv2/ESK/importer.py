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
    arm_obj["esk_source_path"] = path
    arm_obj["esk_version"] = int(esk.version)
    arm_obj["esk_i10"] = int(esk.i_10)
    arm_obj["esk_i12"] = int(esk.i_12)
    arm_obj["esk_i24"] = int(esk.i_24)
    arm_obj["esk_skeleton_flag"] = int(esk.skeleton_flag)
    arm_obj["esk_skeleton_id"] = str(int(esk.skeleton_id))
    arm_obj.rotation_euler[0] = math.radians(90.0)
    if arm_obj.data:
        arm_obj.data.display_type = "STICK"
    return arm_obj


__all__ = ["import_esk"]
