import bpy
from bpy.props import FloatVectorProperty

from ..xv2.bone_scale import apply_bone_scale_preview
from ..xv2.consts import BONE_SCALE_IDENTITY, BONE_SCALE_PROP

_bone_scale_refresh_is_scheduled = False


class XV2_OT_bone_scale_reset_selected(bpy.types.Operator):
    bl_idname = "xv2_bone_scale.reset_selected"
    bl_label = "Reset Bone Scale"
    bl_description = "Reset the active bone XV2 bone scale to 1, 1, 1"

    @classmethod
    def poll(cls, context):
        return context.active_bone is not None

    def execute(self, context):
        setattr(context.active_bone, BONE_SCALE_PROP, BONE_SCALE_IDENTITY)
        return {"FINISHED"}


class BONE_PT_xv2_bone_scale(bpy.types.Panel):
    bl_label = "XV2 Bone Scale"
    bl_idname = "BONE_PT_xv2_bone_scale"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "bone"

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == "ARMATURE"
            and context.active_bone is not None
        )

    def draw(self, context):
        layout = self.layout
        bone = context.active_bone

        layout.prop(bone, BONE_SCALE_PROP, text="Bone Scale")
        layout.operator(XV2_OT_bone_scale_reset_selected.bl_idname, icon="LOOP_BACK")


def _find_armature_for_bone(bone, context):
    if (
        context.object is not None
        and context.object.type == "ARMATURE"
        and bone.name in context.object.data.bones
    ):
        return context.object
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.data is not None and bone.name in obj.data.bones:
            return obj
    return None


def _update_bone_scale(bone, context):
    apply_bone_scale_preview(_find_armature_for_bone(bone, context))


def refresh_all_bone_scale_previews():
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            apply_bone_scale_preview(obj)


def _run_scheduled_bone_scale_preview_refresh():
    global _bone_scale_refresh_is_scheduled

    _bone_scale_refresh_is_scheduled = False
    refresh_all_bone_scale_previews()
    return None


def schedule_bone_scale_preview_refresh():
    global _bone_scale_refresh_is_scheduled

    if _bone_scale_refresh_is_scheduled:
        return
    bpy.app.timers.register(_run_scheduled_bone_scale_preview_refresh, first_interval=0.0)
    _bone_scale_refresh_is_scheduled = True


def register_properties():
    bpy.types.Bone.xv2_bone_scale = FloatVectorProperty(
        name="XV2 Bone Scale",
        description="Per-bone XV2 scale used by exporters when enabled",
        size=3,
        default=BONE_SCALE_IDENTITY,
        min=0.0,
        subtype="XYZ",
        update=_update_bone_scale,
    )
    schedule_bone_scale_preview_refresh()


def unregister_properties():
    global _bone_scale_refresh_is_scheduled

    if bpy.app.timers.is_registered(_run_scheduled_bone_scale_preview_refresh):
        bpy.app.timers.unregister(_run_scheduled_bone_scale_preview_refresh)
    _bone_scale_refresh_is_scheduled = False
    if hasattr(bpy.types.Bone, "xv2_bone_scale"):
        del bpy.types.Bone.xv2_bone_scale


__all__ = [
    "BONE_PT_xv2_bone_scale",
    "XV2_OT_bone_scale_reset_selected",
    "refresh_all_bone_scale_previews",
    "register_properties",
    "schedule_bone_scale_preview_refresh",
    "unregister_properties",
]
