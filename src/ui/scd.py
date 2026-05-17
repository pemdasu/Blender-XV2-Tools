import bpy
from bpy.props import PointerProperty, StringProperty

from ..xv2.consts import (
    SCD_LINK_CONSTRAINT_NAME,
    SCD_LINK_TARGET_ARMATURE_PROP,
    SCD_LINK_TARGET_BONE_PROP,
)


def _armature_poll(_self, obj):
    return obj and obj.type == "ARMATURE"


def _remove_scd_constraints(pose_bone: bpy.types.PoseBone) -> None:
    for constraint in list(pose_bone.constraints):
        if constraint.name == SCD_LINK_CONSTRAINT_NAME or constraint.name.startswith(
            f"{SCD_LINK_CONSTRAINT_NAME}_"
        ):
            pose_bone.constraints.remove(constraint)


def link_scd_armatures(
    source: bpy.types.Object,
    target: bpy.types.Object,
) -> tuple[int, int]:
    if source is None or target is None or source.data is None or target.data is None:
        return 0, 0

    mapped: dict[str, str] = {}
    for bone in source.data.bones:
        name_lower = bone.name.lower()
        if name_lower.startswith("scd_"):
            continue
        if bone.name in target.data.bones:
            mapped[bone.name] = bone.name

    source[SCD_LINK_TARGET_ARMATURE_PROP] = target.name
    source_world = source.matrix_world.copy()
    source.parent = target
    source.matrix_parent_inverse = target.matrix_world.inverted()
    source.matrix_world = source_world

    added = 0
    skipped = 0
    for pose_bone in source.pose.bones:
        _remove_scd_constraints(pose_bone)

    for bone in source.data.bones:
        dest_name = mapped.get(bone.name, "")
        if not dest_name:
            if SCD_LINK_TARGET_BONE_PROP in bone:
                del bone[SCD_LINK_TARGET_BONE_PROP]
            skipped += 1
            continue

        bone[SCD_LINK_TARGET_BONE_PROP] = dest_name
        pose_bone = source.pose.bones.get(bone.name)
        if pose_bone is not None:
            constraint = pose_bone.constraints.new(type="COPY_TRANSFORMS")
            constraint.name = SCD_LINK_CONSTRAINT_NAME
            constraint.target = target
            constraint.subtarget = dest_name
        added += 1

    return added, skipped


class SCDLinkSettings(bpy.types.PropertyGroup):
    source_armature: PointerProperty(  # type: ignore
        name="SCD Armature",
        type=bpy.types.Object,
        poll=_armature_poll,
    )
    target_armature: PointerProperty(  # type: ignore
        name="Target Armature",
        type=bpy.types.Object,
        poll=_armature_poll,
    )
    report: StringProperty(name="Status", default="")  # type: ignore


class XV2_OT_scd_link_to_armature(bpy.types.Operator):
    bl_idname = "xv2.scd_link_to_armature"
    bl_label = "Link SCD"

    @classmethod
    def poll(cls, context):
        return hasattr(context.scene, "xv2_scd_link")

    def execute(self, context):
        settings: SCDLinkSettings = context.scene.xv2_scd_link
        source = settings.source_armature or (
            context.object if context.object and context.object.type == "ARMATURE" else None
        )
        target = settings.target_armature

        if source is None or target is None:
            self.report({"ERROR"}, "Select both SCD and target armatures.")
            return {"CANCELLED"}

        added, skipped = link_scd_armatures(source, target)
        settings.report = f"Linked {added} bone(s); skipped {skipped}."
        self.report({"INFO"}, settings.report)
        return {"FINISHED"}


class VIEW3D_PT_scd_link(bpy.types.Panel):
    bl_label = "SCD Link"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SCD"

    def draw(self, context):
        layout = self.layout
        settings: SCDLinkSettings = context.scene.xv2_scd_link

        layout.prop(settings, "source_armature")
        layout.prop(settings, "target_armature")
        layout.operator(XV2_OT_scd_link_to_armature.bl_idname, icon="CON_TRANSFORM")

        if settings.report:
            layout.label(text=settings.report)
        layout.label(text="Adds SCD copy-transform links and parents the armature.")


classes = [
    SCDLinkSettings,
    XV2_OT_scd_link_to_armature,
    VIEW3D_PT_scd_link,
]


__all__ = [
    "SCDLinkSettings",
    "XV2_OT_scd_link_to_armature",
    "VIEW3D_PT_scd_link",
    "link_scd_armatures",
]
