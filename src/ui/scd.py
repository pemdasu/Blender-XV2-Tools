import bpy
from bpy.props import EnumProperty, PointerProperty, StringProperty


def _armature_poll(_self, obj):
    return obj and obj.type == "ARMATURE"


def _target_bone_items(self, context):
    arm = self.target_armature
    if arm is None and context is not None:
        obj = context.object
        if obj and obj.type == "ARMATURE":
            arm = obj
    items = [("NONE", "None", "Skip fallback")]
    if arm and arm.type == "ARMATURE" and arm.data:
        for bone in arm.data.bones:
            items.append((bone.name, bone.name, ""))
    return items


def link_scd_armatures(source: bpy.types.Object, target: bpy.types.Object) -> tuple[int, int]:
    if source is None or target is None:
        return 0, 0

    mapped: dict[str, str] = {}
    for pbone in source.pose.bones:
        name_lower = pbone.name.lower()
        if name_lower.startswith("scd_"):
            continue
        if pbone.name in target.pose.bones:
            mapped[pbone.name] = pbone.name

    added = 0
    skipped = 0
    for pbone in source.pose.bones:
        dest_name = mapped.get(pbone.name, "")
        if not dest_name:
            skipped += 1
            continue

        existing = pbone.constraints.get("SCD_Link")
        if existing and existing.type != "COPY_TRANSFORMS":
            pbone.constraints.remove(existing)
            existing = None

        constraint = existing or pbone.constraints.new(type="COPY_TRANSFORMS")
        constraint.name = "SCD_Link"
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
    target_bone: EnumProperty(  # type: ignore
        name="Target Bone",
        items=_target_bone_items,
        description="Bone on the target armature used when no name match is found",
        default=0,  # index into items; required when items is a callable
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
        # layout.prop(settings, "target_bone")
        layout.operator(XV2_OT_scd_link_to_armature.bl_idname, icon="CON_TRANSFORM")

        if settings.report:
            layout.label(text=settings.report)
        layout.label(text="Skips bones starting with 'scd_'.")


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
