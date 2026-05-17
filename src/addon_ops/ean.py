from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..xv2.EAN.exporter import export_cam_ean, export_ean
from ..xv2.EAN.importer import import_cam_ean, import_ean_animations


class IMPORT_OT_cam_ean(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_cam_ean"
    bl_label = "Import Camera EAN (Xenoverse 2)"

    filename_ext = ".cam.ean"
    filter_glob: StringProperty(default="*.cam.ean", options={"HIDDEN"})  # type: ignore

    def execute(self, context):
        created = import_cam_ean(self.filepath)
        if created:
            self.report({"INFO"}, "Imported camera EAN")
            return {"FINISHED"}

        self.report({"WARNING"}, "Not a camera EAN or nothing was created.")
        return {"CANCELLED"}


# ---------------------------------------------------------------------------
# Export operator
# ---------------------------------------------------------------------------


class IMPORT_OT_ean(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_ean"
    bl_label = "Import EAN (Xenoverse 2)"

    filename_ext = ".ean"
    filter_glob: StringProperty(default="*.ean", options={"HIDDEN"})  # type: ignore
    replace_armature: BoolProperty(  # type: ignore
        name="Replace selected armature",
        description="Ignore the selected armature and build one from the EAN skeleton",
        default=False,
    )
    preserve_bone_axes: BoolProperty(  # type: ignore
        name="Preserve Bone Axes",
        description=("When creating/replacing an armature, build bones from source local axes"),
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "replace_armature")
        layout.prop(self, "preserve_bone_axes")

    def execute(self, context):
        target = context.object if context.object and context.object.type == "ARMATURE" else None
        arm = import_ean_animations(
            self.filepath,
            target_armature=target,
            replace_armature=self.replace_armature,
            preserve_bone_axes=self.preserve_bone_axes,
        )
        if arm:
            self.report({"INFO"}, f"Imported EAN onto armature {arm.name}")
            return {"FINISHED"}
        self.report({"WARNING"}, "Nothing imported.")
        return {"CANCELLED"}


class EXPORT_OT_cam_ean(Operator, ExportHelper):
    bl_idname = "export_scene.xv2_cam_ean"
    bl_label = "Export Camera EAN (Xenoverse 2)"

    filename_ext = ".cam.ean"
    filter_glob: StringProperty(default="*.cam.ean", options={"HIDDEN"})  # type: ignore

    def execute(self, context):
        rig = context.object
        ok = export_cam_ean(self.filepath, rig_obj=rig)
        if ok:
            self.report({"INFO"}, "Exported Camera EAN")
            return {"FINISHED"}
        self.report({"ERROR"}, "Failed to export Camera EAN (select a camera rig).")
        return {"CANCELLED"}


class EXPORT_OT_ean(Operator, ExportHelper):
    bl_idname = "export_scene.xv2_ean"
    bl_label = "Export EAN (Xenoverse 2)"

    filename_ext = ".ean"
    filter_glob: StringProperty(default="*.ean", options={"HIDDEN"})  # type: ignore
    add_dummy_rest_keys: BoolProperty(  # type: ignore
        name="Add Dummy Keyframes",
        description="Add a rest pose keyframe at frame 0 for bones with no keyframes",
        default=False,
    )
    use_bone_scale: BoolProperty(  # type: ignore
        name="Use XV2 Bone Scale",
        description="Bake per-bone XV2 bone scale into exported EAN positions",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "add_dummy_rest_keys")
        layout.prop(self, "use_bone_scale")

    def execute(self, context):
        arm = context.object if context.object and context.object.type == "ARMATURE" else None
        if arm is None:
            self.report({"ERROR"}, "Select an armature to export.")
            return {"CANCELLED"}
        ok, error = export_ean(
            self.filepath,
            arm,
            add_dummy_rest=self.add_dummy_rest_keys,
            use_bone_scale=self.use_bone_scale,
        )
        if ok:
            self.report({"INFO"}, "Exported EAN")
            return {"FINISHED"}
        self.report({"ERROR"}, error or "Failed to export EAN.")
        return {"CANCELLED"}


CLASSES = [
    IMPORT_OT_cam_ean,
    IMPORT_OT_ean,
    EXPORT_OT_cam_ean,
    EXPORT_OT_ean,
]
