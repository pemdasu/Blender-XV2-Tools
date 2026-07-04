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
    bake_visual_keying: BoolProperty(  # type: ignore
        name="Bake with Visual Keying",
        description=(
            "Sample every frame so constraint, driver, and IK motion is baked into the export"
        ),
        default=True,
    )

    def check(self, context):
        # ExportHelper's default check() runs os.path.splitext, which only sees ".ean" in the
        # compound ".cam.ean" extension and re-appends the full ext on every keystroke. Handle the
        # compound extension ourselves: strip any trailing extension pieces, then add exactly one.
        ext = self.filename_ext
        filepath = self.filepath
        if not filepath:
            return False
        base = filepath
        while True:
            lowered = base.lower()
            for piece in (ext.lower(), ".ean", ".cam"):
                if lowered.endswith(piece):
                    base = base[: -len(piece)]
                    break
            else:
                break
        new_path = base + ext
        if new_path != filepath:
            self.filepath = new_path
            return True
        return False

    def draw(self, context):
        self.layout.prop(self, "bake_visual_keying")

    def execute(self, context):
        rig = context.object
        ok = export_cam_ean(
            self.filepath, rig_obj=rig, bake_visual_keying=self.bake_visual_keying
        )
        if ok:
            self.report({"INFO"}, "Exported Camera EAN")
            return {"FINISHED"}
        self.report(
            {"ERROR"},
            "Failed to export Camera EAN. Select a camera rig (Node_/Target_ actions) "
            "or a legacy rig (camera named 'Node' with +Name/-Name action pairs).",
        )
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
    bake_visual_keying: BoolProperty(  # type: ignore
        name="Bake with Visual Keying",
        description=(
            "Sample every frame so constraint, driver, and IK motion is baked into the export"
        ),
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "add_dummy_rest_keys")
        layout.prop(self, "use_bone_scale")
        layout.prop(self, "bake_visual_keying")

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
            bake_visual_keying=self.bake_visual_keying,
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
