from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..xv2.EMA.exporter import export_ema
from ..xv2.EMA.importer import import_ema_animations


def is_obj_ema_path(path: str) -> bool:
    return path.lower().endswith(".obj.ema")


class IMPORT_OT_ema(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_ema"
    bl_label = "Import EMA (Xenoverse 2)"

    filename_ext = ".obj.ema"
    filter_glob: StringProperty(default="*.obj.ema", options={"HIDDEN"})  # type: ignore
    replace_armature: BoolProperty(  # type: ignore
        name="Replace selected armature",
        description="Ignore the selected armature and build one from the EMA skeleton",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "replace_armature")

    def execute(self, context):
        if not is_obj_ema_path(self.filepath):
            self.report({"ERROR"}, "Only .obj.ema files are supported.")
            return {"CANCELLED"}
        target = context.object if context.object and context.object.type == "ARMATURE" else None
        try:
            arm = import_ema_animations(
                self.filepath,
                target_armature=target,
                replace_armature=self.replace_armature,
                preserve_bone_axes=True,
            )
        except (RuntimeError, OSError, ValueError, TypeError) as error:
            self.report({"ERROR"}, f"Failed to import EMA: {error}")
            return {"CANCELLED"}

        if arm:
            self.report({"INFO"}, f"Imported EMA onto armature {arm.name}")
            return {"FINISHED"}
        self.report({"WARNING"}, "Nothing imported.")
        return {"CANCELLED"}


class EXPORT_OT_ema(Operator, ExportHelper):
    bl_idname = "export_scene.xv2_ema"
    bl_label = "Export EMA (Xenoverse 2)"

    filename_ext = ".obj.ema"
    filter_glob: StringProperty(default="*.obj.ema", options={"HIDDEN"})  # type: ignore
    add_dummy_rest_keys: BoolProperty(  # type: ignore
        name="Add Dummy Keyframes",
        description="Add a rest pose keyframe at frame 0 for bones with no keyframes",
        default=False,
    )

    def execute(self, context):
        if not is_obj_ema_path(self.filepath):
            self.report({"ERROR"}, "EMA export only writes .obj.ema files.")
            return {"CANCELLED"}
        arm = context.object if context.object and context.object.type == "ARMATURE" else None
        if arm is None:
            self.report({"ERROR"}, "Select an armature to export.")
            return {"CANCELLED"}
        ok, error = export_ema(self.filepath, arm, add_dummy_rest=self.add_dummy_rest_keys)
        if ok:
            self.report({"INFO"}, "Exported EMA")
            return {"FINISHED"}
        self.report({"ERROR"}, error or "Failed to export EMA.")
        return {"CANCELLED"}


CLASSES = [
    IMPORT_OT_ema,
    EXPORT_OT_ema,
]
