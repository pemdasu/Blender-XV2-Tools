from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..xv2.ESK.exporter import export_esk
from ..xv2.ESK.importer import import_esk


class IMPORT_OT_esk(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_esk"
    bl_label = "Import ESK (Xenoverse 2)"

    filename_ext = ".esk"
    filter_glob: StringProperty(default="*.esk", options={"HIDDEN"})  # type: ignore
    preserve_bone_axes: BoolProperty(  # type: ignore
        name="Preserve Bone Axes",
        description=(
            "Build armature bones from source local axes. Helps mirrored chains keep matching"
        ),
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preserve_bone_axes")

    def execute(self, context):
        arm = import_esk(self.filepath, preserve_bone_axes=self.preserve_bone_axes)
        if arm:
            context.scene.view_settings.view_transform = "Standard"
            self.report({"INFO"}, f"Imported ESK armature {arm.name}")
            return {"FINISHED"}
        self.report({"ERROR"}, "Failed to import ESK.")
        return {"CANCELLED"}


# ---------------------------------------------------------------------------
# Camera EAN Import (CAM.EAN)
# ---------------------------------------------------------------------------


class EXPORT_OT_esk(Operator, ExportHelper):
    bl_idname = "export_scene.xv2_esk"
    bl_label = "Export ESK (Xenoverse 2)"

    filename_ext = ".esk"
    filter_glob: StringProperty(default="*.esk", options={"HIDDEN"})  # type: ignore

    def execute(self, context):
        arm = context.object if context.object and context.object.type == "ARMATURE" else None
        if arm is None:
            self.report({"ERROR"}, "Select an armature to export.")
            return {"CANCELLED"}
        ok, error = export_esk(self.filepath, arm)
        if ok:
            self.report({"INFO"}, "Exported ESK")
            return {"FINISHED"}
        self.report({"ERROR"}, error or "Failed to export ESK.")
        return {"CANCELLED"}


CLASSES = [
    IMPORT_OT_esk,
    EXPORT_OT_esk,
]
