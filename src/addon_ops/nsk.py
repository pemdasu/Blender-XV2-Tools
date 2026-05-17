import os

import bpy
from bpy.props import BoolProperty, FloatProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..xv2.NSK.exporter import export_nsk
from ..xv2.NSK.importer import import_nsk


class IMPORT_OT_nsk(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_nsk"
    bl_label = "Import NSK (Xenoverse 2)"

    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)  # type: ignore
    directory: StringProperty(subtype="DIR_PATH")  # type: ignore

    filename_ext = ".nsk"
    filter_glob: StringProperty(default="*.nsk", options={"HIDDEN"})  # type: ignore

    import_custom_normals: BoolProperty(  # type: ignore
        name="Import custom split normals",
        description=("Use normals stored in the embedded EMD file."),
        default=True,
    )
    import_tangents: BoolProperty(  # type: ignore
        name="Import tangents (if present)",
        default=False,
    )
    tris_to_quads: BoolProperty(  # type: ignore
        name="Convert tris to quads",
        default=False,
    )
    auto_merge_by_distance: BoolProperty(  # type: ignore
        name="Auto Merge by Distance",
        description="Merge nearby vertices after import",
        default=True,
    )
    merge_distance: FloatProperty(  # type: ignore
        name="Merge Distance",
        description="Distance threshold used by Auto Merge by Distance",
        default=0.0001,
        min=0.0,
        soft_max=0.01,
        precision=4,
        subtype="DISTANCE",
        unit="LENGTH",
    )
    split_into_submeshes: BoolProperty(  # type: ignore
        name="Split into submeshes",
        default=True,
    )
    reuse_materials: BoolProperty(  # type: ignore
        name="Reuse Materials",
        description="Reuse existing materials by name when the shader template matches",
        default=True,
    )
    preserve_bone_axes: BoolProperty(  # type: ignore
        name="Preserve Bone Axes",
        description=(
            "Build armature bones from source local axes. Helps mirrored chains keep matching"
        ),
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "import_custom_normals")
        layout.prop(self, "import_tangents")
        layout.prop(self, "tris_to_quads")
        layout.prop(self, "auto_merge_by_distance")
        layout.prop(self, "preserve_bone_axes")
        layout.prop(self, "reuse_materials")
        if self.auto_merge_by_distance:
            layout.prop(self, "merge_distance")

    def execute(self, context):
        paths: list[str] = []
        if self.files:
            for file_entry in self.files:
                paths.append(os.path.join(self.directory, file_entry.name))
        else:
            paths.append(self.filepath)

        if not paths:
            self.report({"ERROR"}, "Select one or more .nsk files to import.")
            return {"CANCELLED"}

        for path in paths:
            import_nsk(
                path,
                self.import_custom_normals,
                self.import_tangents,
                self.auto_merge_by_distance,
                self.merge_distance,
                self.tris_to_quads,
                self.split_into_submeshes,
                return_armature=False,
                reuse_materials=self.reuse_materials,
                preserve_bone_axes=self.preserve_bone_axes,
                warn=lambda msg: self.report({"WARNING"}, msg),
            )

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# EMO Import (.EMO container)
# ---------------------------------------------------------------------------


class EXPORT_OT_nsk(Operator, ExportHelper):
    bl_idname = "export_scene.xv2_nsk"
    bl_label = "Export NSK (Xenoverse 2)"

    filename_ext = ".nsk"
    filter_glob: StringProperty(default="*.nsk", options={"HIDDEN"})  # type: ignore

    def execute(self, context):
        arm = context.object if context.object and context.object.type == "ARMATURE" else None
        if arm is None:
            self.report({"ERROR"}, "Select an armature to export.")
            return {"CANCELLED"}
        ok, error = export_nsk(self.filepath, arm)
        if ok:
            self.report({"INFO"}, "Exported NSK")
            return {"FINISHED"}
        self.report({"ERROR"}, error or "Failed to export NSK.")
        return {"CANCELLED"}


CLASSES = [
    IMPORT_OT_nsk,
    EXPORT_OT_nsk,
]
