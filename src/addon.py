import os

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from .ui import (
    EMD_OT_texture_sampler_add,
    EMD_OT_texture_sampler_remove,
    EMD_OT_texture_sampler_sync_props,
    EMD_UL_texture_samplers,
    EMDTextureSamplerPropertyGroup,
    PROPERTIES_PT_emd_texture_samplers,
    VIEW3D_PT_emd_texture_samplers,
)
from .xv2.EMD.exporter import export_selected
from .xv2.EMD.importer import import_emd


# ---------------------------------------------------------------------------
# Import operator
# ---------------------------------------------------------------------------
class IMPORT_OT_emd(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_emd"
    bl_label = "Import EMD (Xenoverse 2)"

    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)  # type: ignore
    directory: StringProperty(subtype="DIR_PATH")  # type: ignore

    filename_ext = ".emd"
    filter_glob: StringProperty(default="*.emd", options={"HIDDEN"})  # type: ignore

    auto_detect_esk: BoolProperty(  # type: ignore
        name="Auto-detect ESK",
        default=True,
    )
    import_custom_normals: BoolProperty(  # type: ignore
        name="Import custom split normals",
        description=("Use normals stored in the EMD file."),
        default=True,
    )
    import_tangents: BoolProperty(  # type: ignore
        name="Import tangents (if present)",
        default=False,
    )
    merge_by_distance: BoolProperty(  # type: ignore
        name="Auto merge by distance",
        default=False,
    )
    merge_distance: FloatProperty(  # type: ignore
        name="Merge Distance",
        default=0.0001,
        min=0.0,
    )
    tris_to_quads: BoolProperty(  # type: ignore
        name="Convert tris to quads",
        default=False,
    )
    split_into_submeshes: BoolProperty(  # type: ignore
        name="Split into submeshes",
        default=False,
    )
    preserve_structure: BoolProperty(  # type: ignore
        name="Preserve EMD hierarchy (empties)",
        default=False,
    )
    esk_path: StringProperty(  # type: ignore
        name="ESK File",
        subtype="FILE_PATH",
    )

    def draw(self, context):
        layout = self.layout
        # layout.prop(self, "split_into_submeshes")
        # if self.split_into_submeshes:
        #     layout.prop(self, "preserve_structure")
        layout.prop(self, "auto_detect_esk")
        layout.prop(self, "import_custom_normals")
        layout.prop(self, "import_tangents")
        layout.prop(self, "merge_by_distance")
        if self.merge_by_distance:
            layout.prop(self, "merge_distance")
        layout.prop(self, "tris_to_quads")
        if not self.auto_detect_esk:
            layout.prop(self, "esk_path", text="Skeleton (.esk)")

    def execute(self, context):
        paths: list[str] = []
        if self.files:
            for file_entry in self.files:
                paths.append(os.path.join(self.directory, file_entry.name))
        else:
            paths.append(self.filepath)

        arm_obj = None
        esk = None
        esk_path = "" if self.auto_detect_esk else self.esk_path

        if paths:
            first = paths[0]
            arm_obj, esk = import_emd(
                first,
                esk_path,
                self.import_custom_normals,
                self.import_tangents,
                self.merge_by_distance,
                self.merge_distance,
                self.tris_to_quads,
                self.split_into_submeshes,
                return_armature=True,
                preserve_structure=self.preserve_structure,
            )

        for path in paths[1:]:
            import_emd(
                path,
                esk_path,
                self.import_custom_normals,
                self.import_tangents,
                self.merge_by_distance,
                self.merge_distance,
                self.tris_to_quads,
                self.split_into_submeshes,
                shared_armature=arm_obj,
                preserve_structure=self.preserve_structure,
            )

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Export operator
# ---------------------------------------------------------------------------
class EXPORT_OT_emd(Operator, ExportHelper):
    bl_idname = "export_scene.xv2_emd"
    bl_label = "Export EMD (Xenoverse 2)"

    filename_ext = ".emd"
    filter_glob: StringProperty(default="*.emd", options={"HIDDEN"})  # type: ignore

    def execute(self, context):
        output_dir = os.path.dirname(self.filepath) if self.filepath else ""
        if not output_dir:
            self.report({"ERROR"}, "Please choose an output directory or file path.")
            return {"CANCELLED"}
        if not os.path.isdir(output_dir):
            self.report({"ERROR"}, f"Directory does not exist: {output_dir}")
            return {"CANCELLED"}

        written = export_selected(context, output_dir)
        if not written:
            self.report(
                {"WARNING"},
                (
                    "No meshes were exported (ensure meshes are selected and "
                    "parented to an armature)."
                ),
            )
            return {"CANCELLED"}

        self.report({"INFO"}, f"Exported {len(written)} EMD file(s).")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def menu_func(self, context):
    self.layout.operator(
        IMPORT_OT_emd.bl_idname,
        text="Dragon Ball XV2 EMD (.emd)",
    )


def menu_func_export(self, context):
    self.layout.operator(
        EXPORT_OT_emd.bl_idname,
        text="Dragon Ball XV2 EMD (.emd)",
    )


classes = [
    EMDTextureSamplerPropertyGroup,
    EMD_UL_texture_samplers,
    EMD_OT_texture_sampler_add,
    EMD_OT_texture_sampler_remove,
    EMD_OT_texture_sampler_sync_props,
    VIEW3D_PT_emd_texture_samplers,
    PROPERTIES_PT_emd_texture_samplers,
    IMPORT_OT_emd,
    EXPORT_OT_emd,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Object.emd_texture_samplers = CollectionProperty(type=EMDTextureSamplerPropertyGroup)
    bpy.types.Object.emd_texture_samplers_index = IntProperty(default=0)
    bpy.types.Material.emd_texture_samplers = CollectionProperty(
        type=EMDTextureSamplerPropertyGroup
    )
    bpy.types.Material.emd_texture_samplers_index = IntProperty(default=0)

    bpy.types.TOPBAR_MT_file_import.append(menu_func)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

    del bpy.types.Object.emd_texture_samplers
    del bpy.types.Object.emd_texture_samplers_index
    del bpy.types.Material.emd_texture_samplers
    del bpy.types.Material.emd_texture_samplers_index

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
