import contextlib
import os
from pathlib import Path

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
    CameraEANProperties,
    DATA_PT_xv2_camera_actions,
    EMD_OT_texture_sampler_add,
    EMD_OT_texture_sampler_remove,
    EMD_OT_texture_sampler_sync_props,
    EMD_UL_texture_samplers,
    EMDTextureSamplerPropertyGroup,
    PROPERTIES_PT_emd_texture_samplers,
    SCDLinkSettings,
    VIEW3D_PT_emd_texture_samplers,
    VIEW3D_PT_scd_link,
    XV2_OT_cam_link_bone,
    XV2_OT_scd_link_to_armature,
    link_scd_armatures,
)
from .xv2.EAN.exporter import export_cam_ean, export_ean
from .xv2.EAN.importer import import_cam_ean, import_ean_animations
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

        esk_path = "" if self.auto_detect_esk else self.esk_path

        def is_scd_path(p: str) -> bool:
            return "_scd" in Path(p).stem.lower()

        # Import non-SCD first to get the main armature, then SCD files.
        paths_sorted = sorted(paths, key=lambda p: 1 if is_scd_path(p) else 0)

        main_arm_obj = None
        for path in paths_sorted:
            scd_file = is_scd_path(path)
            shared = None if scd_file else main_arm_obj

            arm_obj, esk = import_emd(
                path,
                esk_path,
                self.import_custom_normals,
                self.import_tangents,
                self.merge_by_distance,
                self.merge_distance,
                self.tris_to_quads,
                self.split_into_submeshes,
                shared_armature=shared,
                return_armature=True,
                preserve_structure=self.preserve_structure,
            )

            if not scd_file and main_arm_obj is None:
                main_arm_obj = arm_obj

            if scd_file and main_arm_obj is not None and arm_obj is not None:
                link_scd_armatures(arm_obj, main_arm_obj)

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Camera EAN Import (CAM.EAN)
# ---------------------------------------------------------------------------
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
                    "No meshes were exported (make sure meshes are selected and "
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
    self.layout.operator(
        IMPORT_OT_ean.bl_idname,
        text="Dragon Ball XV2 EAN (.ean)",
    )
    self.layout.operator(
        IMPORT_OT_cam_ean.bl_idname,
        text="Dragon Ball XV2 Camera EAN (.cam.ean)",
    )


def menu_func_export(self, context):
    self.layout.operator(
        EXPORT_OT_emd.bl_idname,
        text="Dragon Ball XV2 EMD (.emd)",
    )
    self.layout.operator(
        EXPORT_OT_ean.bl_idname,
        text="Dragon Ball XV2 EAN (.ean)",
    )
    self.layout.operator(
        EXPORT_OT_cam_ean.bl_idname,
        text="Dragon Ball XV2 Camera EAN (.cam.ean)",
    )


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

    def execute(self, context):
        target = context.object if context.object and context.object.type == "ARMATURE" else None
        arm = import_ean_animations(
            self.filepath,
            target_armature=target,
            replace_armature=self.replace_armature,
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

    def execute(self, context):
        arm = context.object if context.object and context.object.type == "ARMATURE" else None
        if arm is None:
            self.report({"ERROR"}, "Select an armature to export.")
            return {"CANCELLED"}
        ok = export_ean(self.filepath, arm)
        if ok:
            self.report({"INFO"}, "Exported EAN")
            return {"FINISHED"}
        self.report({"ERROR"}, "Failed to export EAN.")
        return {"CANCELLED"}


classes = [
    EMDTextureSamplerPropertyGroup,
    EMD_UL_texture_samplers,
    EMD_OT_texture_sampler_add,
    EMD_OT_texture_sampler_remove,
    EMD_OT_texture_sampler_sync_props,
    VIEW3D_PT_emd_texture_samplers,
    PROPERTIES_PT_emd_texture_samplers,
    SCDLinkSettings,
    VIEW3D_PT_scd_link,
    XV2_OT_scd_link_to_armature,
    IMPORT_OT_emd,
    IMPORT_OT_cam_ean,
    IMPORT_OT_ean,
    EXPORT_OT_emd,
    EXPORT_OT_ean,
    EXPORT_OT_cam_ean,
    CameraEANProperties,
    DATA_PT_xv2_camera_actions,
    XV2_OT_cam_link_bone,
]


def _register_class(cls):
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        # If already registered (e.g., after a reload), unregister and try again.
        with contextlib.suppress(Exception):
            bpy.utils.unregister_class(cls)
        bpy.utils.register_class(cls)


def _unregister_class(cls):
    with contextlib.suppress(Exception):
        bpy.utils.unregister_class(cls)


def register():
    for cls in classes:
        _register_class(cls)

    bpy.types.Scene.xv2_scd_link = bpy.props.PointerProperty(type=SCDLinkSettings)
    bpy.types.Object.emd_texture_samplers = CollectionProperty(type=EMDTextureSamplerPropertyGroup)
    bpy.types.Object.emd_texture_samplers_index = IntProperty(default=0)
    bpy.types.Material.emd_texture_samplers = CollectionProperty(
        type=EMDTextureSamplerPropertyGroup
    )
    bpy.types.Material.emd_texture_samplers_index = IntProperty(default=0)
    bpy.types.Scene.xv2_cam_props = bpy.props.PointerProperty(type=CameraEANProperties)
    bpy.types.Camera.xv2_fov = bpy.props.FloatProperty(name="XV2 FOV (deg)", default=40.0)
    bpy.types.Camera.xv2_roll = bpy.props.FloatProperty(name="XV2 Roll (deg)", default=0.0)

    bpy.types.TOPBAR_MT_file_import.append(menu_func)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

    del bpy.types.Scene.xv2_scd_link
    del bpy.types.Object.emd_texture_samplers
    del bpy.types.Object.emd_texture_samplers_index
    del bpy.types.Material.emd_texture_samplers
    del bpy.types.Material.emd_texture_samplers_index
    del bpy.types.Scene.xv2_cam_props
    del bpy.types.Camera.xv2_fov
    del bpy.types.Camera.xv2_roll

    for cls in reversed(classes):
        _unregister_class(cls)


if __name__ == "__main__":
    register()
