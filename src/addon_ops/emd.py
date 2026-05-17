import os
from pathlib import Path

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..ui import link_scd_armatures
from ..xv2.EMD.exporter import export_selected
from ..xv2.EMD.importer import import_emd


class IMPORT_OT_emd(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_emd"
    bl_label = "Import EMD (Xenoverse 2)"

    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)  # type: ignore
    directory: StringProperty(subtype="DIR_PATH")  # type: ignore

    filename_ext = ".emd"
    filter_glob: StringProperty(default="*.emd;*.esk", options={"HIDDEN"})  # type: ignore

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
        default=False,
    )
    reuse_materials: BoolProperty(  # type: ignore
        name="Reuse Materials",
        description="Reuse existing materials by name when the shader template matches",
        default=True,
    )
    preserve_structure: BoolProperty(  # type: ignore
        name="Preserve EMD hierarchy (empties)",
        default=False,
    )
    preserve_bone_axes: BoolProperty(  # type: ignore
        name="Preserve Bone Axes",
        description=(
            "Build armature bones from source local axes. Helps mirrored chains keep matching"
        ),
        default=False,
    )
    ignore_lod_files: BoolProperty(  # type: ignore
        name="Ignore LOD Files",
        description="Skip EMD files whose name contains _LOD, such as _LOD01",
        default=True,
    )
    dyt_entry_index: IntProperty(  # type: ignore
        name="DYT Entry Index",
        description="DYT texture entry to use (e.g. 2 -> DATA002)",
        default=0,
        min=0,
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
        layout.prop(self, "dyt_entry_index")
        layout.prop(self, "tris_to_quads")
        layout.prop(self, "auto_merge_by_distance")
        layout.prop(self, "preserve_bone_axes")
        layout.prop(self, "ignore_lod_files")
        layout.prop(self, "reuse_materials")
        if self.auto_merge_by_distance:
            layout.prop(self, "merge_distance")
        if not self.auto_detect_esk:
            layout.label(text="Tip: select an .esk in the file browser.")

    def execute(self, context):
        paths: list[str] = []
        if self.files:
            for file_entry in self.files:
                paths.append(os.path.join(self.directory, file_entry.name))
        else:
            paths.append(self.filepath)

        esk_path = "" if self.auto_detect_esk else self.esk_path
        filtered_paths: list[str] = []
        selected_esk: str | None = None
        for path in paths:
            if os.path.splitext(path)[1].lower() == ".esk":
                if selected_esk is None:
                    selected_esk = path
                continue
            filtered_paths.append(path)
        paths = filtered_paths
        if not self.auto_detect_esk and not esk_path and selected_esk:
            esk_path = selected_esk
        if not paths:
            self.report({"ERROR"}, "Select one or more .emd files to import.")
            return {"CANCELLED"}

        def is_scd_path(p: str) -> bool:
            return "_scd" in Path(p).stem.lower()

        def is_lod_path(p: str) -> bool:
            return "_lod" in Path(p).stem.lower()

        if self.ignore_lod_files:
            before_count = len(paths)
            paths = [path for path in paths if not is_lod_path(path)]
            skipped_count = before_count - len(paths)
            if skipped_count:
                self.report({"INFO"}, f"Skipped {skipped_count} LOD EMD file(s).")

        if not paths:
            self.report({"ERROR"}, "No non-LOD .emd files selected to import.")
            return {"CANCELLED"}

        # Import non-SCD first to get the main armature, then SCD files.
        paths_sorted = sorted(paths, key=lambda p: 1 if is_scd_path(p) else 0)

        main_arm_obj = None
        for path in paths_sorted:
            scd_file = is_scd_path(path)
            shared = None if scd_file else main_arm_obj

            arm_obj, _ = import_emd(
                path,
                esk_path,
                self.import_custom_normals,
                self.import_tangents,
                self.auto_merge_by_distance,
                self.merge_distance,
                self.tris_to_quads,
                self.split_into_submeshes,
                shared_armature=shared,
                return_armature=True,
                preserve_structure=self.preserve_structure,
                preserve_bone_axes=self.preserve_bone_axes,
                dyt_entry_index=self.dyt_entry_index,
                reuse_materials=self.reuse_materials,
                warn=lambda msg: self.report({"WARNING"}, msg),
            )

            if not scd_file and main_arm_obj is None:
                main_arm_obj = arm_obj

            if scd_file and main_arm_obj is not None and arm_obj is not None:
                link_scd_armatures(arm_obj, main_arm_obj)

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# NSK Import (.NSK container)
# ---------------------------------------------------------------------------


class EXPORT_OT_emd(Operator, ExportHelper):
    bl_idname = "export_scene.xv2_emd"
    bl_label = "Export EMD (Xenoverse 2)"

    filename_ext = ".emd"
    filter_glob: StringProperty(default="*.emd", options={"HIDDEN"})  # type: ignore
    use_bone_scale: BoolProperty(  # type: ignore
        name="Use XV2 Bone Scale",
        description="Export meshes with per-bone XV2 bone scale values applied",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "use_bone_scale")

    def execute(self, context):
        output_dir = os.path.dirname(self.filepath) if self.filepath else ""
        if not output_dir:
            self.report({"ERROR"}, "Please choose an output directory or file path.")
            return {"CANCELLED"}
        if not os.path.isdir(output_dir):
            self.report({"ERROR"}, f"Directory does not exist: {output_dir}")
            return {"CANCELLED"}

        written = export_selected(context, output_dir, use_bone_scale=self.use_bone_scale)
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


CLASSES = [
    IMPORT_OT_emd,
    EXPORT_OT_emd,
]
