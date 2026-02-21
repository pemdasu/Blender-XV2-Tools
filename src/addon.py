import contextlib
import os
from collections.abc import Iterator
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
    XV2_OT_cam_create_actions,
    XV2_OT_cam_link_bone,
    XV2_OT_cam_rename_actions,
    XV2_OT_scd_link_to_armature,
    link_scd_armatures,
)
from .xv2.EAN.exporter import export_cam_ean, export_ean
from .xv2.EAN.importer import import_cam_ean, import_ean_animations
from .xv2.EMD.exporter import export_selected
from .xv2.EMD.importer import import_emd
from .xv2.ESK.exporter import export_esk
from .xv2.ESK.importer import import_esk
from .xv2.FMP.exporter import export_map
from .xv2.FMP.importer import import_map_in_steps
from .xv2.NSK.exporter import export_nsk
from .xv2.NSK.importer import import_nsk


# ---------------------------------------------------------------------------
# Import operator
# ---------------------------------------------------------------------------
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
    preserve_structure: BoolProperty(  # type: ignore
        name="Preserve EMD hierarchy (empties)",
        default=False,
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
                dyt_entry_index=self.dyt_entry_index,
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

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "import_custom_normals")
        layout.prop(self, "import_tangents")
        layout.prop(self, "tris_to_quads")
        layout.prop(self, "auto_merge_by_distance")
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
                warn=lambda msg: self.report({"WARNING"}, msg),
            )

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# FMP MAP Import (.MAP)
# ---------------------------------------------------------------------------
class IMPORT_OT_map(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_map"
    bl_label = "Import MAP/FMP (Xenoverse 2)"

    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)  # type: ignore
    directory: StringProperty(subtype="DIR_PATH")  # type: ignore

    filename_ext = ".map"
    filter_glob: StringProperty(default="*.map", options={"HIDDEN"})  # type: ignore

    import_custom_normals: BoolProperty(  # type: ignore
        name="Import custom split normals",
        description=("Use normals stored in embedded NSK EMDs."),
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
    import_colliders: BoolProperty(  # type: ignore
        name="Import colliders",
        description="Create collider empties and collider custom properties",
        default=True,
    )
    import_collision_meshes: BoolProperty(  # type: ignore
        name="Import collider meshes",
        description="Create mesh objects from collision vertex/index data",
        default=True,
    )
    use_collection_instances: BoolProperty(  # type: ignore
        name="Use collection instances (faster)",
        description=(
            "Reuse imported NSK scenes as Blender collection instances. Faster and lighter, "
            "but less direct per-instance mesh editing"
        ),
        default=True,
    )
    _timer = None
    _paths: list[str]
    _next_path_index: int
    _active_path_index: int
    _active_path: str
    _active_iterator: Iterator[tuple[int, int, str]] | None
    _imported_count: int

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "import_custom_normals")
        layout.prop(self, "import_tangents")
        layout.prop(self, "tris_to_quads")
        layout.prop(self, "auto_merge_by_distance")
        if self.auto_merge_by_distance:
            layout.prop(self, "merge_distance")
        layout.prop(self, "import_colliders")
        if self.import_colliders:
            layout.prop(self, "import_collision_meshes")
        layout.prop(self, "use_collection_instances")

    def _cleanup_modal(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        context.window.cursor_set("DEFAULT")
        with contextlib.suppress(AttributeError, RuntimeError):
            context.workspace.status_text_set(None)

    def _start_next_import(self, context) -> bool:
        if self._next_path_index >= len(self._paths):
            return False

        self._active_path_index = self._next_path_index
        self._active_path = self._paths[self._active_path_index]
        self._next_path_index += 1
        self._active_iterator = import_map_in_steps(
            self._active_path,
            import_normals=self.import_custom_normals,
            import_tangents=self.import_tangents,
            merge_by_distance=self.auto_merge_by_distance,
            merge_distance=self.merge_distance,
            tris_to_quads=self.tris_to_quads,
            split_submeshes=self.split_into_submeshes,
            import_colliders=self.import_colliders,
            import_collision_meshes=self.import_collision_meshes,
            use_collection_instances=self.use_collection_instances,
            warn=lambda msg: self.report({"WARNING"}, msg),
        )
        print(f"[XV2 MAP] Importing {os.path.basename(self._active_path)}...")
        return True

    def modal(self, context, event):
        if event.type == "ESC":
            self._cleanup_modal(context)
            self.report({"WARNING"}, "MAP import cancelled.")
            print("[XV2 MAP] Import cancelled.")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"RUNNING_MODAL"}

        if self._active_iterator is None and not self._start_next_import(context):
            self._cleanup_modal(context)
            if self._imported_count == 0:
                self.report({"WARNING"}, "No MAP files were imported.")
                return {"CANCELLED"}
            self.report({"INFO"}, f"Imported {self._imported_count} MAP file(s).")
            print(f"[XV2 MAP] Finished. Imported {self._imported_count} file(s).")
            return {"FINISHED"}

        try:
            done_steps, total_steps, message = next(self._active_iterator)
            path_progress = (float(done_steps) / float(total_steps)) if total_steps > 0 else 1.0
            overall_progress = float(self._active_path_index) + max(0.0, min(1.0, path_progress))
            context.window_manager.progress_update(overall_progress)
            with contextlib.suppress(AttributeError, RuntimeError):
                context.workspace.status_text_set(message)
            print(message)
            return {"RUNNING_MODAL"}
        except StopIteration as stop:
            if stop.value is not None:
                self._imported_count += 1
            context.window_manager.progress_update(float(self._next_path_index))
            self._active_iterator = None
            return {"RUNNING_MODAL"}
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            self._cleanup_modal(context)
            self.report(
                {"ERROR"},
                f"Failed to import MAP {os.path.basename(self._active_path)}: {exc}",
            )
            print(f"[XV2 MAP] Failed to import {os.path.basename(self._active_path)}: {exc}")
            return {"CANCELLED"}

    def execute(self, context):
        paths: list[str] = []
        if self.files:
            for file_entry in self.files:
                paths.append(os.path.join(self.directory, file_entry.name))
        else:
            paths.append(self.filepath)

        if not paths:
            self.report({"ERROR"}, "Select one or more .map files to import.")
            return {"CANCELLED"}

        self._paths = paths
        self._next_path_index = 0
        self._active_path_index = 0
        self._active_path = ""
        self._active_iterator = None
        self._imported_count = 0

        wm = context.window_manager
        wm.progress_begin(0.0, float(len(paths)))
        context.window.cursor_set("WAIT")
        with contextlib.suppress(AttributeError, RuntimeError):
            context.workspace.status_text_set("[MAP] Starting import...")
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}


# ---------------------------------------------------------------------------
# ESK Import (ESK)
# ---------------------------------------------------------------------------
class IMPORT_OT_esk(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_esk"
    bl_label = "Import ESK (Xenoverse 2)"

    filename_ext = ".esk"
    filter_glob: StringProperty(default="*.esk", options={"HIDDEN"})  # type: ignore

    def execute(self, context):
        arm = import_esk(self.filepath)
        if arm:
            self.report({"INFO"}, f"Imported ESK armature {arm.name}")
            return {"FINISHED"}
        self.report({"ERROR"}, "Failed to import ESK.")
        return {"CANCELLED"}


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


class EXPORT_OT_map(Operator, ExportHelper):
    bl_idname = "export_scene.xv2_map"
    bl_label = "Export MAP/FMP (Xenoverse 2)"

    filename_ext = ".map"
    filter_glob: StringProperty(default="*.map", options={"HIDDEN"})  # type: ignore
    export_collision_meshes: BoolProperty(  # type: ignore
        name="Export collision meshes",
        description=(
            "Write edited collider mesh vertices/triangles back into source MAP collision data "
            "when possible"
        ),
        default=False,
    )
    export_linked_nsk: BoolProperty(  # type: ignore
        name="Export linked NSK files",
        description=(
            "Also export NSK armatures referenced by MAP entities using their MAP relative paths"
        ),
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "export_collision_meshes")
        layout.prop(self, "export_linked_nsk")

    def execute(self, context):
        selected = context.object
        map_root = None
        if selected is not None:
            if selected.get("fmp_source_path"):
                map_root = selected
            elif selected.parent and selected.parent.get("fmp_source_path"):
                map_root = selected.parent

        ok, error = export_map(
            self.filepath,
            map_root=map_root,
            export_collision_meshes=self.export_collision_meshes,
            export_linked_nsk=self.export_linked_nsk,
            warn=lambda msg: self.report({"WARNING"}, msg),
        )
        if ok:
            self.report({"INFO"}, "Exported MAP")
            return {"FINISHED"}
        self.report({"ERROR"}, error or "Failed to export MAP.")
        return {"CANCELLED"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def menu_func(self, _context):
    self.layout.operator(
        IMPORT_OT_emd.bl_idname,
        text="Dragon Ball XV2 EMD (.emd)",
    )
    self.layout.operator(
        IMPORT_OT_esk.bl_idname,
        text="Dragon Ball XV2 ESK (.esk)",
    )
    self.layout.operator(
        IMPORT_OT_ean.bl_idname,
        text="Dragon Ball XV2 EAN (.ean)",
    )
    self.layout.operator(
        IMPORT_OT_cam_ean.bl_idname,
        text="Dragon Ball XV2 Camera EAN (.cam.ean)",
    )
    self.layout.operator(
        IMPORT_OT_nsk.bl_idname,
        text="Dragon Ball XV2 NSK (.nsk)",
    )
    self.layout.operator(
        IMPORT_OT_map.bl_idname,
        text="Dragon Ball XV2 MAP/FMP (.map)",
    )


def menu_func_export(self, _context):
    self.layout.operator(
        EXPORT_OT_emd.bl_idname,
        text="Dragon Ball XV2 EMD (.emd)",
    )
    self.layout.operator(
        EXPORT_OT_esk.bl_idname,
        text="Dragon Ball XV2 ESK (.esk)",
    )
    self.layout.operator(
        EXPORT_OT_ean.bl_idname,
        text="Dragon Ball XV2 EAN (.ean)",
    )
    self.layout.operator(
        EXPORT_OT_cam_ean.bl_idname,
        text="Dragon Ball XV2 Camera EAN (.cam.ean)",
    )
    self.layout.operator(
        EXPORT_OT_nsk.bl_idname,
        text="Dragon Ball XV2 NSK (.nsk)",
    )
    self.layout.operator(
        EXPORT_OT_map.bl_idname,
        text="Dragon Ball XV2 MAP/FMP (.map)",
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

    def execute(self, context):
        arm = context.object if context.object and context.object.type == "ARMATURE" else None
        if arm is None:
            self.report({"ERROR"}, "Select an armature to export.")
            return {"CANCELLED"}
        ok, error = export_ean(self.filepath, arm, add_dummy_rest=self.add_dummy_rest_keys)
        if ok:
            self.report({"INFO"}, "Exported EAN")
            return {"FINISHED"}
        self.report({"ERROR"}, error or "Failed to export EAN.")
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
    IMPORT_OT_nsk,
    IMPORT_OT_map,
    IMPORT_OT_esk,
    IMPORT_OT_cam_ean,
    IMPORT_OT_ean,
    EXPORT_OT_emd,
    EXPORT_OT_nsk,
    EXPORT_OT_map,
    EXPORT_OT_esk,
    EXPORT_OT_ean,
    EXPORT_OT_cam_ean,
    CameraEANProperties,
    DATA_PT_xv2_camera_actions,
    XV2_OT_cam_create_actions,
    XV2_OT_cam_link_bone,
    XV2_OT_cam_rename_actions,
]


def _register_class(cls):
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        # If already registered (e.g., after a reload), unregister and try again.
        with contextlib.suppress(RuntimeError):
            bpy.utils.unregister_class(cls)
        bpy.utils.register_class(cls)


def _unregister_class(cls):
    with contextlib.suppress(RuntimeError):
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
