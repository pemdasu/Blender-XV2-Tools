import os
from collections.abc import Iterator

import bpy
from bpy.props import BoolProperty, FloatProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..utils.blender_warnings import warn_on_error
from ..xv2.FMP.exporter import export_map
from ..xv2.FMP.importer import import_map_in_steps


class IMPORT_OT_map(Operator, ImportHelper):
    bl_idname = "import_scene.xv2_map"
    bl_label = "Import MAP (Xenoverse 2)"

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
    reuse_materials: BoolProperty(  # type: ignore
        name="Reuse Materials",
        description="Reuse existing materials by name when the shader template matches",
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
        default=False,
    )
    preserve_bone_axes: BoolProperty(  # type: ignore
        name="Preserve Bone Axes",
        description=(
            "Build armature bones from source local axes. Helps mirrored chains keep matching"
        ),
        default=False,
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
        layout.prop(self, "reuse_materials")
        if self.auto_merge_by_distance:
            layout.prop(self, "merge_distance")
        layout.prop(self, "import_colliders")
        if self.import_colliders:
            layout.prop(self, "import_collision_meshes")
        layout.prop(self, "preserve_bone_axes")
        layout.prop(self, "use_collection_instances")

    def _cleanup_modal(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        context.window.cursor_set("DEFAULT")
        with warn_on_error(
            "Could not clear the MAP import status text",
            AttributeError,
            RuntimeError,
        ):
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
            reuse_materials=self.reuse_materials,
            preserve_bone_axes=self.preserve_bone_axes,
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
            with warn_on_error(
                "Could not update the MAP import status text",
                AttributeError,
                RuntimeError,
            ):
                context.workspace.status_text_set(message)
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
        with warn_on_error(
            "Could not set the MAP import status text",
            AttributeError,
            RuntimeError,
        ):
            context.workspace.status_text_set("[MAP] Starting import...")
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}


# ---------------------------------------------------------------------------
# ESK Import (ESK)
# ---------------------------------------------------------------------------


class EXPORT_OT_map(Operator, ExportHelper):
    bl_idname = "export_scene.xv2_map"
    bl_label = "Export MAP (Xenoverse 2)"

    filename_ext = ".map"
    filter_glob: StringProperty(default="*.map", options={"HIDDEN"})  # type: ignore
    export_collision_meshes: BoolProperty(  # type: ignore
        name="Export collision meshes (EXPERIMENTAL)",
        description=(
            "Write edited collider mesh vertices/triangles back into source MAP collision data "
            "when possible"
        ),
        default=False,
    )
    export_linked_nsk: BoolProperty(  # type: ignore
        name="Export linked NSK files (EXPERIMENTAL)",
        description=(
            "Also export NSK files referenced by MAP entities using their MAP relative paths"
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


CLASSES = [
    IMPORT_OT_map,
    EXPORT_OT_map,
]
