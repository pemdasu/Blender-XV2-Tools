import contextlib

import bmesh
import bpy

from .blender_warnings import warn_on_error
from .consts import MAX_BLENDER_VERSION_EXCLUSIVE, MIN_BLENDER_VERSION


def check_blender_version() -> None:
    version = bpy.app.version
    if MIN_BLENDER_VERSION <= version < MAX_BLENDER_VERSION_EXCLUSIVE:
        return

    supported = "4.0.0 through 5.0.x"
    current = ".".join(str(part) for part in version)
    raise RuntimeError(
        f"Blender XV2 Tools supports Blender {supported}. "
        f"You are running Blender {current}. Install Blender 4.0 or newer, up to Blender 5.0.x."
    )


def iter_action_fcurves(action):
    if action is None:
        return

    fcurves = getattr(action, "fcurves", None)
    if fcurves is not None:
        yield from fcurves
        return

    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                yield from getattr(channelbag, "fcurves", ())


def find_action_fcurve(action, data_path: str, index: int = 0):
    if action is None:
        return None

    fcurves = getattr(action, "fcurves", None)
    if fcurves is not None:
        return fcurves.find(data_path, index=index)

    for fcurve in iter_action_fcurves(action):
        if fcurve.data_path == data_path and fcurve.array_index == index:
            return fcurve
    return None


def ensure_action_fcurve(action, data_block, data_path: str, index: int = 0):
    fcurve = find_action_fcurve(action, data_path, index=index)
    if fcurve is not None:
        return fcurve

    ensure_for_data_block = getattr(action, "fcurve_ensure_for_datablock", None)
    if ensure_for_data_block is not None:
        return ensure_for_data_block(data_block, data_path, index=index)

    return action.fcurves.new(data_path, index=index)


def calc_split_normals(mesh: bpy.types.Mesh) -> None:
    calc_normals_split = getattr(mesh, "calc_normals_split", None)
    if calc_normals_split is not None:
        with warn_on_error(
            "Could not calculate split normals",
            RuntimeError,
            AttributeError,
        ):
            calc_normals_split()


def set_custom_split_normals(mesh: bpy.types.Mesh, loop_normals) -> bool:
    create_normals_split = getattr(mesh, "create_normals_split", None)
    if create_normals_split is not None:
        with warn_on_error(
            "Could not create custom split normals",
            RuntimeError,
            AttributeError,
        ):
            create_normals_split()

    set_normals = getattr(mesh, "normals_split_custom_set", None)
    if set_normals is None:
        return False

    try:
        set_normals(loop_normals)
    except RuntimeError:
        clear_custom_split_normals(mesh)
        return False

    validate = getattr(mesh, "validate", None)
    if validate is not None:
        with warn_on_error(
            "Could not validate custom split normals",
            RuntimeError,
            AttributeError,
        ):
            validate(clean_customdata=False)
    return True


def clear_custom_split_normals(mesh: bpy.types.Mesh) -> None:
    free_normals_split = getattr(mesh, "free_normals_split", None)
    if free_normals_split is not None:
        with warn_on_error(
            "Could not clear custom split normals",
            RuntimeError,
            AttributeError,
        ):
            free_normals_split()


def _run_mesh_operator(name: str, *kwargs_options: dict) -> bool:
    op = getattr(bpy.ops.mesh, name, None)
    if op is not None:
        for kwargs in kwargs_options:
            with contextlib.suppress(AttributeError, TypeError, RuntimeError):
                op(**kwargs)
                return True
    return False


def _merge_selected_by_distance_bmesh(threshold: float) -> None:
    obj = bpy.context.object
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Merge by distance needs an active mesh object.")

    mesh = obj.data
    if bpy.context.mode == "EDIT_MESH":
        edit_mesh = bmesh.from_edit_mesh(mesh)
        bmesh.ops.remove_doubles(edit_mesh, verts=edit_mesh.verts, dist=threshold)
        bmesh.update_edit_mesh(mesh)
        return

    mesh_bmesh = bmesh.new()
    try:
        mesh_bmesh.from_mesh(mesh)
        bmesh.ops.remove_doubles(mesh_bmesh, verts=mesh_bmesh.verts, dist=threshold)
        mesh_bmesh.to_mesh(mesh)
        mesh.update()
    finally:
        mesh_bmesh.free()


def merge_selected_by_distance(threshold: float, use_sharp_edge_from_normals: bool = False) -> None:
    kwargs_with_normals = {
        "threshold": threshold,
        "use_sharp_edge_from_normals": use_sharp_edge_from_normals,
    }
    if _run_mesh_operator(
        "merge_by_distance",
        kwargs_with_normals,
        {"threshold": threshold},
        {"distance": threshold, "use_sharp_edge_from_normals": use_sharp_edge_from_normals},
        {"distance": threshold},
    ):
        return

    if _run_mesh_operator(
        "remove_doubles",
        kwargs_with_normals,
        {"threshold": threshold},
        {"distance": threshold, "use_sharp_edge_from_normals": use_sharp_edge_from_normals},
        {"distance": threshold},
    ):
        return

    _merge_selected_by_distance_bmesh(threshold)
