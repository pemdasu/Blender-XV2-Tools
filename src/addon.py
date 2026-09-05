import contextlib

import bpy
from bpy.props import CollectionProperty, IntProperty

from .addon_ops import (
    MENU_CLASSES,
    OPERATOR_CLASSES,
    menu_func,
    menu_func_export,
    register_icons,
    unregister_icons,
)
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
)
from .utils.blender_compat import check_blender_version

UI_CLASSES = [
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
    CameraEANProperties,
    DATA_PT_xv2_camera_actions,
    XV2_OT_cam_create_actions,
    XV2_OT_cam_link_bone,
    XV2_OT_cam_rename_actions,
]

CLASSES = [
    *UI_CLASSES,
    *MENU_CLASSES,
    *OPERATOR_CLASSES,
]


def _register_class(cls):
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        with contextlib.suppress(RuntimeError):
            bpy.utils.unregister_class(cls)
        bpy.utils.register_class(cls)


def _unregister_class(cls):
    with contextlib.suppress(RuntimeError):
        bpy.utils.unregister_class(cls)


def register():
    check_blender_version()
    register_icons()

    for cls in CLASSES:
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

    for cls in reversed(CLASSES):
        _unregister_class(cls)

    unregister_icons()


if __name__ == "__main__":
    register()
