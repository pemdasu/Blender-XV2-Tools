from pathlib import Path

import bpy
import bpy.utils.previews
from bpy.types import Menu

from .ean import (
    CLASSES as EAN_CLASSES,
)
from .ean import (
    EXPORT_OT_cam_ean,
    EXPORT_OT_ean,
    IMPORT_OT_cam_ean,
    IMPORT_OT_ean,
)
from .ema import CLASSES as EMA_CLASSES
from .ema import EXPORT_OT_ema, IMPORT_OT_ema
from .emd import CLASSES as EMD_CLASSES
from .emd import EXPORT_OT_emd, IMPORT_OT_emd
from .emo import CLASSES as EMO_CLASSES
from .emo import EXPORT_OT_emo, IMPORT_OT_emo
from .esk import CLASSES as ESK_CLASSES
from .esk import EXPORT_OT_esk, IMPORT_OT_esk
from .fmp import CLASSES as FMP_CLASSES
from .fmp import EXPORT_OT_map, IMPORT_OT_map
from .nsk import CLASSES as NSK_CLASSES
from .nsk import EXPORT_OT_nsk, IMPORT_OT_nsk

_custom_icons = None
_xv2_assets_icon_id = 0
_entry_icon_ids: dict[str, int] = {}
_icon_dir = Path(__file__).resolve().parent.parent / "icons"
_xv2_assets_icon_path = _icon_dir / "DBXV2.png"
_entry_icon_paths = {
    "emd": _icon_dir / "icon_emd.png",
    "esk": _icon_dir / "icon_esk.png",
    "ean": _icon_dir / "icon_ean.png",
    "ema": _icon_dir / "icon_ean.png",
    "cam": _icon_dir / "icon_cam.png",
    "emo": _icon_dir / "icon_emo.png",
    "nsk": _icon_dir / "icon_nsk.png",
    "map": _icon_dir / "icon_map.png",
}


class XV2_MT_import_assets(Menu):
    bl_idname = "TOPBAR_MT_xv2_import_assets"
    bl_label = "Dragon Ball XV2 Assets"

    def draw(self, _context):
        layout = self.layout
        layout.operator(
            IMPORT_OT_emd.bl_idname,
            text="Dragon Ball XV2 EMD (.emd)",
            icon_value=_entry_icon_ids["emd"],
        )
        layout.operator(
            IMPORT_OT_esk.bl_idname,
            text="Dragon Ball XV2 ESK (.esk)",
            icon_value=_entry_icon_ids["esk"],
        )
        layout.operator(
            IMPORT_OT_ean.bl_idname,
            text="Dragon Ball XV2 EAN (.ean)",
            icon_value=_entry_icon_ids["ean"],
        )
        layout.operator(
            IMPORT_OT_cam_ean.bl_idname,
            text="Dragon Ball XV2 Camera EAN (.cam.ean)",
            icon_value=_entry_icon_ids["cam"],
        )
        layout.separator()
        layout.operator(
            IMPORT_OT_emo.bl_idname,
            text="Dragon Ball XV2 EMO (.emo)",
            icon_value=_entry_icon_ids["emo"],
        )
        layout.operator(
            IMPORT_OT_ema.bl_idname,
            text="Dragon Ball XV2 EMA (.ema)",
            icon_value=_entry_icon_ids["ema"],
        )
        layout.separator()
        layout.operator(
            IMPORT_OT_nsk.bl_idname,
            text="Dragon Ball XV2 NSK (.nsk)",
            icon_value=_entry_icon_ids["nsk"],
        )
        layout.operator(
            IMPORT_OT_map.bl_idname,
            text="Dragon Ball XV2 MAP (.map)",
            icon_value=_entry_icon_ids["map"],
        )


class XV2_MT_export_assets(Menu):
    bl_idname = "TOPBAR_MT_xv2_export_assets"
    bl_label = "Dragon Ball XV2 Assets"

    def draw(self, _context):
        layout = self.layout
        layout.operator(
            EXPORT_OT_emd.bl_idname,
            text="Dragon Ball XV2 EMD (.emd)",
            icon_value=_entry_icon_ids["emd"],
        )
        layout.operator(
            EXPORT_OT_esk.bl_idname,
            text="Dragon Ball XV2 ESK (.esk)",
            icon_value=_entry_icon_ids["esk"],
        )
        layout.operator(
            EXPORT_OT_ean.bl_idname,
            text="Dragon Ball XV2 EAN (.ean)",
            icon_value=_entry_icon_ids["ean"],
        )
        layout.operator(
            EXPORT_OT_cam_ean.bl_idname,
            text="Dragon Ball XV2 Camera EAN (.cam.ean)",
            icon_value=_entry_icon_ids["cam"],
        )
        layout.separator()
        layout.operator(
            EXPORT_OT_emo.bl_idname,
            text="Dragon Ball XV2 EMO (.emo)",
            icon_value=_entry_icon_ids["emo"],
        )
        layout.operator(
            EXPORT_OT_ema.bl_idname,
            text="Dragon Ball XV2 EMA (.ema)",
            icon_value=_entry_icon_ids["ema"],
        )
        layout.separator()
        layout.operator(
            EXPORT_OT_nsk.bl_idname,
            text="Dragon Ball XV2 NSK (.nsk)",
            icon_value=_entry_icon_ids["nsk"],
        )
        layout.operator(
            EXPORT_OT_map.bl_idname,
            text="Dragon Ball XV2 MAP (.map)",
            icon_value=_entry_icon_ids["map"],
        )


def menu_func(self, _context):
    self.layout.menu(
        XV2_MT_import_assets.bl_idname,
        text="Dragon Ball XV2 Assets",
        icon_value=_xv2_assets_icon_id,
    )


def menu_func_export(self, _context):
    self.layout.menu(
        XV2_MT_export_assets.bl_idname,
        text="Dragon Ball XV2 Assets",
        icon_value=_xv2_assets_icon_id,
    )


def register_icons():
    global _custom_icons, _xv2_assets_icon_id, _entry_icon_ids

    if not _xv2_assets_icon_path.is_file():
        raise FileNotFoundError(f"Missing required icon file: {_xv2_assets_icon_path}")
    for icon_key, icon_path in _entry_icon_paths.items():
        if not icon_path.is_file():
            raise FileNotFoundError(f"Missing required icon file: {icon_path} ({icon_key})")

    _custom_icons = bpy.utils.previews.new()
    _custom_icons.load("xv2_assets", str(_xv2_assets_icon_path), "IMAGE")
    _xv2_assets_icon_id = int(_custom_icons["xv2_assets"].icon_id)
    _entry_icon_ids = {}
    for icon_key, icon_path in _entry_icon_paths.items():
        icon_name = f"xv2_{icon_key}"
        _custom_icons.load(icon_name, str(icon_path), "IMAGE")
        _entry_icon_ids[icon_key] = int(_custom_icons[icon_name].icon_id)


def unregister_icons():
    global _custom_icons, _xv2_assets_icon_id, _entry_icon_ids

    if _custom_icons is not None:
        bpy.utils.previews.remove(_custom_icons)
    _custom_icons = None
    _xv2_assets_icon_id = 0
    _entry_icon_ids = {}


OPERATOR_CLASSES = [
    *EMD_CLASSES,
    *EMO_CLASSES,
    *NSK_CLASSES,
    *FMP_CLASSES,
    *ESK_CLASSES,
    *EAN_CLASSES,
    *EMA_CLASSES,
]

MENU_CLASSES = [
    XV2_MT_import_assets,
    XV2_MT_export_assets,
]
