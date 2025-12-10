from . import addon

bl_info = {
    "name": "DRAGON BALL XENOVERSE 2 Tools",
    "author": "Pemdasu",
    "version": (1, 0, 3),
    "blender": (4, 0, 2),
    "location": "File > Import/Export",
    "description": "Import and export Dragon Ball Xenoverse 2 files",
    "category": "Import-Export",
}


def register():
    addon.register()


def unregister():
    addon.unregister()


if __name__ == "__main__":
    register()
