import bpy
from bpy.props import (
    EnumProperty,
    FloatProperty,
    IntProperty,
)

from ..xv2.EMD import (
    EMD_TextureSamplerDef,
    set_sampler_custom_properties,
)


class EMDTextureSamplerPropertyGroup(bpy.types.PropertyGroup):
    flag0: IntProperty(name="Flag0", default=0, min=0, max=255)  # type: ignore
    texture_index: IntProperty(name="Texture Index", default=0, min=0, max=255)  # type: ignore
    address_mode_u: EnumProperty(  # type: ignore
        name="Address U",
        items=(
            ("0", "Wrap", ""),
            ("1", "Mirror", ""),
            ("2", "Clamp", ""),
        ),
        default="0",
    )
    address_mode_v: EnumProperty(  # type: ignore
        name="Address V",
        items=(
            ("0", "Wrap", ""),
            ("1", "Mirror", ""),
            ("2", "Clamp", ""),
        ),
        default="0",
    )
    filtering_min: EnumProperty(  # type: ignore
        name="Filter Min",
        items=(
            ("0", "None", ""),
            ("1", "Point", ""),
            ("2", "Linear", ""),
        ),
        default="2",
    )
    filtering_mag: EnumProperty(  # type: ignore
        name="Filter Mag",
        items=(
            ("0", "None", ""),
            ("1", "Point", ""),
            ("2", "Linear", ""),
        ),
        default="2",
    )
    scale_u: FloatProperty(name="Scale U", default=1.0)  # type: ignore
    scale_v: FloatProperty(name="Scale V", default=1.0)  # type: ignore


def sampler_defs_to_collection(container, samplers: list[EMD_TextureSamplerDef]):
    if not hasattr(container, "emd_texture_samplers"):
        return
    container.emd_texture_samplers.clear()
    for sampler in samplers:
        item = container.emd_texture_samplers.add()
        item.flag0 = int(sampler.flag0)
        item.texture_index = int(sampler.texture_index)
        item.address_mode_u = str(int(sampler.address_mode_u))
        item.address_mode_v = str(int(sampler.address_mode_v))
        item.filtering_min = str(int(sampler.filtering_min))
        item.filtering_mag = str(int(sampler.filtering_mag))
        item.scale_u = float(sampler.scale_u)
        item.scale_v = float(sampler.scale_v)


def collection_to_sampler_defs(container) -> list[EMD_TextureSamplerDef]:
    samplers: list[EMD_TextureSamplerDef] = []
    if not hasattr(container, "emd_texture_samplers"):
        return samplers
    for item in container.emd_texture_samplers:
        sampler = EMD_TextureSamplerDef()
        sampler.flag0 = int(item.flag0)
        sampler.texture_index = int(item.texture_index)
        sampler.address_mode_u = int(item.address_mode_u)
        sampler.address_mode_v = int(item.address_mode_v)
        sampler.filtering_min = int(item.filtering_min)
        sampler.filtering_mag = int(item.filtering_mag)
        sampler.scale_u = float(item.scale_u)
        sampler.scale_v = float(item.scale_v)
        samplers.append(sampler)
    return samplers


def refresh_sampler_custom_properties_from_collection(target):
    samplers = collection_to_sampler_defs(target)
    if samplers:
        set_sampler_custom_properties(target, samplers)


def sync_sampler_data(container, counterpart=None):
    if container is None:
        return
    samplers = collection_to_sampler_defs(container)
    if samplers:
        set_sampler_custom_properties(container, samplers)
        if counterpart is not None and hasattr(counterpart, "emd_texture_samplers"):
            sampler_defs_to_collection(counterpart, samplers)
            set_sampler_custom_properties(counterpart, samplers)


def get_sampler_container(context):
    obj = context.object
    if obj is None:
        return None, None
    mat = obj.active_material
    if mat is not None and hasattr(mat, "emd_texture_samplers"):
        return mat, obj
    if hasattr(obj, "emd_texture_samplers"):
        return obj, mat
    return None, None


class EMD_UL_texture_samplers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        sampler = item
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.label(
                text=f"{index}: Tex {sampler.texture_index}, "
                f"Scale U {sampler.scale_u}, Scale V {sampler.scale_v}"
            )
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=str(index))


class EMD_OT_texture_sampler_add(bpy.types.Operator):
    bl_idname = "emd_texture_sampler.add"
    bl_label = "Add Texture Sampler"
    bl_description = "Add a texture sampler entry to the active object"

    def execute(self, context):
        container, counterpart = get_sampler_container(context)
        if container is None:
            return {"CANCELLED"}
        new_item = container.emd_texture_samplers.add()
        new_item.texture_index = 0
        container.emd_texture_samplers_index = len(container.emd_texture_samplers) - 1
        sync_sampler_data(container, counterpart)
        return {"FINISHED"}


class EMD_OT_texture_sampler_remove(bpy.types.Operator):
    bl_idname = "emd_texture_sampler.remove"
    bl_label = "Remove Texture Sampler"
    bl_description = "Remove the selected texture sampler from the active object"

    @classmethod
    def poll(cls, context):
        container, _counterpart = get_sampler_container(context)
        return container and container.emd_texture_samplers

    def execute(self, context):
        container, counterpart = get_sampler_container(context)
        if container is None:
            return {"CANCELLED"}
        index = container.emd_texture_samplers_index
        if index < 0 or index >= len(container.emd_texture_samplers):
            return {"CANCELLED"}
        container.emd_texture_samplers.remove(index)
        container.emd_texture_samplers_index = max(0, index - 1)
        sync_sampler_data(container, counterpart)
        return {"FINISHED"}


class VIEW3D_PT_emd_texture_samplers(bpy.types.Panel):
    bl_label = "Xenoverse 2 EMD"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "EMD"

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None

    def draw(self, context):
        layout = self.layout
        container, counterpart = get_sampler_container(context)

        if container is None:
            layout.label(text="No sampler data available.")
            return
        is_material = isinstance(container, bpy.types.Material)
        header = "Material" if is_material else "Object"

        box_sampler = layout.box()
        box_sampler.label(text=f"{header} Texture Samplers", icon="TEXTURE")

        row = box_sampler.row()
        row.template_list(
            "EMD_UL_texture_samplers",
            "",
            container,
            "emd_texture_samplers",
            container,
            "emd_texture_samplers_index",
            rows=4,
        )

        col = row.column(align=True)
        col.operator(EMD_OT_texture_sampler_add.bl_idname, icon="ADD", text="")
        col.operator(EMD_OT_texture_sampler_remove.bl_idname, icon="REMOVE", text="")

        if container.emd_texture_samplers:
            active_idx = container.emd_texture_samplers_index
            if 0 <= active_idx < len(container.emd_texture_samplers):
                sampler = container.emd_texture_samplers[active_idx]
                inner = box_sampler.box()
                inner.prop(sampler, "texture_index")
                inner.prop(sampler, "flag0")
                inner.prop(sampler, "address_mode_u")
                inner.prop(sampler, "address_mode_v")
                inner.prop(sampler, "filtering_min")
                inner.prop(sampler, "filtering_mag")
                inner.prop(sampler, "scale_u")
                inner.prop(sampler, "scale_v")

        emm_box = layout.box()
        emm_box.label(text="EMM Parameters", icon="SHADING_RENDERED")
        if hasattr(container, "get"):
            emm_box.prop(container, '["emm_name"]', text="Name")
            emm_box.prop(container, '["emm_shader"]', text="Shader")
            displayed = False
            for key in sorted(container.keys()):
                if not key.startswith("emm_param_"):
                    continue
                label = key.replace("emm_param_", "", 1)
                emm_box.prop(container, f'["{key}"]', text=label)
                displayed = True
            if not displayed:
                emm_box.label(text="No EMM data.", icon="INFO")
        else:
            emm_box.label(text="No EMM data.", icon="INFO")


class PROPERTIES_PT_emd_texture_samplers(bpy.types.Panel):
    bl_label = "Xenoverse 2 EMD"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        mat = context.material or (context.object.active_material if context.object else None)
        return mat is not None

    def draw(self, context):
        layout = self.layout
        mat = context.material or (context.object.active_material if context.object else None)
        # obj = context.object
        container = mat if mat and hasattr(mat, "emd_texture_samplers") else None
        # counterpart = obj if obj and hasattr(obj, "emd_texture_samplers") else None

        if container is None:
            layout.label(text="No sampler data on material.")
            return
        box_sampler = layout.box()
        box_sampler.label(text="Material Texture Samplers", icon="TEXTURE")

        row = box_sampler.row()
        row.template_list(
            "EMD_UL_texture_samplers",
            "",
            container,
            "emd_texture_samplers",
            container,
            "emd_texture_samplers_index",
            rows=4,
        )

        col = row.column(align=True)
        col.operator(EMD_OT_texture_sampler_add.bl_idname, icon="ADD", text="")
        col.operator(EMD_OT_texture_sampler_remove.bl_idname, icon="REMOVE", text="")

        if container.emd_texture_samplers:
            active_idx = container.emd_texture_samplers_index
            if 0 <= active_idx < len(container.emd_texture_samplers):
                sampler = container.emd_texture_samplers[active_idx]
                inner = box_sampler.box()
                inner.prop(sampler, "texture_index")
                inner.prop(sampler, "flag0")
                inner.prop(sampler, "address_mode_u")
                inner.prop(sampler, "address_mode_v")
                inner.prop(sampler, "filtering_min")
                inner.prop(sampler, "filtering_mag")
                inner.prop(sampler, "scale_u")
                inner.prop(sampler, "scale_v")

        emm_box = layout.box()
        emm_box.label(text="EMM Parameters", icon="SHADING_RENDERED")
        if hasattr(container, "get"):
            emm_box.prop(container, '["emm_name"]', text="Name")
            emm_box.prop(container, '["emm_shader"]', text="Shader")
            displayed = False
            for key in sorted(container.keys()):
                if not key.startswith("emm_param_"):
                    continue
                label = key.replace("emm_param_", "", 1)
                emm_box.prop(container, f'["{key}"]', text=label)
                displayed = True
            if not displayed:
                emm_box.label(text="No EMM data.", icon="INFO")
        else:
            emm_box.label(text="No EMM data.", icon="INFO")


class EMD_OT_texture_sampler_sync_props(bpy.types.Operator):
    bl_idname = "emd_texture_sampler.sync_props"
    bl_label = "Refresh Custom Properties"
    bl_description = "Regenerate custom properties on the object/material from the list"

    def execute(self, context):
        container, counterpart = get_sampler_container(context)
        sync_sampler_data(container, counterpart)
        return {"FINISHED"}
