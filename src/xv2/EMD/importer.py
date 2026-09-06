import contextlib
import math
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path

import bpy
import mathutils

from ...ui import sampler_defs_to_collection
from ...utils import remove_unused_vertex_groups
from ...utils.blender_compat import (
    clear_custom_split_normals,
    merge_selected_by_distance,
    set_custom_split_normals,
)
from ..consts import AUTO_SMOOTH_ANGLE_DEGREES
from ..EMB import (
    _extract_dyt_lines,
    emb_stem_from_path,
    load_emb_image,
    locate_emb_files,
    read_emb,
)
from ..EMM import locate_emm, parse_emm
from ..ESK import ESK_File, build_armature, parse_esk
from ..NSK.importer import (
    apply_nsk_placeholder_material as _apply_nsk_placeholder_material,
)
from ..NSK.importer import (
    emd_has_any_triangle_bones as _emd_has_any_triangle_bones,
)
from ..NSK.importer import (
    find_armature_bone as _find_armature_bone,
)
from ..NSK.importer import (
    get_esk_world_matrix_by_bone_name as _get_esk_world_matrix_by_bone_name,
)
from ..NSK.importer import (
    resolve_source_behavior as _resolve_source_behavior,
)
from ..NSK.importer import (
    submesh_has_blend_weights as _submesh_has_blend_weights,
)
from .EMD import EMD_File, EMD_Submesh, parse_emd, set_sampler_custom_properties

_BIND_SCALE_EPS = 1e-3


def _grow_tiny_mesh(obj: bpy.types.Object, arm_obj: bpy.types.Object, esk: ESK_File) -> None:
    if arm_obj is None or arm_obj.data is None:
        return

    def bone_scale(index: int) -> mathutils.Vector:
        bone = esk.bones[index]
        if bone.matrix is None:
            return mathutils.Vector((1.0, 1.0, 1.0))
        return bone.matrix.decompose()[2]

    def is_scaled(scale: mathutils.Vector) -> bool:
        return (
            abs(scale.x - 1.0) > _BIND_SCALE_EPS
            or abs(scale.y - 1.0) > _BIND_SCALE_EPS
            or abs(scale.z - 1.0) > _BIND_SCALE_EPS
        )

    total_cache: dict[int, mathutils.Vector] = {}

    def total_scale(index: int) -> mathutils.Vector:
        if index in total_cache:
            return total_cache[index]
        scale = bone_scale(index)
        parent = esk.bones[index].parent_index
        if 0 <= parent < len(esk.bones) and esk.bones[parent] is not esk.bones[index]:
            parent_scale = total_scale(parent)
            scale = mathutils.Vector(
                (scale.x * parent_scale.x, scale.y * parent_scale.y, scale.z * parent_scale.z)
            )
        total_cache[index] = scale
        return scale

    def scaled_parent(index: int) -> int | None:
        current = index
        while current >= 0:
            if is_scaled(bone_scale(current)):
                return current
            parent = esk.bones[current].parent_index
            if not (0 <= parent < len(esk.bones)) or esk.bones[parent] is esk.bones[current]:
                return None
            current = parent
        return None

    pivots: dict[str, tuple[mathutils.Vector, float]] = {}
    for index, bone in enumerate(esk.bones):
        scale = total_scale(index)
        bind = min(scale.x, scale.y, scale.z)
        if not is_scaled(scale) or bind <= 0.0:
            continue
        parent = scaled_parent(index)
        if parent is None:
            continue
        parent_bone = arm_obj.data.bones.get(esk.bones[parent].name)
        if parent_bone is None:
            continue
        pivots[bone.name] = (parent_bone.head_local.copy(), bind)
    if not pivots:
        return

    group_pivots: dict[int, tuple[mathutils.Vector, float]] = {}
    for group in obj.vertex_groups:
        info = pivots.get(group.name)
        if info is not None:
            group_pivots[group.index] = info
    if not group_pivots:
        return

    mesh = obj.data
    used: dict[str, tuple[mathutils.Vector, float]] = {}
    for vertex in mesh.vertices:
        top_weight = 0.0
        top: tuple[mathutils.Vector, float] | None = None
        top_name = ""
        for element in vertex.groups:
            info = group_pivots.get(element.group)
            if info is None or element.weight <= top_weight:
                continue
            top_weight = element.weight
            top = info
            top_name = obj.vertex_groups[element.group].name
        if top is None:
            continue
        pivot, bind = top
        vertex.co = pivot + (vertex.co - pivot) / bind
        used[top_name] = top

    if not used:
        return

    names = sorted(used)
    flat: list[float] = []
    for name in names:
        pivot, bind = used[name]
        flat.extend((pivot.x, pivot.y, pivot.z, bind))
    obj["xv2_bind_bones"] = names
    obj["xv2_bind_data"] = flat


def bind_weights(
    obj: bpy.types.Object,
    sub: EMD_Submesh,
    arm_obj: bpy.types.Object,
    esk: ESK_File,
):
    vgroups_by_name: dict[str, bpy.types.VertexGroup] = {}
    for bone in esk.bones[1:]:
        if not bone.name:
            continue
        vertex_group = obj.vertex_groups.get(bone.name)
        if vertex_group is None:
            vertex_group = obj.vertex_groups.new(name=bone.name)
        vgroups_by_name[bone.name] = vertex_group

    has_palettes = bool(getattr(sub, "triangle_groups", None)) and any(
        triangle_group.bone_names for triangle_group in sub.triangle_groups
    )

    if has_palettes:
        for triangle_group in sub.triangle_groups:
            if not triangle_group.bone_names:
                continue

            palette_to_vertex_group: list[bpy.types.VertexGroup | None] = [
                vgroups_by_name.get(bname) for bname in triangle_group.bone_names
            ]

            for vertex_index in triangle_group.indices:
                vertex = sub.vertices[vertex_index]

                total_weight = sum(vertex.bone_weights)
                if total_weight > 1e-6:
                    weights = [weight_value / total_weight for weight_value in vertex.bone_weights]
                else:
                    weights = list(vertex.bone_weights)

                for weight_index in range(4):
                    weight_value = weights[weight_index]
                    if weight_value <= 0.0:
                        continue

                    palette_index = vertex.bone_ids[weight_index]
                    if 0 <= palette_index < len(palette_to_vertex_group):
                        vertex_group = palette_to_vertex_group[palette_index]
                        if vertex_group is not None:
                            vertex_group.add(
                                [vertex_index],
                                float(weight_value),
                                "REPLACE",
                            )
    else:
        for vertex_index, vertex in enumerate(sub.vertices):
            total_weight = sum(vertex.bone_weights)
            if total_weight > 1e-6:
                weights = [weight_value / total_weight for weight_value in vertex.bone_weights]
            else:
                weights = list(vertex.bone_weights)

            for weight_index in range(4):
                weight_value = weights[weight_index]
                if weight_value <= 0.0:
                    continue

                bone_index = vertex.bone_ids[weight_index]
                if not (0 <= bone_index < len(esk.bones)):
                    continue

                bone_name = esk.bones[bone_index].name
                if not bone_name:
                    continue

                vertex_group = vgroups_by_name.get(bone_name)
                if vertex_group is not None:
                    vertex_group.add(
                        [vertex_index],
                        float(weight_value),
                        "REPLACE",
                    )

    modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = arm_obj
    modifier.show_in_editmode = True
    modifier.show_on_cage = True


@cache
def _get_shader_template(template_name: str = "shader") -> bpy.types.Material | None:
    # importer.py -> src/xv2/EMD -> parents[2] == src; shader in src/shader/shader.blend
    blend_path = Path(__file__).resolve().parents[2] / "shader" / "shader.blend"
    if not blend_path.is_file():
        return None
    try:
        loaded = []
        with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
            if template_name in data_from.materials:
                data_to.materials = [template_name]
                loaded = list(data_to.materials)
        if loaded:
            mat = loaded[0]
            if isinstance(mat, str):
                mat = bpy.data.materials.get(mat)
            return mat
    except (OSError, RuntimeError, ReferenceError, ValueError) as exc:
        print("Failed to load shader template:", exc)
    return None


def _make_shader_material(
    name: str,
    template_name: str | None = None,
    reuse_materials: bool = True,
) -> bpy.types.Material:
    material_name = name or "EMD_Material"
    if not template_name:
        template_name = "eye_shader" if name and name.lower().startswith("eye_") else "shader"

    # Reuse existing material when enabled to avoid Blender auto-suffixes.
    existing = bpy.data.materials.get(material_name)
    if (
        reuse_materials
        and existing is not None
        and str(existing.get("_xv2_shader_template_name", "")) == template_name
    ):
        return existing

    template = _get_shader_template(template_name)
    # If the cached template was removed, refresh the cache and try again.
    try:
        missing = (template is None) or (template.name not in bpy.data.materials)
    except ReferenceError:
        missing = True
    if missing:
        _get_shader_template.cache_clear()
        template = _get_shader_template(template_name)
    if template:
        try:
            mat = template.copy()
        except ReferenceError:
            _get_shader_template.cache_clear()
            template = _get_shader_template(template_name)
            mat = template.copy() if template else None
        if mat:
            mat.name = material_name
            mat.use_fake_user = False
            mat["_xv2_shader_template"] = True
            mat["_xv2_shader_template_name"] = template_name
            return mat
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    material["_xv2_shader_template_name"] = template_name
    if material.node_tree:
        material.node_tree.nodes.clear()
        material.node_tree.links.clear()
    return material


def _resolve_shader_template(
    submesh_name: str,
    source_format: str = "EMD",
    emm_shader: str | None = None,
    force_shader_template: str | None = None,
) -> str:
    shader_name = (emm_shader or "").strip().upper()
    if shader_name.startswith("TOON_UNIF") and shader_name.endswith("_OWR"):
        return "unif_env_owr_shader" if "UNIF_ENV" in shader_name else "owr_shader"
    if shader_name == "TOON_UNIF_SCROLL":
        return "scroll_shader"
    if force_shader_template:
        return force_shader_template
    if "UNIF_ENV" in shader_name:
        return "unif_env_shader"
    format_tag = (source_format or "EMD").strip().upper()
    # Hook for format-specific shader defaults.
    if format_tag == "NSK":
        return "shader"
    return "eye_shader" if submesh_name and submesh_name.lower().startswith("eye_") else "shader"


def bind_weights_built(
    obj: bpy.types.Object,
    sub: EMD_Submesh,
    arm_obj: bpy.types.Object,
    esk: ESK_File,
    built_source_indices: list[int],
    built_palette_groups: list[object | None],
):
    vgroups_by_name: dict[str, bpy.types.VertexGroup] = {}
    for bone in esk.bones[1:]:
        if not bone.name:
            continue
        vg = obj.vertex_groups.get(bone.name) or obj.vertex_groups.new(name=bone.name)
        vgroups_by_name[bone.name] = vg

    for v_idx, src_idx in enumerate(built_source_indices):
        if src_idx < 0 or src_idx >= len(sub.vertices):
            continue
        vertex = sub.vertices[src_idx]

        total_weight = sum(vertex.bone_weights)
        weights = (
            [w / total_weight for w in vertex.bone_weights]
            if total_weight > 1e-6
            else list(vertex.bone_weights)
        )

        tri_group = built_palette_groups[v_idx]
        palette_names = tri_group.bone_names if tri_group else None

        for w_idx in range(4):
            weight_value = weights[w_idx]
            if weight_value <= 0.0:
                continue

            if palette_names:
                palette_index = vertex.bone_ids[w_idx]
                if not (0 <= palette_index < len(palette_names)):
                    continue
                bone_name = palette_names[palette_index]
            else:
                bone_index = vertex.bone_ids[w_idx]
                if not (0 <= bone_index < len(esk.bones)):
                    continue
                bone_name = esk.bones[bone_index].name

            if not bone_name:
                continue
            vg = vgroups_by_name.get(bone_name)
            if vg is None:
                vg = obj.vertex_groups.new(name=bone_name)
                vgroups_by_name[bone_name] = vg
            vg.add([v_idx], float(weight_value), "REPLACE")

    modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = arm_obj
    modifier.show_in_editmode = True
    modifier.show_on_cage = True


def create_material(
    submesh_name: str,
    source_format: str = "EMD",
    emm_shader: str | None = None,
    force_shader_template: str | None = None,
    reuse_materials: bool = True,
) -> bpy.types.Material:
    template_name = _resolve_shader_template(
        submesh_name=submesh_name,
        source_format=source_format,
        emm_shader=emm_shader,
        force_shader_template=force_shader_template,
    )
    return _make_shader_material(
        submesh_name,
        template_name=template_name,
        reuse_materials=reuse_materials,
    )


def _image_from_sampler(
    sampler_defs,
    sampler_index: int,
    emb_main,
    warn: Callable[[str], None] | None = None,
) -> bpy.types.Image | None:
    if not sampler_defs or emb_main is None:
        return None
    if not (0 <= sampler_index < len(sampler_defs)):
        return None
    tex_index = int(sampler_defs[sampler_index].texture_index)
    if tex_index < 0 or tex_index >= len(emb_main.entries):
        return None
    entry = emb_main.entries[tex_index]
    entry_name = (entry.name or "").lower()
    if entry_name.endswith(".dyt") or ".dyt." in entry_name:
        if warn:
            warn(
                f"Skipping DYT source texture '{entry.name or f'DATA{entry.index:03d}.dds'}' "
                f"from '{os.path.basename(emb_main.path)}'."
            )
        return None
    return load_emb_image(
        entry,
        emb_main.path,
        warn=warn,
    )


def _configure_uv_scroll(mat, params, sampler_defs):
    mapping = mat.node_tree.nodes["XV2_SCROLL_UV"]
    scale_u = sampler_defs[0].scale_u if sampler_defs else 1.0
    scale_v = sampler_defs[0].scale_v if sampler_defs else 1.0
    mapping.inputs["Scale"].default_value = (scale_u, scale_v, 1.0)
    # EMD import flips V, so Blender uses a positive V scroll and a tile offset.
    for index, name in enumerate(("TexScrl0U", "TexScrl0V")):
        speed = float(params.get(name, 0.0))
        driver = mapping.inputs["Location"].driver_add("default_value", index).driver
        driver.type = "SCRIPTED"
        while driver.variables:
            driver.variables.remove(driver.variables[0])
        for variable_name, path in (("fps", "render.fps"), ("fps_base", "render.fps_base")):
            variable = driver.variables.new()
            variable.name = variable_name
            variable.targets[0].id_type = "SCENE"
            variable.targets[0].id = bpy.context.scene
            variable.targets[0].data_path = path
        offset = 0.0 if index == 0 else 1.0 - scale_v
        driver.expression = f"{offset!r} + frame * fps_base / fps * {speed!r}"


def _apply_shader_material(
    mat: bpy.types.Material,
    sampler_defs,
    emb_main,
    emb_dyt,
    emm_info,
    dyt_entry_index: int = 0,
    warn: Callable[[str], None] | None = None,
) -> None:
    if not mat or not mat.node_tree:
        return

    def _remove_image(image: bpy.types.Image | None) -> None:
        if image is None:
            return
        with contextlib.suppress(RuntimeError):
            bpy.data.images.remove(image, do_unlink=True)

    nodes = mat.node_tree.nodes

    # Apply sampler textures
    emb_node = nodes.get("XV2_EMB_SAMPLER")
    dual_node = nodes.get("XV2_DUAL_EMB_SAMPLER")
    msk_node = nodes.get("XV2_MSK_EMB_SAMPLER")
    dual_toggle = nodes.get("XV2_DUAL_EMB_TOGGLE")
    msk_toggle = nodes.get("XV2_MSK_EMB_TOGGLE")
    shader_name = (getattr(emm_info, "shader", "") or "").upper()
    params = {param.name: param.value for param in emm_info.params} if emm_info else {}
    if shader_name == "TOON_UNIF_SCROLL":
        _configure_uv_scroll(mat, params, sampler_defs)
    alpha_node = nodes.get("XV2_ALPHA_BLEND")
    if alpha_node:
        alpha_enabled = int(params.get("AlphaBlend", 0)) == 1
        blend_type = int(params.get("AlphaBlendType", 0))
        supported = shader_name.startswith("TOON_UNIF") and blend_type == 0
        alpha_node.inputs["Enabled"].default_value = float(alpha_enabled and supported)
        if alpha_enabled and not supported and warn:
            warn(
                f"Alpha blending for '{shader_name}' with blend type {blend_type} is not supported."
            )
        if bpy.app.version >= (4, 2, 0):
            mat.surface_render_method = "BLENDED" if alpha_enabled and supported else "DITHERED"
        else:
            mat.blend_method = "BLEND" if alpha_enabled and supported else "OPAQUE"
    use_unif_env = "UNIF_ENV" in shader_name
    use_toon_uniffx = "TOON_UNIFFX" in shader_name
    toon_uniffx_dyt_entry = None

    main_sampler_index = 0
    if use_toon_uniffx and sampler_defs:
        if len(sampler_defs) > 1:
            # TOON_UNIFfx usually keeps DYT in sampler slot 0 and diffuse in slot 1.
            main_sampler_index = 1
        if emb_main is not None:
            dyt_tex_index = int(sampler_defs[0].texture_index)
            if 0 <= dyt_tex_index < len(emb_main.entries):
                toon_uniffx_dyt_entry = emb_main.entries[dyt_tex_index]
            elif warn:
                warn(
                    f"TOON_UNIFfx DYT sampler index {dyt_tex_index} is out of range for "
                    f"'{os.path.basename(emb_main.path)}'."
                )

    main_img = _image_from_sampler(sampler_defs, main_sampler_index, emb_main, warn=warn)
    dual_img = _image_from_sampler(sampler_defs, 2, emb_main, warn=warn)

    def _configure_image(tex_node: bpy.types.Node, img: bpy.types.Image, is_dyt: bool) -> None:
        tex_node.image = img
        try:
            tex_node.interpolation = "Closest" if is_dyt else "Linear"
            tex_node.projection = "FLAT"
            tex_node.extension = "EXTEND" if is_dyt else "REPEAT"
            if not is_dyt and img and hasattr(img, "colorspace_settings"):
                img.colorspace_settings.name = "Non-Color" if not use_unif_env else "sRGB"
        except (AttributeError, TypeError, ValueError):
            pass

    if emb_node and main_img:
        _configure_image(emb_node, main_img, is_dyt=False)
        if use_unif_env:
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                emb_node.extension = "EXTEND"
    use_dual = dual_img is not None and emm_info and "d2_" in (emm_info.shader or "")
    if dual_node and dual_img and use_dual:
        _configure_image(dual_node, dual_img, is_dyt=False)
    if dual_toggle and hasattr(dual_toggle, "inputs"):
        with contextlib.suppress(TypeError, ValueError, AttributeError, KeyError):
            dual_toggle.inputs[0].default_value = 1.0 if use_dual else 0.0
    use_msk = dual_img is not None and emm_info and "MSK" in (emm_info.shader or "")
    if msk_node and dual_img and use_msk:
        _configure_image(msk_node, dual_img, is_dyt=False)
    if msk_toggle and hasattr(msk_toggle, "inputs"):
        with contextlib.suppress(TypeError, ValueError, AttributeError, KeyError):
            msk_toggle.inputs[0].default_value = 1.0 if use_msk else 0.0

    # Apply DYT lines based on MatScale1X (default 0)
    mat_scale = 0
    if emm_info:
        for param in emm_info.params:
            if param.name == "MatScale1X":
                with contextlib.suppress(TypeError, ValueError):
                    mat_scale = int(round(float(param.value)))
                break
    # Fallback: use custom prop on material if present
    if mat_scale == 0 and "emm_param_MatScale1X" in mat:
        with contextlib.suppress(TypeError, ValueError):
            mat_scale = int(round(float(mat.get("emm_param_MatScale1X", 0))))

    def _apply_dyt_entry(dyt_entry, source_emb_path: str) -> None:
        base_name = os.path.splitext(dyt_entry.name or f"DATA{dyt_entry.index:03d}.dds")[0]
        dyt_image = load_emb_image(
            dyt_entry,
            source_emb_path,
            base_override=f"{base_name}.dyt.dds",
            warn=warn,
        )
        if dyt_image is None:
            return

        block_idx = max(0, mat_scale)
        lines = _extract_dyt_lines(
            dyt_image,
            f"{emb_stem_from_path(source_emb_path)}_toon",
            block_index=block_idx,
            source_token=str(dyt_image.get("emb_source_token", "")),
        )
        primary = lines.get("p") or next(iter(lines.values()), None)
        rim = lines.get("r")
        spec = lines.get("s")
        secondary = lines.get("d")

        assign_map = {
            "XV2_DYT_MAIN": primary,
            "XV2_DYT_RIM": rim,
            "XV2_DYT_SPEC": spec,
            "XV2_DYT_DUAL": secondary,
            "XV2_DYT_OWR": secondary,
        }
        for node_name, img_obj in assign_map.items():
            node = nodes.get(node_name)
            if node and img_obj:
                _configure_image(node, img_obj, is_dyt=True)

        # Keep only extracted DYT line images in the blend file.
        _remove_image(dyt_image)

    if use_toon_uniffx and toon_uniffx_dyt_entry is not None and emb_main is not None:
        _apply_dyt_entry(toon_uniffx_dyt_entry, emb_main.path)
    elif emb_dyt:
        dyt_entries = emb_dyt.entries or []
        requested_idx = max(0, int(dyt_entry_index))
        selected_idx = requested_idx
        emb_name = os.path.basename(emb_dyt.path) or "dyt.emb"

        if selected_idx >= len(dyt_entries):
            if warn and requested_idx != 0:
                warn(
                    f"DYT entry DATA{selected_idx:03d} was not found in '{emb_name}'. "
                    "Falling back to DATA000."
                )
            selected_idx = 0

        if selected_idx >= len(dyt_entries):
            if warn:
                warn(f"DYT entry DATA000 was not found in '{emb_name}'. Skipping DYT import.")
        else:
            _apply_dyt_entry(dyt_entries[selected_idx], emb_dyt.path)

    def _skip_matcol_import() -> bool:
        return use_unif_env or use_toon_uniffx

    def _apply_params_to_group(group_name: str) -> None:
        group_node = nodes.get(group_name)
        if not (group_node and hasattr(group_node, "inputs") and emm_info):
            return
        for param in emm_info.params:
            if "ON/OFF" in param.name:
                continue
            if _skip_matcol_import() and param.name.startswith("MatCol"):
                continue
            try:
                val = float(param.value)
            except (TypeError, ValueError):
                continue
            if param.name in group_node.inputs:
                with contextlib.suppress(TypeError, ValueError, AttributeError):
                    group_node.inputs[param.name].default_value = val
        if _skip_matcol_import():
            for input_name in ("MatCol0R", "MatCol0G", "MatCol0B"):
                if input_name in group_node.inputs:
                    with contextlib.suppress(TypeError, ValueError, AttributeError):
                        group_node.inputs[input_name].default_value = 0.0

    _apply_params_to_group("XV2_BASIC_SHADER")
    _apply_params_to_group("XV2_BASIC_EYE_SHADER")
    _apply_params_to_group("TOON_UNIF_ENV")


def _validate_face_indices(
    i0: int,
    i1: int,
    i2: int,
    *,
    strict_face_indices: bool,
    max_index: int,
) -> list[int] | None:
    if strict_face_indices:
        if i0 < 0 or i0 > max_index or i1 < 0 or i1 > max_index or i2 < 0 or i2 > max_index:
            return None
        face_indices = [i0, i1, i2]
    else:
        face_indices = [
            max(0, min(i0, max_index)),
            max(0, min(i1, max_index)),
            max(0, min(i2, max_index)),
        ]

    if face_indices[0] in (face_indices[1], face_indices[2]) or face_indices[1] == face_indices[2]:
        return None

    return face_indices


MERGE_DISTANCE_MIN_EDGE_RATIO = 0.5


def _clamp_merge_distance(mesh: bpy.types.Mesh, requested: float) -> float:
    if requested <= 0.0:
        return requested
    vertices = mesh.vertices
    shortest_edge = None
    for edge in mesh.edges:
        first_co = vertices[edge.vertices[0]].co
        second_co = vertices[edge.vertices[1]].co
        length = (first_co - second_co).length
        if length > 1e-12 and (shortest_edge is None or length < shortest_edge):
            shortest_edge = length
    if shortest_edge is None:
        return requested
    return min(requested, shortest_edge * MERGE_DISTANCE_MIN_EDGE_RATIO)


def import_emd(
    path: str,
    esk_override: str = "",
    import_normals: bool = False,
    import_tangents: bool = False,
    merge_by_distance: bool = False,
    merge_distance: float = 0.0001,
    tris_to_quads: bool = False,
    split_submeshes: bool = True,
    shared_armature=None,
    return_armature: bool = False,
    preserve_structure: bool = False,
    dyt_entry_index: int = 0,
    warn: Callable[[str], None] | None = None,
    preloaded_emd: EMD_File | None = None,
    preloaded_esk: ESK_File | None = None,
    source_format: str = "EMD",
    preserve_bone_axes: bool = False,
    disable_dyt: bool = False,
    force_shader_template: str | None = None,
    reuse_materials: bool = True,
    emb_override: str = "",
    emm_override: str = "",
):
    warned_messages: set[str] = set()

    def _warn_once(message: str) -> None:
        if not message or message in warned_messages:
            return
        warned_messages.add(message)
        if warn:
            warn(message)
        else:
            print("Warning:", message)

    source_tag, source_behavior = _resolve_source_behavior(source_format)
    if source_behavior.disable_dyt_default:
        disable_dyt = True
    if source_behavior.preserve_structure_default:
        preserve_structure = True

    emd: EMD_File = preloaded_emd if preloaded_emd is not None else parse_emd(path)
    nsk_has_bones_entries = source_tag == "NSK" and _emd_has_any_triangle_bones(emd)
    nsk_use_rigid_model_placement = source_tag == "NSK" and not nsk_has_bones_entries
    use_linked_model_placement = source_tag == "EMO"
    emb_main = None
    emb_dyt = None
    emb_override_path = (emb_override or "").strip()
    if emb_override_path:
        if os.path.isfile(emb_override_path):
            emb_main = read_emb(emb_override_path)
            if emb_main is None:
                _warn_once(
                    f"Failed to parse EMB override '{os.path.basename(emb_override_path)}'. "
                    "Falling back to default EMB lookup."
                )
            elif not disable_dyt:
                emb_dyt_candidates = [
                    f"{os.path.splitext(emb_override_path)[0]}_dyt.emb",
                    f"{os.path.splitext(emb_override_path)[0]}.dyt.emb",
                ]
                for candidate in emb_dyt_candidates:
                    if os.path.isfile(candidate):
                        emb_dyt = read_emb(candidate)
                        if emb_dyt is not None:
                            break
        else:
            _warn_once(
                f"EMB override was not found: '{emb_override_path}'. "
                "Falling back to default EMB lookup."
            )

    if emb_main is None:
        emb_main, emb_dyt = locate_emb_files(path)

    if disable_dyt and emb_dyt is not None:
        _warn_once("DYT textures are disabled for this import format; skipping DYT lookup.")
        emb_dyt = None

    emm_override_path = (emm_override or "").strip()
    emm_path = ""
    if emm_override_path:
        if os.path.isfile(emm_override_path):
            emm_path = emm_override_path
        else:
            _warn_once(
                f"EMM override was not found: '{emm_override_path}'. "
                "Falling back to default EMM lookup."
            )

    if not emm_path:
        emm_path = locate_emm(path) or ""

    emm_materials = parse_emm(emm_path) if emm_path else []
    emm_by_name = {mat.name.lower(): mat for mat in emm_materials}

    folder = os.path.dirname(path)
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    parts = stem.split("_")

    char_code = parts[0] if parts else stem

    stem_esk = os.path.join(folder, f"{stem}.esk")
    preferred_esk = os.path.join(folder, f"{char_code}_000.esk")
    alt_esk = os.path.join(folder, f"{char_code}.esk")

    esk_path = ""
    esk_candidates = [stem_esk, preferred_esk, alt_esk]

    esk: ESK_File | None = preloaded_esk
    arm_obj = shared_armature

    if esk is not None:
        arm_name = esk.bones[0].name if esk.bones else "Armature"
        if not arm_obj:
            arm_obj = build_armature(esk, arm_name, preserve_bone_axes=preserve_bone_axes)
        if source_tag == "NSK":
            arm_obj.name = stem or arm_name
            arm_obj["esk_root_name_original"] = arm_name
            arm_obj["nsk_source_name"] = stem or ""
        else:
            arm_obj.name = arm_name
        arm_obj["esk_source_path"] = path
        arm_obj["esk_version"] = int(esk.version)
        arm_obj["esk_i10"] = int(esk.i_10)
        arm_obj["esk_i12"] = int(esk.i_12)
        arm_obj["esk_i24"] = int(esk.i_24)
        arm_obj["esk_skeleton_flag"] = int(esk.skeleton_flag)
        arm_obj["esk_skeleton_id"] = str(int(esk.skeleton_id))
        arm_obj.rotation_euler[0] = math.radians(90.0)
        if arm_obj.data:
            arm_obj.data.display_type = "STICK"
    else:
        if esk_override and os.path.exists(esk_override):
            esk_path = esk_override
        else:
            for candidate in esk_candidates:
                if candidate and os.path.exists(candidate):
                    esk_path = candidate
                    break

        if os.path.exists(esk_path):
            try:
                esk = parse_esk(esk_path)
                arm_name = esk.bones[0].name if esk.bones else "Armature"
                if not arm_obj:
                    arm_obj = build_armature(esk, arm_name, preserve_bone_axes=preserve_bone_axes)
                if source_tag == "NSK":
                    arm_obj.name = stem or arm_name
                    arm_obj["esk_root_name_original"] = arm_name
                    arm_obj["nsk_source_name"] = stem or ""
                else:
                    arm_obj.name = arm_name
                arm_obj["esk_source_path"] = esk_path
                arm_obj["esk_version"] = int(esk.version)
                arm_obj["esk_i10"] = int(esk.i_10)
                arm_obj["esk_i12"] = int(esk.i_12)
                arm_obj["esk_i24"] = int(esk.i_24)
                arm_obj["esk_skeleton_flag"] = int(esk.skeleton_flag)
                arm_obj["esk_skeleton_id"] = str(int(esk.skeleton_id))
                arm_obj.rotation_euler[0] = math.radians(90.0)
                if arm_obj.data:
                    arm_obj.data.display_type = "STICK"
            except (OSError, ValueError, RuntimeError, TypeError) as error:
                print("Failed to load ESK:", error)

    imported_objects: list[bpy.types.Object] = []
    structure_parents: dict[object, bpy.types.Object] = {}

    for model in emd.models:
        model_bone_name = (model.name or "").strip()
        model_bone = (
            _find_armature_bone(arm_obj, model_bone_name) if source_tag in {"NSK", "EMO"} else None
        )
        model_has_named_bone = bool(model_bone_name and model_bone is not None)
        model_world_matrix = (
            _get_esk_world_matrix_by_bone_name(esk, model_bone_name)
            if model_has_named_bone
            else None
        )
        model_parent = None
        if preserve_structure:
            # Empty to represent the EMD model
            model_empty_name = f"{model.name}_model" if model.name else "EMD_Model"
            model_parent = bpy.data.objects.new(model_empty_name, None)
            bpy.context.collection.objects.link(model_parent)
            if (
                nsk_use_rigid_model_placement or use_linked_model_placement
            ) and model_has_named_bone:
                if arm_obj:
                    model_parent.parent = arm_obj
                if model_world_matrix is not None:
                    # Use raw skeleton placement so linked rigid parts match source data.
                    model_parent.matrix_local = model_world_matrix
                else:
                    # Fallback when matrix lookup fails.
                    model_parent.location = model_bone.head_local.copy()
            elif arm_obj:
                model_parent.parent = arm_obj
            structure_parents[model] = model_parent

        for mesh in model.meshes:
            mesh_parent = None
            if preserve_structure:
                # Empty to represent the EMD mesh
                mesh_empty_name = f"{mesh.name}_mesh" if mesh.name else "EMD_Mesh"
                mesh_parent = bpy.data.objects.new(mesh_empty_name, None)
                bpy.context.collection.objects.link(mesh_parent)
                if model_parent:
                    mesh_parent.parent = model_parent
                elif arm_obj:
                    mesh_parent.parent = arm_obj
                structure_parents[mesh] = mesh_parent

            for sub_index, sub in enumerate(mesh.submeshes):
                if preserve_structure:
                    submesh_name_base = (mesh.name or sub.name or "EMD_Mesh").strip() or "EMD_Mesh"
                    if len(mesh.submeshes) > 1:
                        submesh_object_name = f"{submesh_name_base}_submesh_{sub_index:02d}"
                    else:
                        submesh_object_name = f"{submesh_name_base}_submesh"
                else:
                    submesh_object_name = sub.name or "EMD_Mesh"

                # Create mesh + object
                me = bpy.data.meshes.new(submesh_object_name)
                obj = bpy.data.objects.new(submesh_object_name, me)
                bpy.context.collection.objects.link(obj)

                # Parenting:
                if preserve_structure and mesh in structure_parents:
                    obj.parent = structure_parents[mesh]
                elif arm_obj:
                    obj.parent = arm_obj
                    if use_linked_model_placement and model_world_matrix is not None:
                        obj.matrix_local = model_world_matrix

                max_index = len(sub.vertices) - 1
                built_positions: list[tuple[float, float, float]] = []
                built_normals: list[mathutils.Vector] = []
                built_uvs: list[tuple[float, float]] = []
                built_uv2s: list[tuple[float, float]] = []
                built_colors: list[tuple[float, float, float, float]] = []
                built_faces: list[tuple[int, int, int]] = []
                built_source_indices: list[int] = []
                built_palette_groups: list[object | None] = []
                has_blend_weights = _submesh_has_blend_weights(sub)
                use_emd_weight_logic_for_submesh = (
                    source_tag == "NSK" and nsk_has_bones_entries and has_blend_weights
                )
                use_indexed_geometry_for_submesh = (
                    source_behavior.use_indexed_geometry and not use_emd_weight_logic_for_submesh
                )
                strict_face_indices = source_behavior.strict_face_indices
                per_vertex_palette_groups: list[object | None] = [None] * max(0, len(sub.vertices))

                if use_indexed_geometry_for_submesh:
                    built_positions = [vertex.pos for vertex in sub.vertices]
                    built_normals = [mathutils.Vector(vertex.normal) for vertex in sub.vertices]
                    built_uvs = [vertex.uv for vertex in sub.vertices]
                    built_uv2s = [vertex.uv2 for vertex in sub.vertices]
                    built_colors = [vertex.color for vertex in sub.vertices]
                    built_source_indices = list(range(len(sub.vertices)))

                    if getattr(sub, "triangle_groups", None):
                        for tri_group in sub.triangle_groups:
                            indices = getattr(tri_group, "indices", [])
                            for i in range(0, len(indices), 3):
                                if i + 2 >= len(indices):
                                    continue
                                face_idxs = _validate_face_indices(
                                    int(indices[i]),
                                    int(indices[i + 1]),
                                    int(indices[i + 2]),
                                    strict_face_indices=strict_face_indices,
                                    max_index=max_index,
                                )
                                if face_idxs is None:
                                    continue
                                built_faces.append(tuple(face_idxs))
                                for src_idx in face_idxs:
                                    if per_vertex_palette_groups[src_idx] is None:
                                        per_vertex_palette_groups[src_idx] = tri_group
                    else:
                        for face in sub.faces:
                            if len(face) < 3:
                                continue
                            face_idxs = _validate_face_indices(
                                int(face[0]),
                                int(face[1]),
                                int(face[2]),
                                strict_face_indices=strict_face_indices,
                                max_index=max_index,
                            )
                            if face_idxs is None:
                                continue
                            built_faces.append(tuple(face_idxs))

                    built_palette_groups = per_vertex_palette_groups
                else:
                    if getattr(sub, "triangle_groups", None):
                        for tri_group in sub.triangle_groups:
                            indices = getattr(tri_group, "indices", [])
                            for i in range(0, len(indices), 3):
                                if i + 2 >= len(indices):
                                    continue
                                face_idxs = _validate_face_indices(
                                    int(indices[i]),
                                    int(indices[i + 1]),
                                    int(indices[i + 2]),
                                    strict_face_indices=strict_face_indices,
                                    max_index=max_index,
                                )
                                if face_idxs is None:
                                    continue
                                new_face: list[int] = []
                                for src_idx in face_idxs:
                                    v = sub.vertices[src_idx]
                                    new_idx = len(built_positions)
                                    built_positions.append(v.pos)
                                    built_normals.append(mathutils.Vector(v.normal))
                                    built_uvs.append(v.uv)
                                    built_uv2s.append(v.uv2)
                                    built_colors.append(v.color)
                                    built_source_indices.append(src_idx)
                                    built_palette_groups.append(tri_group)
                                    new_face.append(new_idx)
                                built_faces.append(tuple(new_face))
                    else:
                        for face in sub.faces:
                            if len(face) < 3:
                                continue
                            face_idxs = _validate_face_indices(
                                int(face[0]),
                                int(face[1]),
                                int(face[2]),
                                strict_face_indices=strict_face_indices,
                                max_index=max_index,
                            )
                            if face_idxs is None:
                                continue
                            new_face: list[int] = []
                            for src_idx in face_idxs:
                                v = sub.vertices[src_idx]
                                new_idx = len(built_positions)
                                built_positions.append(v.pos)
                                built_normals.append(mathutils.Vector(v.normal))
                                built_uvs.append(v.uv)
                                built_uv2s.append(v.uv2)
                                built_colors.append(v.color)
                                built_source_indices.append(src_idx)
                                built_palette_groups.append(None)
                                new_face.append(new_idx)
                            built_faces.append(tuple(new_face))

                if not built_faces:
                    print("No usable faces after rebuild, skipping:", sub.name)
                    continue

                me.from_pydata(built_positions, [], built_faces)
                me.update()

                if built_normals:
                    loop_normals = [
                        built_normals[loop.vertex_index].normalized() for loop in me.loops
                    ]
                    set_custom_split_normals(me, loop_normals)

                for poly in me.polygons:
                    poly.use_smooth = True
                if hasattr(me, "use_auto_smooth"):
                    me.use_auto_smooth = True
                if hasattr(me, "auto_smooth_angle"):
                    me.auto_smooth_angle = math.radians(AUTO_SMOOTH_ANGLE_DEGREES)

                # UV Map 0
                if any(uv != (0.0, 0.0) for uv in built_uvs):
                    uv_layer = me.uv_layers.new(name="UVMap")
                    if len(built_uvs) == len(me.loops):
                        for loop_index, uv_val in enumerate(built_uvs):
                            uv_layer.data[loop_index].uv = uv_val
                    else:
                        for loop in me.loops:
                            if 0 <= loop.vertex_index < len(built_uvs):
                                uv_layer.data[loop.index].uv = built_uvs[loop.vertex_index]

                # UV Map 1 (second UV set)
                if any(uv2 != (0.0, 0.0) for uv2 in built_uv2s):
                    uv2_layer = me.uv_layers.new(name="UVMap_2")
                    if len(built_uv2s) == len(me.loops):
                        for loop_index, uv_val in enumerate(built_uv2s):
                            uv2_layer.data[loop_index].uv = uv_val
                    else:
                        for loop in me.loops:
                            if 0 <= loop.vertex_index < len(built_uv2s):
                                uv2_layer.data[loop.index].uv = built_uv2s[loop.vertex_index]

                # Vertex colors
                if built_positions and any(color != (1.0, 1.0, 1.0, 1.0) for color in built_colors):
                    col_layer = me.color_attributes.new(
                        name="Col",
                        domain="CORNER",
                        type="FLOAT_COLOR",
                    )
                    if len(built_colors) == len(me.loops):
                        for loop_index, col_val in enumerate(built_colors):
                            col_layer.data[loop_index].color = col_val
                    else:
                        for loop in me.loops:
                            if 0 <= loop.vertex_index < len(built_colors):
                                col_layer.data[loop.index].color = built_colors[loop.vertex_index]

                bpy.context.view_layer.objects.active = obj

                if arm_obj is not None and esk is not None and has_blend_weights:
                    if use_indexed_geometry_for_submesh:
                        # Indexed source meshes (NSK-style) must bind on source indices directly.
                        bind_weights(obj, sub, arm_obj, esk)
                    else:
                        bind_weights_built(
                            obj, sub, arm_obj, esk, built_source_indices, built_palette_groups
                        )
                    _grow_tiny_mesh(obj, arm_obj, esk)
                    remove_unused_vertex_groups(obj)

                if split_submeshes:
                    # When not importing custom normals, let Blender manage split normals.
                    # If custom normals were imported, keep them intact.
                    if not import_normals:
                        clear_custom_split_normals(me)

                    if import_tangents:
                        with contextlib.suppress(RuntimeError):
                            me.calc_tangents()

                    if merge_by_distance:
                        effective_distance = _clamp_merge_distance(me, merge_distance)
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.mesh.select_all(action="SELECT")
                        merge_selected_by_distance(
                            effective_distance,
                            use_sharp_edge_from_normals=True,
                        )
                        bpy.ops.object.mode_set(mode="OBJECT")

                    if tris_to_quads:
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.mesh.select_all(action="SELECT")
                        bpy.ops.mesh.tris_convert_to_quads(
                            uvs=True,
                            vcols=True,
                            materials=True,
                            seam=True,
                            sharp=True,
                        )
                        bpy.ops.object.mode_set(mode="OBJECT")

                has_uv2_data = any(uv2 != (0.0, 0.0) for uv2 in built_uv2s)
                emm_info = emm_by_name.get(sub.name.lower())
                shared_material = None
                if source_tag == "EMD" and stem.lower().endswith("_scd") and not emm_path:
                    existing = bpy.data.materials.get(sub.name)
                    if existing is not None and existing.get("emm_shader"):
                        shared_material = existing if reuse_materials else existing.copy()
                    else:
                        _warn_once(
                            f"Material '{sub.name}' is missing. "
                            "Import the main model before this SCD."
                        )

                material = (
                    shared_material
                    if shared_material is not None
                    else create_material(
                        sub.name,
                        source_format=source_tag,
                        emm_shader=getattr(emm_info, "shader", None),
                        force_shader_template=force_shader_template,
                        reuse_materials=reuse_materials,
                    )
                )
                if me.materials:
                    me.materials[0] = material
                else:
                    me.materials.append(material)

                if emm_info:
                    material["emm_name"] = emm_info.name
                    material["emm_shader"] = emm_info.shader
                    material["emm_params"] = [
                        {"name": p.name, "type": int(p.type), "value": p.value}
                        for p in emm_info.params
                    ]
                    for p in emm_info.params:
                        key = f"emm_param_{p.name}"
                        if key not in material:
                            material[key] = p.value
                if shared_material is None and source_behavior.use_placeholder_material:
                    _apply_nsk_placeholder_material(
                        material,
                        sub.texture_sampler_defs,
                        emb_main,
                        image_from_sampler=_image_from_sampler,
                        emm_info=emm_info,
                        has_uv2=has_uv2_data,
                        warn=_warn_once,
                    )
                elif shared_material is None:
                    _apply_shader_material(
                        material,
                        sub.texture_sampler_defs,
                        emb_main,
                        emb_dyt,
                        emm_info,
                        dyt_entry_index=dyt_entry_index,
                        warn=_warn_once,
                    )

                if shared_material is None and sub.texture_sampler_defs:
                    set_sampler_custom_properties(material, sub.texture_sampler_defs)
                    sampler_defs_to_collection(material, sub.texture_sampler_defs)
                material["emd_vertex_flags"] = int(sub.vertex_flags)
                if sub.triangle_groups and sub.triangle_groups[0].bone_names:
                    material["emd_bone_palette"] = list(sub.triangle_groups[0].bone_names)
                obj["emd_file_version"] = int(emd.version)

                imported_objects.append(obj)

    if not split_submeshes and imported_objects:
        ctx = bpy.context
        with contextlib.suppress(RuntimeError):
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.select_all(action="DESELECT")
        for imported_object in imported_objects:
            imported_object.select_set(True)
        ctx.view_layer.objects.active = imported_objects[0]

        parent_before = imported_objects[0].parent

        bpy.ops.object.join()

        merged = ctx.view_layer.objects.active
        merged.name = os.path.splitext(os.path.basename(path))[0]

        if parent_before:
            merged.parent = parent_before

        mesh_data = merged.data

        if import_tangents:
            with contextlib.suppress(RuntimeError):
                mesh_data.calc_tangents()

        if merge_by_distance:
            effective_distance = _clamp_merge_distance(mesh_data, merge_distance)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            merge_selected_by_distance(
                effective_distance,
                use_sharp_edge_from_normals=True,
            )
            bpy.ops.object.mode_set(mode="OBJECT")

        if tris_to_quads:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.tris_convert_to_quads(
                uvs=True,
                vcols=True,
                materials=True,
                seam=True,
                sharp=True,
            )
            bpy.ops.object.mode_set(mode="OBJECT")

    if return_armature:
        return arm_obj, esk


__all__ = [
    "bind_weights",
    "create_material",
    "import_emd",
]
