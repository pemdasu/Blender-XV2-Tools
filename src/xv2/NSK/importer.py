import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import bpy
import mathutils

from ..EMD.EMD import VERTEX_BLENDWEIGHT, EMD_File, EMD_Submesh
from ..ESK import ESK_File
from .NSK import parse_nsk


@dataclass(frozen=True)
class SourceImportBehavior:
    disable_dyt_default: bool = False
    preserve_structure_default: bool = False
    strict_face_indices: bool = False
    use_indexed_geometry: bool = False
    use_placeholder_material: bool = False


SOURCE_BEHAVIORS: dict[str, SourceImportBehavior] = {
    "EMD": SourceImportBehavior(),
    "NSK": SourceImportBehavior(
        disable_dyt_default=True,
        preserve_structure_default=True,
        strict_face_indices=True,
        use_indexed_geometry=True,
        use_placeholder_material=True,
    ),
}


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


SPECIAL_SHADER_RULES: dict[str, dict[str, object]] = {
    "U3_BM_FAMILY": {
        "includes": "U3_BM",
        "alias": "U3_MUV_SM_BLEND_S",
    },
    "U2_MUV_FAMILY": {
        "includes": "U2_MUV",
        "alias": "U2_MUV_C_L_SM",
    },
    "U2_SUV_SM_BLEND_MOD_S": {
        "profile": "stage_blend_mod",
        "roles": ["BaseA", "BaseB"],
    },
    "U3SPUA_BM_BUMP_SM_S": {
        "profile": "stage_bump_blend",
        "roles": ["BaseA", "ShadowUV2", "BaseB", "Normal"],
    },
}


def _find_special_shader_rule(shader_name: str) -> dict[str, object] | None:
    exact_rule = SPECIAL_SHADER_RULES.get(shader_name)
    if exact_rule is not None:
        return exact_rule

    for rule in SPECIAL_SHADER_RULES.values():
        includes_value = rule.get("includes")
        if isinstance(includes_value, str):
            include_token = includes_value.strip().upper()
            if include_token and include_token in shader_name:
                return rule
        elif isinstance(includes_value, list):
            for token in includes_value:
                include_token = str(token).strip().upper()
                if include_token and include_token in shader_name:
                    return rule

    return None


def resolve_source_behavior(source_format: str) -> tuple[str, SourceImportBehavior]:
    source_tag = (source_format or "EMD").strip().upper() or "EMD"
    return source_tag, SOURCE_BEHAVIORS.get(source_tag, SOURCE_BEHAVIORS["EMD"])


@cache
def _get_stage_node_group_template(group_name: str) -> bpy.types.NodeTree | None:
    blend_path = Path(__file__).resolve().parents[2] / "shader" / "stage_shader_nodes.blend"
    if not blend_path.is_file():
        return None
    try:
        loaded = []
        with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
            if group_name in data_from.node_groups:
                data_to.node_groups = [group_name]
                loaded = list(data_to.node_groups)
        if loaded:
            node_group = loaded[0]
            if isinstance(node_group, str):
                node_group = bpy.data.node_groups.get(node_group)
            return node_group
    except (OSError, RuntimeError, ReferenceError, ValueError) as exc:
        print("Failed to load stage shader node group:", group_name, exc)
    return None


def _create_stage_group_node(
    nodes: bpy.types.Nodes,
    group_name: str,
    location: tuple[float, float],
) -> bpy.types.Node | None:
    group_template = _get_stage_node_group_template(group_name)
    try:
        missing = (group_template is None) or (group_template.name not in bpy.data.node_groups)
    except ReferenceError:
        missing = True
    if missing:
        _get_stage_node_group_template.cache_clear()
        group_template = _get_stage_node_group_template(group_name)
    if group_template is None:
        return None
    node = nodes.new("ShaderNodeGroup")
    node.location = location
    node.name = f"XV2_{group_name}"
    node.label = group_name
    node.node_tree = group_template
    return node


def _set_group_input_value(node: bpy.types.Node | None, socket_name: str, value: float) -> None:
    if node is None or not hasattr(node, "inputs"):
        return
    if socket_name not in node.inputs:
        return
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        node.inputs[socket_name].default_value = value


def _get_group_socket(
    node: bpy.types.Node | None,
    socket_name: str,
    *,
    output: bool,
    allow_fallback: bool = True,
) -> bpy.types.NodeSocket | None:
    if node is None:
        return None
    sockets = node.outputs if output else node.inputs
    if socket_name in sockets:
        return sockets[socket_name]
    if allow_fallback and sockets:
        return sockets[0]
    return None


def _get_stage_shader_roles(
    emm_shader: str | None,
    sampler_count: int,
    has_uv2: bool,
) -> tuple[str, list[str]]:
    shader_name = (emm_shader or "").strip().upper()
    special_rule = _find_special_shader_rule(shader_name)
    if special_rule is not None:
        alias_name = str(special_rule.get("alias", "")).strip().upper()
        if alias_name:
            shader_name = alias_name

        explicit_profile = special_rule.get("profile")
        explicit_roles = special_rule.get("roles")
        if isinstance(explicit_profile, str) and isinstance(explicit_roles, list):
            role_values = [str(role) for role in explicit_roles[:sampler_count]]
            if len(role_values) < sampler_count:
                role_values.extend(["Unknown"] * (sampler_count - len(role_values)))
            return explicit_profile, role_values

    roles = ["Unknown"] * max(0, sampler_count)
    if sampler_count > 0:
        roles[0] = "BaseA"

    has_sm = "_SM" in shader_name
    has_bump = "BUMP" in shader_name
    has_blend = "BLEND" in shader_name
    has_blendmap = "BLENDMAP" in shader_name
    has_mod = "_MOD_" in shader_name
    shader_family = (
        "u2" if shader_name.startswith("U2_") else "u3" if shader_name.startswith("U3") else "other"
    )

    match True:
        case _ if has_bump:
            profile = "stage_bump"
        case _ if has_mod:
            profile = "stage_mod"
        case _ if has_blendmap:
            profile = "stage_blendmap"
        case _ if has_blend:
            profile = "stage_blend"
        case _ if has_sm:
            profile = "stage_shadow"
        case _ if shader_name.startswith("T1") and not has_bump and not has_blend:
            profile = "stage_single"
        case _:
            profile = "stage_basic"

    if sampler_count > 1:
        match True:
            case _ if has_mod:
                roles[1] = "BaseB"
            case _ if has_blendmap:
                roles[1] = "BlendMaskAlpha"
            case _ if has_sm:
                roles[1] = "ShadowUV2" if has_uv2 else "ShadowOrControl"
            case _ if has_blend:
                roles[1] = "BlendControl"

    if has_blend and sampler_count > 2:
        roles[2] = "BaseB"
    if has_blendmap and sampler_count > 2:
        roles[2] = "BaseB"

    if has_bump:
        normal_slot: int | None
        match shader_family:
            case "u2":
                normal_slot = 2 if sampler_count > 2 else None
            case "u3" if sampler_count > 3:
                normal_slot = 3
            case "u3" if sampler_count > 2:
                normal_slot = 2
            case "u3":
                normal_slot = None
            case _:
                normal_slot = 2 if sampler_count > 2 else None

        if normal_slot is not None:
            roles[normal_slot] = "Normal"

    return profile, roles


def _nsk_role_colorspace(role_name: str) -> str:
    match role_name:
        case "Normal":
            return "Non-Color"
        case _:
            return "sRGB"


def _read_emm_float_param(emm_info, param_name: str, default_value: float) -> float:
    if emm_info is None:
        return default_value
    for param in getattr(emm_info, "params", []) or []:
        if getattr(param, "name", "") != param_name:
            continue
        return _to_float(getattr(param, "value", default_value), default_value)
    return default_value


def _get_alpha_blend_setup(emm_info) -> tuple[bool, float]:
    if emm_info is None:
        return False, 0.0

    alpha_blend_enabled = False
    alpha_blend_type_value: float | None = None
    has_alpha_blend_type_param = False

    for param in getattr(emm_info, "params", []) or []:
        param_name = str(getattr(param, "name", ""))
        param_name_lower = param_name.lower()
        param_value = getattr(param, "value", 0.0)

        if param_name_lower == "alphablend":
            alpha_blend_enabled = _to_float(param_value, 0.0) == 1.0
        elif param_name_lower.startswith("alphablendtype"):
            has_alpha_blend_type_param = True
            if alpha_blend_type_value is None:
                alpha_blend_type_value = _to_float(param_value, 0.0)

    return alpha_blend_enabled or has_alpha_blend_type_param, alpha_blend_type_value or 0.0


def apply_nsk_placeholder_material(
    mat: bpy.types.Material,
    sampler_defs,
    emb_main,
    image_from_sampler: Callable[
        [object, int, object, Callable[[str], None] | None], bpy.types.Image | None
    ],
    emm_info=None,
    has_uv2: bool = False,
    warn: Callable[[str], None] | None = None,
) -> None:
    if not mat:
        return
    mat.use_nodes = True
    if not mat.node_tree:
        return

    emm_shader = getattr(emm_info, "shader", "") if emm_info is not None else ""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    links.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (760, 0)

    if not sampler_defs:
        mat["nsk_placeholder_profile"] = "stage_basic"
        mat["nsk_source_shader_name"] = str(emm_shader or "")
        mat["nsk_placeholder_has_uv2"] = bool(has_uv2)
        return

    profile_name, slot_roles = _get_stage_shader_roles(
        emm_shader=emm_shader,
        sampler_count=len(sampler_defs),
        has_uv2=has_uv2,
    )
    use_alpha_blend, alpha_blend_type_value = _get_alpha_blend_setup(emm_info)

    texture_nodes: list[bpy.types.Node] = []
    for sampler_index, sampler in enumerate(sampler_defs):
        image = image_from_sampler(sampler_defs, sampler_index, emb_main, warn=warn)
        role_name = slot_roles[sampler_index] if sampler_index < len(slot_roles) else "Unknown"
        tex_index = int(sampler.texture_index)

        y_offset = -220 * sampler_index
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.location = (-640, y_offset)
        tex_node.name = f"NSK_Sampler_{sampler_index:02d}"
        tex_node.label = f"S{sampler_index} [{role_name}] -> EMB[{tex_index}]"
        tex_node.image = image

        if image is not None:
            if hasattr(tex_node, "interpolation"):
                tex_node.interpolation = "Linear"
            if hasattr(tex_node, "projection"):
                tex_node.projection = "FLAT"
            if hasattr(tex_node, "extension"):
                tex_node.extension = "REPEAT"
            with contextlib.suppress(AttributeError, ValueError):
                image.colorspace_settings.name = _nsk_role_colorspace(role_name)
            with contextlib.suppress(AttributeError, ValueError):
                image.alpha_mode = "CHANNEL_PACKED"

        use_uv2_sampler = has_uv2 and (
            role_name in {"ShadowUV2", "BlendMaskAlpha"}
            or (role_name == "BaseB" and profile_name != "stage_blend_mod")
        )
        uv_group_name = "UV2Scale" if use_uv2_sampler else "UVScale"
        uv_node = _create_stage_group_node(
            nodes,
            uv_group_name,
            (-920, y_offset),
        )
        if uv_node is not None:
            _set_group_input_value(uv_node, "U", float(getattr(sampler, "scale_u", 1.0)))
            _set_group_input_value(uv_node, "V", float(getattr(sampler, "scale_v", 1.0)))
            uv_out = _get_group_socket(uv_node, "UV", output=True)
            if uv_out is not None:
                with contextlib.suppress(KeyError, RuntimeError, TypeError, ValueError):
                    links.new(uv_out, tex_node.inputs["Vector"])

        texture_nodes.append(tex_node)
        mat[f"nsk_sampler_{sampler_index:02d}_role"] = role_name
        mat[f"nsk_sampler_{sampler_index:02d}_texture_index"] = tex_index
        mat[f"nsk_sampler_{sampler_index:02d}_uv_group"] = uv_group_name
        mat[f"nsk_sampler_{sampler_index:02d}_scale_u"] = float(getattr(sampler, "scale_u", 1.0))
        mat[f"nsk_sampler_{sampler_index:02d}_scale_v"] = float(getattr(sampler, "scale_v", 1.0))

    def _valid_texture_node(slot_index: int) -> bpy.types.Node | None:
        if slot_index < 0 or slot_index >= len(texture_nodes):
            return None
        node = texture_nodes[slot_index]
        if getattr(node, "image", None) is None:
            return None
        return node

    def _slot_color(slot_index: int) -> bpy.types.NodeSocket | None:
        node = _valid_texture_node(slot_index)
        return node.outputs.get("Color") if node is not None else None

    def _slot_alpha(slot_index: int) -> bpy.types.NodeSocket | None:
        node = _valid_texture_node(slot_index)
        return node.outputs.get("Alpha") if node is not None else None

    def _find_slot_index_for_roles(*roles: str) -> int:
        wanted = set(roles)
        for index, role_name in enumerate(slot_roles):
            if role_name in wanted:
                return index
        return -1

    def _link_if(
        output_socket: bpy.types.NodeSocket | None,
        input_socket: bpy.types.NodeSocket | None,
    ) -> None:
        if output_socket is None or input_socket is None:
            return
        with contextlib.suppress(RuntimeError, TypeError, ValueError):
            links.new(output_socket, input_socket)

    def _connect_surface_to_output(surface_socket: bpy.types.NodeSocket | None) -> None:
        if surface_socket is None:
            return
        _link_if(surface_socket, out_node.inputs.get("Surface"))

    def _set_alpha_blend_material_settings() -> None:
        if hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = "HASHED"
        if hasattr(mat, "alpha_threshold"):
            mat.alpha_threshold = 0.5

    base_slot = _find_slot_index_for_roles("BaseA")
    if base_slot < 0:
        base_slot = 0
    blend_slot = _find_slot_index_for_roles("BaseB")
    control_slot = _find_slot_index_for_roles(
        "BlendControl",
        "BlendMaskAlpha",
        "ShadowUV2",
        "ShadowOrControl",
    )
    normal_slot = _find_slot_index_for_roles("Normal")

    surface_color = _slot_color(base_slot)
    skip_post_chain = False
    skip_multiply = False

    if profile_name == "stage_single":
        _set_alpha_blend_material_settings()

    match profile_name:
        case "stage_blend" | "stage_mod":
            if blend_slot < 0 and len(texture_nodes) > 1:
                blend_slot = 1
            blend_group = _create_stage_group_node(nodes, "BLEND", (-260, 80))
            if blend_group is not None and blend_slot >= 0:
                _link_if(
                    _slot_color(base_slot),
                    _get_group_socket(blend_group, "Tex0Col", output=False, allow_fallback=False),
                )
                _link_if(
                    _slot_color(blend_slot),
                    _get_group_socket(blend_group, "Tex2Col", output=False, allow_fallback=False),
                )
                surface_color = _get_group_socket(
                    blend_group, "Result", output=True
                ) or _get_group_socket(
                    blend_group,
                    "Color",
                    output=True,
                )
            elif blend_slot >= 0:
                mix_node = nodes.new("ShaderNodeMixRGB")
                mix_node.location = (-220, 80)
                mix_node.blend_type = "MIX"
                mix_node.inputs["Fac"].default_value = 0.5
                _link_if(_slot_color(base_slot), mix_node.inputs.get("Color1"))
                _link_if(_slot_color(blend_slot), mix_node.inputs.get("Color2"))
                surface_color = mix_node.outputs.get("Color")
        case "stage_blendmap":
            if blend_slot < 0 and len(texture_nodes) > 2:
                blend_slot = 2
            if control_slot < 0 and len(texture_nodes) > 1:
                control_slot = 1
            blendmap_group = _create_stage_group_node(nodes, "BLENDmap", (-260, 80))
            if blendmap_group is not None and blend_slot >= 0 and control_slot >= 0:
                _link_if(
                    _slot_color(base_slot),
                    _get_group_socket(
                        blendmap_group,
                        "Tex0Col",
                        output=False,
                        allow_fallback=False,
                    ),
                )
                _link_if(
                    _slot_color(blend_slot),
                    _get_group_socket(
                        blendmap_group,
                        "Tex2Col",
                        output=False,
                        allow_fallback=False,
                    ),
                )
                _link_if(
                    _slot_alpha(base_slot),
                    _get_group_socket(
                        blendmap_group,
                        "Tex0Alp",
                        output=False,
                        allow_fallback=False,
                    ),
                )
                _link_if(
                    _slot_alpha(control_slot),
                    _get_group_socket(
                        blendmap_group,
                        "Tex1Alp",
                        output=False,
                        allow_fallback=False,
                    ),
                )
                surface_color = _get_group_socket(
                    blendmap_group,
                    "Result",
                    output=True,
                ) or _get_group_socket(
                    blendmap_group,
                    "Color",
                    output=True,
                )
            elif blend_slot >= 0 and control_slot >= 0:
                mix_node = nodes.new("ShaderNodeMixRGB")
                mix_node.location = (-220, 80)
                mix_node.blend_type = "MIX"
                _link_if(_slot_alpha(control_slot), mix_node.inputs.get("Fac"))
                _link_if(_slot_color(base_slot), mix_node.inputs.get("Color1"))
                _link_if(_slot_color(blend_slot), mix_node.inputs.get("Color2"))
                surface_color = mix_node.outputs.get("Color")
        case "stage_bump":
            if normal_slot < 0 and len(texture_nodes) > 2:
                normal_slot = 2
            bump_group = _create_stage_group_node(nodes, "BUMP", (-260, 80))
            if bump_group is not None:
                _link_if(
                    _slot_color(base_slot),
                    _get_group_socket(bump_group, "Tex0Col", output=False, allow_fallback=False),
                )
                _link_if(
                    _slot_alpha(base_slot),
                    _get_group_socket(bump_group, "Tex0Alp", output=False, allow_fallback=False),
                )
                _link_if(
                    _slot_color(normal_slot),
                    _get_group_socket(bump_group, "Tex2Col", output=False, allow_fallback=False),
                )
                _set_group_input_value(
                    bump_group,
                    "MatSpcR",
                    _read_emm_float_param(emm_info, "MatSpcR", 0.8),
                )
                _set_group_input_value(
                    bump_group,
                    "MatSpcG",
                    _read_emm_float_param(emm_info, "MatSpcG", 0.8),
                )
                _set_group_input_value(
                    bump_group,
                    "MatSpcB",
                    _read_emm_float_param(emm_info, "MatSpcB", 0.8),
                )
                surface_color = _get_group_socket(
                    bump_group, "Color", output=True
                ) or _get_group_socket(
                    bump_group,
                    "Result",
                    output=True,
                )
        case "stage_bump_blend":
            if normal_slot < 0:
                if len(texture_nodes) > 3:
                    normal_slot = 3
                elif len(texture_nodes) > 2:
                    normal_slot = 2
            if blend_slot < 0 and len(texture_nodes) > 2:
                blend_slot = 2

            bump_output = None
            bump_group = _create_stage_group_node(nodes, "BUMP", (-420, 80))
            if bump_group is not None:
                _link_if(
                    _slot_color(base_slot),
                    _get_group_socket(bump_group, "Tex0Col", output=False, allow_fallback=False),
                )
                _link_if(
                    _slot_alpha(base_slot),
                    _get_group_socket(bump_group, "Tex0Alp", output=False, allow_fallback=False),
                )
                _link_if(
                    _slot_color(normal_slot),
                    _get_group_socket(bump_group, "Tex2Col", output=False, allow_fallback=False),
                )
                _set_group_input_value(
                    bump_group,
                    "MatSpcR",
                    _read_emm_float_param(emm_info, "MatSpcR", 0.8),
                )
                _set_group_input_value(
                    bump_group,
                    "MatSpcG",
                    _read_emm_float_param(emm_info, "MatSpcG", 0.8),
                )
                _set_group_input_value(
                    bump_group,
                    "MatSpcB",
                    _read_emm_float_param(emm_info, "MatSpcB", 0.8),
                )
                bump_output = _get_group_socket(
                    bump_group, "Color", output=True
                ) or _get_group_socket(
                    bump_group,
                    "Result",
                    output=True,
                )

            surface_color = bump_output
            if bump_output is not None and blend_slot >= 0:
                blend_group = _create_stage_group_node(nodes, "BLEND", (-220, 80))
                if blend_group is not None:
                    _link_if(
                        bump_output,
                        _get_group_socket(
                            blend_group, "Tex0Col", output=False, allow_fallback=False
                        ),
                    )
                    _link_if(
                        _slot_color(blend_slot),
                        _get_group_socket(
                            blend_group, "Tex2Col", output=False, allow_fallback=False
                        ),
                    )
                    surface_color = _get_group_socket(
                        blend_group, "Result", output=True
                    ) or _get_group_socket(
                        blend_group,
                        "Color",
                        output=True,
                    )
        case "stage_blend_mod":
            if blend_slot < 0 and len(texture_nodes) > 2:
                blend_slot = 2
            elif blend_slot < 0 and len(texture_nodes) > 1:
                blend_slot = 1

            blend_mod_group = _create_stage_group_node(nodes, "BLEND_MOD", (-260, 80))
            if blend_mod_group is not None and blend_slot >= 0:
                _link_if(
                    _slot_color(base_slot),
                    _get_group_socket(
                        blend_mod_group, "Tex0Col", output=False, allow_fallback=False
                    ),
                )
                _link_if(
                    _slot_color(blend_slot),
                    _get_group_socket(
                        blend_mod_group, "Tex2Col", output=False, allow_fallback=False
                    ),
                )
                _link_if(
                    _slot_alpha(blend_slot),
                    _get_group_socket(
                        blend_mod_group, "Tex2Alp", output=False, allow_fallback=False
                    ),
                )
                _set_group_input_value(
                    blend_mod_group,
                    "MatSpcR",
                    _read_emm_float_param(emm_info, "MatSpcR", 0.8),
                )
                _set_group_input_value(
                    blend_mod_group,
                    "MatSpcG",
                    _read_emm_float_param(emm_info, "MatSpcG", 0.8),
                )
                _set_group_input_value(
                    blend_mod_group,
                    "MatSpcB",
                    _read_emm_float_param(emm_info, "MatSpcB", 0.8),
                )
                surface_color = _get_group_socket(
                    blend_mod_group, "Result", output=True
                ) or _get_group_socket(
                    blend_mod_group,
                    "Color",
                    output=True,
                )
                skip_multiply = True
        case "stage_single":
            shadow_group = _create_stage_group_node(nodes, "ShadowCast", (-420, 80))
            shadow_output = _slot_color(base_slot)
            if shadow_group is not None:
                _link_if(
                    _slot_color(base_slot),
                    _get_group_socket(shadow_group, "MatInput", output=False, allow_fallback=False),
                )
                shadow_output = _get_group_socket(
                    shadow_group, "Result", output=True
                ) or _get_group_socket(
                    shadow_group,
                    "Color",
                    output=True,
                )

            alpha_group = _create_stage_group_node(nodes, "AlphaBlend", (-220, 80))
            if alpha_group is not None:
                _link_if(
                    shadow_output,
                    _get_group_socket(alpha_group, "Tex0Col", output=False, allow_fallback=False),
                )
                _link_if(
                    _slot_alpha(base_slot),
                    _get_group_socket(alpha_group, "Tex0Alp", output=False, allow_fallback=False),
                )
                _set_group_input_value(
                    alpha_group,
                    "AlphaBlendType",
                    alpha_blend_type_value,
                )
                surface_color = (
                    _get_group_socket(alpha_group, "Shader", output=True)
                    or _get_group_socket(
                        alpha_group,
                        "Result",
                        output=True,
                    )
                    or _get_group_socket(
                        alpha_group,
                        "Color",
                        output=True,
                    )
                )
                skip_post_chain = True
        case _:
            pass

    if not skip_post_chain and (
        control_slot < 0
        and len(texture_nodes) > 1
        and profile_name != "stage_blend_mod"
        and ("_SM" in emm_shader.upper() or profile_name == "stage_shadow")
    ):
        control_slot = 1

    if (
        not skip_post_chain
        and not skip_multiply
        and control_slot >= 0
        and _valid_texture_node(control_slot) is not None
    ):
        multiply_group = _create_stage_group_node(nodes, "Multiply", (20, 80))
        if multiply_group is not None and surface_color is not None:
            _link_if(
                surface_color,
                _get_group_socket(
                    multiply_group,
                    "MatInput",
                    output=False,
                    allow_fallback=False,
                ),
            )
            _link_if(
                _slot_color(control_slot),
                _get_group_socket(
                    multiply_group,
                    "Tex1Col",
                    output=False,
                    allow_fallback=False,
                ),
            )
            surface_color = _get_group_socket(
                multiply_group, "Result", output=True
            ) or _get_group_socket(
                multiply_group,
                "Color",
                output=True,
            )

    if not skip_post_chain and (
        "_SM" in emm_shader.upper()
        or profile_name in {"stage_blend", "stage_blendmap", "stage_shadow"}
    ):
        shadow_group = _create_stage_group_node(nodes, "ShadowCast", (280, 80))
        if shadow_group is not None and surface_color is not None:
            _link_if(
                surface_color,
                _get_group_socket(shadow_group, "MatInput", output=False, allow_fallback=False),
            )
            if control_slot >= 0:
                _link_if(
                    _slot_color(control_slot),
                    _get_group_socket(
                        shadow_group,
                        "Tex1Col",
                        output=False,
                        allow_fallback=False,
                    ),
                )
            surface_color = _get_group_socket(
                shadow_group, "Result", output=True
            ) or _get_group_socket(
                shadow_group,
                "Color",
                output=True,
            )

    if profile_name != "stage_single" and use_alpha_blend and surface_color is not None:
        alpha_group = _create_stage_group_node(nodes, "AlphaBlend", (540, 80))
        if alpha_group is not None:
            _link_if(
                surface_color,
                _get_group_socket(alpha_group, "Tex0Col", output=False, allow_fallback=False),
            )
            _link_if(
                _slot_alpha(base_slot),
                _get_group_socket(alpha_group, "Tex0Alp", output=False, allow_fallback=False),
            )
            _set_group_input_value(
                alpha_group,
                "AlphaBlendType",
                alpha_blend_type_value,
            )
            surface_color = (
                _get_group_socket(alpha_group, "Shader", output=True)
                or _get_group_socket(alpha_group, "Result", output=True)
                or _get_group_socket(alpha_group, "Color", output=True)
            )
            _set_alpha_blend_material_settings()

    if surface_color is not None:
        _connect_surface_to_output(surface_color)

    mat["nsk_placeholder_profile"] = profile_name
    mat["nsk_source_shader_name"] = str(emm_shader or "")
    mat["nsk_placeholder_has_uv2"] = bool(has_uv2)


def find_armature_bone(arm_obj: bpy.types.Object | None, bone_name: str):
    if arm_obj is None or not bone_name:
        return None
    arm_data = getattr(arm_obj, "data", None)
    if not arm_data or not hasattr(arm_data, "bones"):
        return None
    return arm_data.bones.get(bone_name)


def get_esk_world_matrix_by_bone_name(
    esk: ESK_File | None,
    bone_name: str,
) -> mathutils.Matrix | None:
    if esk is None or not bone_name:
        return None
    index_by_name: dict[str, int] = {}
    for index, bone in enumerate(esk.bones):
        if bone.name:
            index_by_name[bone.name] = index
    bone_index = index_by_name.get(bone_name)
    if bone_index is None:
        return None

    world_mats: dict[int, mathutils.Matrix] = {}

    def _compute_world(index: int) -> mathutils.Matrix:
        if index in world_mats:
            return world_mats[index]
        bone = esk.bones[index]
        matrix = bone.matrix.copy()
        if 0 <= bone.parent_index < len(esk.bones) and esk.bones[bone.parent_index] is not bone:
            matrix = _compute_world(bone.parent_index) @ matrix
        world_mats[index] = matrix
        return matrix

    return _compute_world(bone_index)


def submesh_has_blend_weights(sub: EMD_Submesh) -> bool:
    return bool(int(getattr(sub, "vertex_flags", 0)) & VERTEX_BLENDWEIGHT)


def emd_has_any_triangle_bones(emd: EMD_File) -> bool:
    for model in emd.models:
        for mesh in model.meshes:
            for sub in mesh.submeshes:
                for tri_group in getattr(sub, "triangle_groups", []) or []:
                    if getattr(tri_group, "bone_names", None):
                        return True
    return False


def import_nsk(
    path: str,
    import_normals: bool = False,
    import_tangents: bool = False,
    merge_by_distance: bool = False,
    merge_distance: float = 0.0001,
    tris_to_quads: bool = False,
    split_submeshes: bool = True,
    return_armature: bool = False,
    reuse_materials: bool = True,
    warn: Callable[[str], None] | None = None,
    emb_override: str = "",
    emm_override: str = "",
):
    from ..EMD.importer import import_emd

    nsk = parse_nsk(path)
    return import_emd(
        path,
        esk_override="",
        import_normals=import_normals,
        import_tangents=import_tangents,
        merge_by_distance=merge_by_distance,
        merge_distance=merge_distance,
        tris_to_quads=tris_to_quads,
        split_submeshes=split_submeshes,
        shared_armature=None,
        return_armature=return_armature,
        preserve_structure=True,
        dyt_entry_index=0,
        warn=warn,
        preloaded_emd=nsk.emd_file,
        preloaded_esk=nsk.esk_file,
        source_format="NSK",
        disable_dyt=True,
        force_shader_template="shader",
        reuse_materials=reuse_materials,
        emb_override=emb_override,
        emm_override=emm_override,
    )


__all__ = [
    "SourceImportBehavior",
    "apply_nsk_placeholder_material",
    "emd_has_any_triangle_bones",
    "find_armature_bone",
    "get_esk_world_matrix_by_bone_name",
    "import_nsk",
    "resolve_source_behavior",
    "submesh_has_blend_weights",
]
