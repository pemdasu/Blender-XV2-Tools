import contextlib
import math
import os

import bpy
import mathutils

from ...ui import sampler_defs_to_collection
from ...utils import remove_unused_vertex_groups
from ..EMB import attach_emb_textures_to_material, locate_emb_files
from ..ESK import ESK_File, build_armature, parse_esk
from .EMD import (
    EMD_File,
    EMD_Submesh,
    parse_emd,
    set_sampler_custom_properties,
)

AUTO_SMOOTH_ANGLE_DEGREES = 30.0


def bind_weights(
    obj: bpy.types.Object,
    sub: EMD_Submesh,
    arm_obj: bpy.types.Object,
    esk: ESK_File,
):
    # Create vertex groups for all real bones (skip dummy root at index 0)
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
        # Fallback: no palettes in the file, so treat bone_ids as global ESK indices.
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


def create_material(submesh_name: str) -> bpy.types.Material:
    material = bpy.data.materials.new(name=submesh_name or "EMD_Material")
    material.use_nodes = True
    if material.node_tree:
        material.node_tree.nodes.clear()
        material.node_tree.links.clear()
    return material


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
):
    emd: EMD_File = parse_emd(path)
    emb_main, emb_dyt = locate_emb_files(path)

    folder = os.path.dirname(path)
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    parts = stem.split("_")

    char_code = parts[0] if parts else stem

    preferred_esk = os.path.join(folder, f"{char_code}_000.esk")
    alt_esk1 = os.path.join(folder, f"{char_code}.esk")

    esk_path = ""
    if os.path.exists(preferred_esk):
        esk_path = preferred_esk
    elif os.path.exists(alt_esk1):
        esk_path = alt_esk1

    esk: ESK_File | None = None
    arm_obj = shared_armature

    if esk_override and os.path.exists(esk_override):
        esk_path = esk_override

    if os.path.exists(esk_path):
        try:
            esk = parse_esk(esk_path)
            arm_name = esk.bones[0].name if esk.bones else "Armature"
            if not arm_obj:
                arm_obj = build_armature(esk, arm_name)
            arm_obj.name = arm_name
            arm_obj.rotation_euler[0] = math.radians(90.0)
            if arm_obj.data:
                arm_obj.data.display_type = "STICK"
        except Exception as error:
            print("Failed to load ESK:", error)

    imported_objects: list[bpy.types.Object] = []
    structure_parents: dict[object, bpy.types.Object] = {}

    for model in emd.models:
        model_parent = None
        if preserve_structure:
            # Empty to represent the EMD model
            model_parent = bpy.data.objects.new(model.name or "EMD_Model", None)
            bpy.context.collection.objects.link(model_parent)
            if arm_obj:
                model_parent.parent = arm_obj
            structure_parents[model] = model_parent

        for mesh in model.meshes:
            mesh_parent = None
            if preserve_structure:
                # Empty to represent the EMD mesh
                mesh_parent = bpy.data.objects.new(mesh.name or "EMD_Mesh", None)
                bpy.context.collection.objects.link(mesh_parent)
                if model_parent:
                    mesh_parent.parent = model_parent
                elif arm_obj:
                    mesh_parent.parent = arm_obj
                structure_parents[mesh] = mesh_parent

            for sub in mesh.submeshes:
                # Create mesh + object
                me = bpy.data.meshes.new(sub.name or "EMD_Mesh")
                obj = bpy.data.objects.new(sub.name or "EMD_Mesh", me)
                bpy.context.collection.objects.link(obj)

                # Parenting:
                if preserve_structure and mesh in structure_parents:
                    obj.parent = structure_parents[mesh]
                elif arm_obj:
                    obj.parent = arm_obj

                max_index = len(sub.vertices) - 1
                built_positions: list[tuple[float, float, float]] = []
                built_normals: list[mathutils.Vector] = []
                built_uvs: list[tuple[float, float]] = []
                built_uv2s: list[tuple[float, float]] = []
                built_colors: list[tuple[float, float, float, float]] = []
                built_faces: list[tuple[int, int, int]] = []
                built_source_indices: list[int] = []
                built_palette_groups: list[object | None] = []

                if getattr(sub, "triangle_groups", None):
                    for tri_group in sub.triangle_groups:
                        indices = getattr(tri_group, "indices", [])
                        for i in range(0, len(indices), 3):
                            if i + 2 >= len(indices):
                                continue
                            face_idxs = [
                                max(0, min(indices[i], max_index)),
                                max(0, min(indices[i + 1], max_index)),
                                max(0, min(indices[i + 2], max_index)),
                            ]
                            if (
                                face_idxs[0] in (face_idxs[1], face_idxs[2])
                                or face_idxs[1] == face_idxs[2]
                            ):
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
                        face_idxs = [
                            max(0, min(face[0], max_index)),
                            max(0, min(face[1], max_index)),
                            max(0, min(face[2], max_index)),
                        ]
                        if (
                            face_idxs[0] in (face_idxs[1], face_idxs[2])
                            or face_idxs[1] == face_idxs[2]
                        ):
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
                    with contextlib.suppress(Exception):
                        me.create_normals_split()
                    try:
                        me.normals_split_custom_set(loop_normals)
                    except Exception:
                        with contextlib.suppress(Exception):
                            me.free_normals_split()
                    with contextlib.suppress(Exception):
                        me.validate(clean_customdata=False)

                with contextlib.suppress(Exception):
                    for poly in me.polygons:
                        poly.use_smooth = True
                    me.use_auto_smooth = True
                    me.auto_smooth_angle = math.radians(AUTO_SMOOTH_ANGLE_DEGREES)

                # UV Map 0
                if any(uv != (0.0, 0.0) for uv in built_uvs):
                    uv_layer = me.uv_layers.new(name="UVMap")
                    for loop_index, uv_val in enumerate(built_uvs):
                        uv_layer.data[loop_index].uv = uv_val

                # UV Map 1 (second UV set)
                if any(uv2 != (0.0, 0.0) for uv2 in built_uv2s):
                    uv2_layer = me.uv_layers.new(name="UVMap_2")
                    for loop_index, uv_val in enumerate(built_uv2s):
                        uv2_layer.data[loop_index].uv = uv_val

                # Vertex colors
                if built_positions and any(color != (1.0, 1.0, 1.0, 1.0) for color in built_colors):
                    col_layer = me.color_attributes.new(
                        name="Col",
                        domain="CORNER",
                        type="FLOAT_COLOR",
                    )
                    for loop_index, col_val in enumerate(built_colors):
                        col_layer.data[loop_index].color = col_val

                bpy.context.view_layer.objects.active = obj

                if split_submeshes:
                    # When not importing custom normals, let Blender manage split normals.
                    # If custom normals were imported, keep them intact.
                    if not import_normals:
                        with contextlib.suppress(Exception):
                            me.free_normals_split()

                    if import_tangents:
                        with contextlib.suppress(RuntimeError):
                            me.calc_tangents()

                    if merge_by_distance:
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.mesh.select_all(action="SELECT")
                        bpy.ops.mesh.remove_doubles(
                            threshold=merge_distance,
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

                material = create_material(sub.name)
                if me.materials:
                    me.materials[0] = material
                else:
                    me.materials.append(material)

                if sub.texture_sampler_defs:
                    set_sampler_custom_properties(material, sub.texture_sampler_defs)
                    sampler_defs_to_collection(material, sub.texture_sampler_defs)
                    attach_emb_textures_to_material(
                        material,
                        sub.texture_sampler_defs,
                        emb_main,
                        emb_dyt,
                    )
                material["emd_vertex_flags"] = int(sub.vertex_flags)
                if sub.triangle_groups and sub.triangle_groups[0].bone_names:
                    material["emd_bone_palette"] = list(sub.triangle_groups[0].bone_names)
                obj["emd_file_version"] = int(emd.version)

                imported_objects.append(obj)

                if (
                    arm_obj is not None
                    and esk is not None
                    and sub.vertices
                    and any(vertex.bone_ids for vertex in sub.vertices)
                ):
                    bind_weights_built(
                        obj, sub, arm_obj, esk, built_source_indices, built_palette_groups
                    )
                    remove_unused_vertex_groups(obj)

    if not split_submeshes and imported_objects:
        ctx = bpy.context
        with contextlib.suppress(Exception):
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
            with contextlib.suppress(Exception):
                mesh_data.calc_tangents()

        if merge_by_distance:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.remove_doubles(
                threshold=merge_distance,
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
