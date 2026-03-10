from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import bpy
import mathutils

from ..NSK.importer import import_nsk
from .FMP import (
    AXIS_XV2_TO_BLENDER,
    iter_object_instance_matrices,
    map_name_from_path,
    normalize_cache_path,
    parse_fmp,
    pick_entity_lod,
    resolve_nsk_path,
    resolve_optional_asset_path,
    sanitize_name,
    serialize_instance_data,
    serialize_visual_lods,
    to_blender_axis,
    to_json_string,
)

_AXIS3_XV2_TO_BLENDER = AXIS_XV2_TO_BLENDER.to_3x3()


def _copy_object_tree(root_object: bpy.types.Object) -> bpy.types.Object:
    source_by_ptr: dict[int, bpy.types.Object] = {}
    source_to_copy: dict[int, bpy.types.Object] = {}
    stack: list[bpy.types.Object] = [root_object]

    while stack:
        source_object = stack.pop()
        source_ptr = int(source_object.as_pointer())
        if source_ptr in source_to_copy:
            continue
        source_by_ptr[source_ptr] = source_object
        source_to_copy[source_ptr] = source_object.copy()
        for child in source_object.children:
            stack.append(child)

    collection = bpy.context.collection
    link_object = collection.objects.link
    for source_ptr, copied_object in source_to_copy.items():
        source_object = source_by_ptr[source_ptr]
        link_object(copied_object)
        source_parent = source_object.parent
        source_parent_ptr = int(source_parent.as_pointer()) if source_parent is not None else 0
        if source_parent_ptr in source_to_copy:
            copied_object.parent = source_to_copy[source_parent_ptr]
            copied_object.matrix_parent_inverse = source_object.matrix_parent_inverse.copy()
        else:
            copied_object.parent = None
            copied_object.matrix_local = source_object.matrix_local.copy()

    for copied_object in source_to_copy.values():
        if copied_object.type != "MESH":
            continue
        for modifier in copied_object.modifiers:
            if modifier.type != "ARMATURE":
                continue
            source_modifier_object = modifier.object
            if source_modifier_object is None:
                continue
            source_modifier_ptr = int(source_modifier_object.as_pointer())
            if source_modifier_ptr in source_to_copy:
                modifier.object = source_to_copy[source_modifier_ptr]

    return source_to_copy[int(root_object.as_pointer())]


def _iter_object_tree(root_object: bpy.types.Object) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    stack: list[bpy.types.Object] = [root_object]
    while stack:
        obj = stack.pop()
        objects.append(obj)
        for child in obj.children:
            stack.append(child)
    return objects


def _move_object_tree_to_collection(
    root_object: bpy.types.Object,
    target_collection: bpy.types.Collection,
) -> None:
    for obj in _iter_object_tree(root_object):
        for source_collection in tuple(obj.users_collection):
            source_collection.objects.unlink(obj)
        target_collection.objects.link(obj)


def _set_subpart_props(
    collider_empty: bpy.types.Object,
    prefix: str,
    subpart,
) -> None:
    if subpart is None:
        return

    collider_empty[f"{prefix}_i_00"] = int(subpart.i_00)
    collider_empty[f"{prefix}_i_02"] = int(subpart.i_02)
    collider_empty[f"{prefix}_i_04"] = int(subpart.i_04)
    collider_empty[f"{prefix}_i_06"] = int(subpart.i_06)
    collider_empty[f"{prefix}_i_08"] = int(subpart.i_08)
    collider_empty[f"{prefix}_mass"] = float(subpart.mass)
    collider_empty[f"{prefix}_f_16"] = float(subpart.f_16)
    collider_empty[f"{prefix}_f_20"] = float(subpart.f_20)
    collider_empty[f"{prefix}_f_24"] = float(subpart.f_24)
    collider_empty[f"{prefix}_f_28"] = float(subpart.f_28)
    collider_empty[f"{prefix}_f_32"] = float(subpart.f_32)
    collider_empty[f"{prefix}_f_36"] = float(subpart.f_36)
    collider_empty[f"{prefix}_f_40"] = float(subpart.f_40)
    collider_empty[f"{prefix}_f_44"] = float(subpart.f_44)
    collider_empty[f"{prefix}_f_48"] = float(subpart.f_48)
    collider_empty[f"{prefix}_f_52"] = float(subpart.f_52)
    collider_empty[f"{prefix}_width_x"] = float(subpart.width[0])
    collider_empty[f"{prefix}_width_y"] = float(subpart.width[1])
    collider_empty[f"{prefix}_width_z"] = float(subpart.width[2])
    collider_empty[f"{prefix}_quat_x"] = float(subpart.quaternion[0])
    collider_empty[f"{prefix}_quat_y"] = float(subpart.quaternion[1])
    collider_empty[f"{prefix}_quat_z"] = float(subpart.quaternion[2])
    collider_empty[f"{prefix}_quat_w"] = float(subpart.quaternion[3])


def _to_blender_point(position: tuple[float, float, float]) -> tuple[float, float, float]:
    vec = _AXIS3_XV2_TO_BLENDER @ mathutils.Vector(position)
    return (float(vec.x), float(vec.y), float(vec.z))


def _build_collision_mesh_data(
    mesh_name: str,
    vertex_data,
) -> bpy.types.Mesh | None:
    if vertex_data is None:
        return None
    if not vertex_data.vertices or not vertex_data.faces:
        return None

    vertices = [_to_blender_point(vertex.position) for vertex in vertex_data.vertices]
    faces: list[tuple[int, int, int]] = []
    face_indices = vertex_data.faces

    for face_offset in range(0, len(face_indices) - 2, 3):
        index0 = int(face_indices[face_offset])
        index1 = int(face_indices[face_offset + 1])
        index2 = int(face_indices[face_offset + 2])
        if (
            index0 in (index1, index2)
            or index0 < 0
            or index1 < 0
            or index2 < 0
            or index0 >= len(vertices)
            or index1 >= len(vertices)
            or index2 >= len(vertices)
            or index1 == index2
        ):
            continue
        faces.append((index0, index1, index2))

    if not faces:
        return None

    mesh_data = bpy.data.meshes.new(mesh_name)
    mesh_data.from_pydata(vertices, [], faces)
    mesh_data.update()
    return mesh_data


def _create_collision_mesh_object(
    collider_empty: bpy.types.Object,
    mesh_name: str,
    collection: bpy.types.Collection | None = None,
    mesh_data: bpy.types.Mesh | None = None,
    vertex_data=None,
) -> bpy.types.Object | None:
    if mesh_data is None:
        mesh_data = _build_collision_mesh_data(mesh_name, vertex_data)
        if mesh_data is None:
            return None

    mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
    target_collection = collection if collection is not None else bpy.context.collection
    target_collection.objects.link(mesh_obj)
    mesh_obj.parent = collider_empty
    mesh_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    mesh_obj.matrix_local = mathutils.Matrix.Identity(4)
    if hasattr(mesh_obj, "display_type"):
        mesh_obj.display_type = "WIRE"
    return mesh_obj


def import_map_in_steps(
    path: str,
    import_normals: bool = True,
    import_tangents: bool = False,
    merge_by_distance: bool = True,
    merge_distance: float = 0.0001,
    tris_to_quads: bool = False,
    split_submeshes: bool = True,
    warn: Callable[[str], None] | None = None,
    import_colliders: bool = True,
    import_collision_meshes: bool = True,
    use_collection_instances: bool = True,
    reuse_materials: bool = True,
) -> Iterator[tuple[int, int, str]]:
    fmp = parse_fmp(path)
    map_name = map_name_from_path(path) or "XV2_MAP"
    collection = bpy.context.collection
    link_object = collection.objects.link
    object_count = len(fmp.objects)
    object_instances = [iter_object_instance_matrices(obj) for obj in fmp.objects]
    total_steps = max(1, sum(max(1, len(instances)) for instances in object_instances))
    done_steps = 0
    yield (0, total_steps, f"[MAP] Loading {map_name} ({object_count} objects)")

    map_root = bpy.data.objects.new(f"{map_name}_map", None)
    link_object(map_root)
    map_root["fmp_source_path"] = str(path)
    map_root["fmp_version"] = int(fmp.version)
    map_root["fmp_i08"] = int(fmp.i_08)
    map_root["fmp_i12"] = int(fmp.i_12)
    map_root["fmp_i96_0"] = int(fmp.i_96[0])
    map_root["fmp_i96_1"] = int(fmp.i_96[1])
    map_root["fmp_i96_2"] = int(fmp.i_96[2])
    map_root["fmp_i96_3"] = int(fmp.i_96[3])

    base_dir = Path(path).parent
    imported_any = False
    nsk_import_cache: dict[
        tuple,
        tuple[bpy.types.Object, mathutils.Matrix, bpy.types.Collection | None],
    ] = {}
    nsk_import_failures: set[tuple] = set()
    nsk_path_cache: dict[str, Path] = {}
    asset_path_cache: dict[str, Path | None] = {}
    collision_mesh_data_cache: dict[tuple[int, int], bpy.types.Mesh | None] = {}
    missing_nsk_warned: set[str] = set()
    nsk_source_collection = (
        bpy.data.collections.new(f"{map_name}_nsk_sources") if use_collection_instances else None
    )

    for object_index, object_data in enumerate(fmp.objects):
        object_name = sanitize_name(object_data.name, f"object_{object_index:03d}")
        instance_matrices = object_instances[object_index]
        instance_count = max(1, len(instance_matrices))
        multiple_instances = len(instance_matrices) > 1
        instance_data_json = serialize_instance_data(object_data.instance_data)
        instance_data_json_str = (
            to_json_string(instance_data_json) if instance_data_json is not None else None
        )
        collision_group = None
        collision_group_index = int(object_data.hitbox_group_index)
        if collision_group_index != 0xFFFF and 0 <= collision_group_index < len(
            fmp.collision_groups
        ):
            collision_group = fmp.collision_groups[collision_group_index]

        entity_entries: list[
            tuple[
                int,
                object,
                int,
                object,
                str,
                str,
                str,
                tuple,
                str,
                mathutils.Matrix,
                str | None,
            ]
        ] = []
        for entity_index, entity in enumerate(object_data.entities):
            lod_entry = pick_entity_lod(entity)
            if lod_entry is None:
                continue
            lod_index, lod = lod_entry

            nsk_raw = str(lod.nsk_file)
            nsk_path = nsk_path_cache.get(nsk_raw)
            if nsk_path is None:
                nsk_path = resolve_nsk_path(base_dir, nsk_raw)
                nsk_path_cache[nsk_raw] = nsk_path
            if not nsk_path.exists():
                nsk_warning_path = str(nsk_path)
                if nsk_warning_path not in missing_nsk_warned:
                    if warn:
                        warn(f"Missing NSK for map entity: {nsk_path}")
                    else:
                        print("Warning: Missing NSK for map entity:", nsk_path)
                    missing_nsk_warned.add(nsk_warning_path)
                continue

            emb_raw = entity.visual.emb_file if entity.visual is not None else ""
            emm_raw = lod.emm_file
            emb_key = f"emb:{emb_raw}"
            emm_key = f"emm:{emm_raw}"
            try:
                emb_override = asset_path_cache[emb_key]
            except KeyError:
                emb_override = resolve_optional_asset_path(base_dir, emb_raw)
                asset_path_cache[emb_key] = emb_override
            try:
                emm_override = asset_path_cache[emm_key]
            except KeyError:
                emm_override = resolve_optional_asset_path(base_dir, emm_raw)
                asset_path_cache[emm_key] = emm_override

            emb_override_str = str(emb_override) if emb_override is not None else ""
            emm_override_str = str(emm_override) if emm_override is not None else ""
            import_key = (
                normalize_cache_path(str(nsk_path)),
                normalize_cache_path(emb_override_str),
                normalize_cache_path(emm_override_str),
                bool(import_normals),
                bool(import_tangents),
                bool(merge_by_distance),
                float(merge_distance),
                bool(tris_to_quads),
                bool(split_submeshes),
            )
            lods_json_str = (
                to_json_string(serialize_visual_lods(entity)) if entity.visual is not None else None
            )
            entity_entries.append(
                (
                    int(entity_index),
                    entity,
                    int(lod_index),
                    lod,
                    str(nsk_path),
                    emb_override_str,
                    emm_override_str,
                    import_key,
                    sanitize_name(
                        entity.visual.name if entity.visual is not None else "",
                        f"entity_{entity_index:03d}",
                    ),
                    to_blender_axis(entity.transform.matrix),
                    lods_json_str,
                )
            )

        collider_entries: list[
            tuple[int, object, object | None, str, mathutils.Matrix, str | None]
        ] = []
        if import_colliders:
            for collider_index, collider in enumerate(object_data.collider_instances):
                group_collider = None
                if collision_group is not None and 0 <= collider_index < len(
                    collision_group.colliders
                ):
                    group_collider = collision_group.colliders[collider_index]
                base_collider_name = (
                    sanitize_name(group_collider.name, "") if group_collider is not None else ""
                )
                if not base_collider_name:
                    base_collider_name = f"{object_name}_collider_{collider_index:03d}"
                havok_params_json = (
                    to_json_string(
                        [
                            {
                                "param1": int(param.param1),
                                "param2": int(param.param2),
                            }
                            for param in collider.havok_group_parameters
                        ]
                    )
                    if collider.havok_group_parameters
                    else None
                )
                collider_entries.append(
                    (
                        int(collider_index),
                        collider,
                        group_collider,
                        base_collider_name,
                        to_blender_axis(collider.matrix.matrix),
                        havok_params_json,
                    )
                )

        for instance_index, object_matrix in enumerate(instance_matrices):
            object_empty_name = (
                f"{object_name}_object_{instance_index:03d}"
                if multiple_instances
                else f"{object_name}_object"
            )
            object_empty = bpy.data.objects.new(object_empty_name, None)
            link_object(object_empty)
            object_empty.parent = map_root
            object_empty.matrix_local = to_blender_axis(object_matrix)
            object_empty["fmp_object_index"] = int(object_data.idx)
            object_empty["fmp_object_name"] = object_data.name
            object_empty["fmp_i_04"] = int(object_data.i_04)
            object_empty["fmp_flags"] = int(object_data.flags)
            object_empty["fmp_instance_index"] = int(instance_index)
            object_empty["fmp_initial_entity_index"] = int(object_data.initial_entity_index)
            object_empty["fmp_f_32"] = float(object_data.f_32)
            if instance_data_json_str is not None:
                object_empty["fmp_instance_data_json"] = instance_data_json_str
            if import_colliders and object_data.collider_instances:
                imported_any = True

            if import_colliders:
                for (
                    collider_index,
                    collider,
                    group_collider,
                    base_collider_name,
                    collider_matrix_local,
                    havok_params_json,
                ) in collider_entries:
                    collider_name = (
                        f"{base_collider_name}_{instance_index:03d}"
                        if multiple_instances
                        else base_collider_name
                    )
                    collider_empty = bpy.data.objects.new(collider_name, None)
                    link_object(collider_empty)
                    collider_empty.parent = object_empty
                    collider_empty.matrix_local = collider_matrix_local
                    collider_empty["fmp_collider_index"] = int(collider_index)
                    collider_empty["fmp_collider_i20"] = int(collider.i_20)
                    collider_empty["fmp_collider_i22"] = int(collider.i_22)
                    collider_empty["fmp_collider_f24"] = float(collider.f_24)
                    collider_empty["fmp_collider_f28"] = float(collider.f_28)
                    collider_empty["fmp_collider_action_offset"] = int(collider.action_offset)

                    if havok_params_json is not None:
                        collider_empty["fmp_collider_havok_params_json"] = havok_params_json
                    if collider.subpart1 is not None:
                        _set_subpart_props(
                            collider_empty,
                            "fmp_collider_subpart1",
                            collider.subpart1,
                        )
                    if collider.subpart2 is not None:
                        _set_subpart_props(
                            collider_empty,
                            "fmp_collider_subpart2",
                            collider.subpart2,
                        )
                    if import_collision_meshes and group_collider is not None:
                        collision_mesh_name = f"{base_collider_name}_collision"
                        mesh_cache_key = (collision_group_index, int(collider_index))
                        if mesh_cache_key not in collision_mesh_data_cache:
                            collision_mesh_data_cache[mesh_cache_key] = _build_collision_mesh_data(
                                collision_mesh_name,
                                group_collider.collision_vertex_data,
                            )
                        collision_mesh_data = collision_mesh_data_cache[mesh_cache_key]
                        collision_mesh_obj = _create_collision_mesh_object(
                            collider_empty,
                            collision_mesh_name,
                            collection=collection,
                            mesh_data=collision_mesh_data,
                        )
                        if collision_mesh_obj is not None:
                            collision_mesh_obj["fmp_collision_mesh"] = True
                            fragment_group = None
                            for havok_entry in getattr(group_collider, "havok_entries", []):
                                fragment_group = int(getattr(havok_entry, "fragment_group", 0))
                                break
                            type_sources = (
                                fragment_group,
                                collider.subpart1.i_00 if collider.subpart1 is not None else None,
                                collider.subpart2.i_00 if collider.subpart2 is not None else None,
                            )
                            for collision_type in type_sources:
                                if collision_type is None:
                                    continue
                                collision_mesh_obj["fmp_collision_type"] = int(collision_type)
                                collision_mesh_obj["fmp_collision_type_original"] = int(
                                    collision_type
                                )
                                break

            for (
                entity_index,
                entity,
                lod_index,
                lod,
                nsk_path_str,
                emb_override_str,
                emm_override_str,
                import_key,
                entity_name,
                entity_matrix_local,
                lods_json_str,
            ) in entity_entries:
                entity_empty_name = (
                    f"{object_name}_{entity_name}_entity_{entity_index:02d}_{instance_index:03d}"
                    if multiple_instances
                    else f"{object_name}_{entity_name}_entity_{entity_index:02d}"
                )
                entity_empty = bpy.data.objects.new(entity_empty_name, None)
                link_object(entity_empty)
                entity_empty.parent = object_empty
                entity_empty.matrix_local = entity_matrix_local
                entity_empty["fmp_entity_index"] = int(entity_index)
                entity_empty["fmp_entity_i04"] = int(entity.i_04)
                entity_empty["fmp_lod_distance"] = float(lod.distance)
                entity_empty["fmp_lod_index"] = int(lod_index)
                entity_empty["fmp_nsk_file"] = str(lod.nsk_file)
                entity_empty["fmp_emm_file"] = str(lod.emm_file)
                entity_empty["fmp_emb_file"] = (
                    str(entity.visual.emb_file) if entity.visual is not None else ""
                )
                if entity.visual is not None:
                    entity_empty["fmp_visual_name"] = str(entity.visual.name)
                    entity_empty["fmp_visual_i04"] = int(entity.visual.i_04)
                    entity_empty["fmp_ema_file"] = str(entity.visual.ema_file)
                    entity_empty["fmp_visual_i24"] = int(entity.visual.i_24)
                    entity_empty["fmp_visual_i28"] = int(entity.visual.i_28)
                    entity_empty["fmp_visual_i36"] = int(entity.visual.i_36)
                    entity_empty["fmp_visual_f40"] = float(entity.visual.f_40)
                    entity_empty["fmp_visual_f44"] = float(entity.visual.f_44)
                    if lods_json_str is not None:
                        entity_empty["fmp_lods_json"] = lods_json_str
                if import_key in nsk_import_failures:
                    continue
                cached_entry = nsk_import_cache.get(import_key)
                template_arm_obj = cached_entry[0] if cached_entry is not None else None
                template_root_local = cached_entry[1] if cached_entry is not None else None
                template_collection = cached_entry[2] if cached_entry is not None else None

                if template_arm_obj is None:
                    arm_obj, _ = import_nsk(
                        nsk_path_str,
                        import_normals=import_normals,
                        import_tangents=import_tangents,
                        merge_by_distance=merge_by_distance,
                        merge_distance=merge_distance,
                        tris_to_quads=tris_to_quads,
                        split_submeshes=split_submeshes,
                        return_armature=True,
                        reuse_materials=reuse_materials,
                        warn=warn,
                        emb_override=emb_override_str,
                        emm_override=emm_override_str,
                    )
                    new_template_collection = None
                    if arm_obj is not None:
                        if use_collection_instances and nsk_source_collection is not None:
                            new_template_collection = bpy.data.collections.new(
                                f"{arm_obj.name}_src"
                            )
                            nsk_source_collection.children.link(new_template_collection)
                            _move_object_tree_to_collection(arm_obj, new_template_collection)
                        nsk_import_cache[import_key] = (
                            arm_obj,
                            arm_obj.matrix_local.copy(),
                            new_template_collection,
                        )
                        template_arm_obj = arm_obj
                        template_root_local = arm_obj.matrix_local.copy()
                        template_collection = new_template_collection
                    else:
                        nsk_import_failures.add(import_key)

                    if (
                        use_collection_instances
                        and template_arm_obj is not None
                        and template_collection is not None
                    ):
                        arm_obj = bpy.data.objects.new(
                            f"{template_arm_obj.name}_instance",
                            None,
                        )
                        link_object(arm_obj)
                        arm_obj.instance_type = "COLLECTION"
                        arm_obj.instance_collection = template_collection
                        arm_obj.matrix_local = mathutils.Matrix.Identity(4)
                else:
                    if use_collection_instances and template_collection is not None:
                        arm_obj = bpy.data.objects.new(
                            f"{template_arm_obj.name}_instance",
                            None,
                        )
                        link_object(arm_obj)
                        arm_obj.instance_type = "COLLECTION"
                        arm_obj.instance_collection = template_collection
                        arm_obj.matrix_local = mathutils.Matrix.Identity(4)
                    else:
                        arm_obj = _copy_object_tree(template_arm_obj)
                        if template_root_local is not None:
                            arm_obj.matrix_local = template_root_local.copy()

                if arm_obj is None:
                    continue

                arm_obj.parent = entity_empty
                arm_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                imported_any = True

            done_steps += 1
            yield (
                done_steps,
                total_steps,
                f"[MAP] {map_name}: object {object_index + 1}/{object_count}, "
                f"instance {instance_index + 1}/{instance_count}",
            )

    if done_steps < total_steps:
        yield (total_steps, total_steps, f"[MAP] Finished {map_name}")

    return map_root if imported_any else None


def import_map(
    path: str,
    import_normals: bool = True,
    import_tangents: bool = False,
    merge_by_distance: bool = True,
    merge_distance: float = 0.0001,
    tris_to_quads: bool = False,
    split_submeshes: bool = True,
    warn: Callable[[str], None] | None = None,
    import_colliders: bool = True,
    import_collision_meshes: bool = True,
    use_collection_instances: bool = True,
    reuse_materials: bool = True,
) -> bpy.types.Object | None:
    iterator = import_map_in_steps(
        path=path,
        import_normals=import_normals,
        import_tangents=import_tangents,
        merge_by_distance=merge_by_distance,
        merge_distance=merge_distance,
        tris_to_quads=tris_to_quads,
        split_submeshes=split_submeshes,
        warn=warn,
        import_colliders=import_colliders,
        import_collision_meshes=import_collision_meshes,
        use_collection_instances=use_collection_instances,
        reuse_materials=reuse_materials,
    )
    while True:
        try:
            next(iterator)
        except StopIteration as stop:
            return stop.value


__all__ = [
    "import_map",
    "import_map_in_steps",
]
