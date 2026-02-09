import contextlib
import math
import os

import bpy
import mathutils

from ...ui import link_scd_armatures
from ..ESK import build_armature
from .EAN import ComponentType, EANAnimation, EANFile, EANNode, read_ean


def _find_cam_node(anim: EANAnimation) -> EANNode | None:
    for node in anim.nodes:
        if node.bone_name.lower() == "node":
            return node
    return anim.nodes[0] if anim.nodes else None


def _find_target_node(anim: EANAnimation, cam_node: EANNode | None) -> EANNode | None:
    preferred_names = {"CameraTarget"}
    for node in anim.nodes:
        if cam_node is not None and node is cam_node:
            continue
        if node.bone_name.lower() in preferred_names:
            return node
    for node in anim.nodes:
        if cam_node is not None and node is cam_node:
            continue
        return node
    return cam_node


def _map_vec(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (x, -z, y)


def _get_component(node: EANNode, comp_type: ComponentType):
    for comp in node.components:
        if comp.type == comp_type:
            return comp
    return None


def _create_skeleton_matrices(esk) -> None:
    for bone in esk.bones:
        if getattr(bone, "matrix", None) is not None:
            continue
        pos = getattr(bone, "position", (0.0, 0.0, 0.0))
        rot = getattr(bone, "rotation", (1.0, 0.0, 0.0, 0.0))
        scale = getattr(bone, "scale", (1.0, 1.0, 1.0))
        quat = mathutils.Quaternion((rot[0], rot[1], rot[2], rot[3]))
        bone.matrix = mathutils.Matrix.LocRotScale(
            mathutils.Vector(pos),
            quat,
            mathutils.Vector(scale),
        )


def _build_rest_from_esk(
    esk,
) -> tuple[dict[str, mathutils.Matrix], dict[str, mathutils.Matrix], dict[str, str | None]]:
    _create_skeleton_matrices(esk)
    abs_mats: dict[str, mathutils.Matrix] = {}
    local_mats: dict[str, mathutils.Matrix] = {}
    parents: dict[str, str | None] = {}

    def compute_abs(bone) -> mathutils.Matrix:
        if bone.name in abs_mats:
            return abs_mats[bone.name]
        mat_local = (
            bone.matrix.copy() if getattr(bone, "matrix", None) else mathutils.Matrix.Identity(4)
        )
        if 0 <= bone.parent_index < len(esk.bones):
            parent = esk.bones[bone.parent_index]
            parent_abs = compute_abs(parent)
            abs_mats[bone.name] = parent_abs @ mat_local
            parents[bone.name] = parent.name
        else:
            abs_mats[bone.name] = mat_local
            parents[bone.name] = None
        local_mats[bone.name] = mat_local
        return abs_mats[bone.name]

    for b in esk.bones:
        compute_abs(b)

    return abs_mats, local_mats, parents


def _get_rest_matrices(
    ean: EANFile,
    target_armature: bpy.types.Object | None,
    source_path: str,
    transform_ref: bpy.types.Object | None = None,
) -> tuple[
    bpy.types.Object | None,
    dict[str, mathutils.Matrix],  # target abs
    dict[str, mathutils.Matrix],  # ean abs
    dict[str, mathutils.Matrix],  # ean local
    dict[str, str | None],  # ean parents
]:
    ean_abs, ean_local, ean_parents = _build_rest_from_esk(ean.skeleton)
    target_abs: dict[str, mathutils.Matrix] = {}

    def _collect(bones: bpy.types.ArmatureBones):
        for b in bones:
            target_abs[b.name] = b.matrix_local.copy()

    armature_obj = (
        target_armature if target_armature and target_armature.type == "ARMATURE" else None
    )

    # If no armature was selected, build one from the embedded EAN skeleton.
    if armature_obj is None:
        _create_skeleton_matrices(ean.skeleton)
        arm_name = ean.skeleton.bones[0].name if ean.skeleton.bones else "Armature"
        armature_obj = build_armature(ean.skeleton, armature_name=arm_name)
        if armature_obj:
            armature_obj.name = arm_name
            if armature_obj.data:
                armature_obj.data.name = arm_name
            if transform_ref:
                armature_obj.matrix_world = transform_ref.matrix_world.copy()
            else:
                armature_obj.rotation_euler[0] = math.radians(90.0)
            if armature_obj.data:
                armature_obj.data.display_type = "STICK"
            armature_obj["ean_source"] = os.path.abspath(source_path)

    if armature_obj is None:
        return None, target_abs, ean_abs, ean_local, ean_parents

    _collect(armature_obj.data.bones)
    return armature_obj, target_abs, ean_abs, ean_local, ean_parents


def _interp_component(
    comp, frame: int, default_vals: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    if comp is None or not comp.keyframes:
        return default_vals

    keyframes = sorted(comp.keyframes, key=lambda k: k.frame_index)
    prev = None
    nxt = None
    for keyframe in keyframes:
        if keyframe.frame_index == frame:
            return (keyframe.x, keyframe.y, keyframe.z, keyframe.w)
        if keyframe.frame_index < frame:
            prev = keyframe
        if keyframe.frame_index > frame:
            nxt = keyframe
            break

    if prev is None:
        prev_vals = default_vals
        prev_frame = frame - 1
    else:
        prev_vals = (prev.x, prev.y, prev.z, prev.w)
        prev_frame = prev.frame_index

    if nxt is None:
        next_vals = default_vals
        next_frame = frame + 1
    else:
        next_vals = (nxt.x, nxt.y, nxt.z, nxt.w)
        next_frame = nxt.frame_index

    if next_frame == prev_frame:
        return prev_vals

    factor = (frame - prev_frame) / float(next_frame - prev_frame)
    return tuple(prev_vals[i] + (next_vals[i] - prev_vals[i]) * factor for i in range(4))


def _prep_action(obj: bpy.types.Object, action_name: str) -> None:
    obj.animation_data_create()
    action = bpy.data.actions.new(action_name)
    obj.animation_data.action = action
    action.use_fake_user = True


def _prep_action_data(data_block, action_name: str) -> None:
    data_block.animation_data_create()
    action = bpy.data.actions.new(action_name)
    data_block.animation_data.action = action
    action.use_fake_user = True


def _apply_camera_keyframes(camera: bpy.types.Object, anim: EANAnimation, action_name: str):
    node = _find_cam_node(anim)
    if node is None:
        return

    _prep_action(camera, action_name)
    data_action_name = f"{action_name}_data"
    _prep_action_data(camera.data, data_action_name)

    if camera.animation_data and camera.animation_data.action:
        camera.animation_data.action["ean_index"] = anim.index
    if camera.data and camera.data.animation_data and camera.data.animation_data.action:
        camera.data.animation_data.action["ean_index"] = anim.index

    pos_comp = _get_component(node, ComponentType.Position)
    scale_comp = _get_component(node, ComponentType.Scale)

    camera.rotation_mode = "XYZ"

    if pos_comp:
        for keyframe in pos_comp.keyframes:
            camera.location = _map_vec(keyframe.x, keyframe.y, keyframe.z)
            camera.keyframe_insert(data_path="location", frame=keyframe.frame_index)

    if scale_comp:
        for keyframe in scale_comp.keyframes:
            roll_deg = -math.degrees(keyframe.x)
            fov_deg = math.degrees(keyframe.y)
            if hasattr(camera.data, "xv2_roll") and hasattr(camera.data, "xv2_fov"):
                camera.data.xv2_roll = roll_deg
                camera.data.xv2_fov = fov_deg
                camera.data.keyframe_insert(data_path="xv2_roll", frame=keyframe.frame_index)
                camera.data.keyframe_insert(data_path="xv2_fov", frame=keyframe.frame_index)

    if camera.animation_data and camera.animation_data.action:
        camera.animation_data.action.name = action_name
    if camera.data.animation_data and camera.data.animation_data.action:
        camera.data.animation_data.action.name = data_action_name


def _apply_target_keyframes(target: bpy.types.Object, anim: EANAnimation, action_name: str):
    cam_node = _find_cam_node(anim)
    node = _find_target_node(anim, cam_node)
    if node is None:
        return

    _prep_action(target, action_name)
    target.rotation_mode = "XYZ"

    if target.animation_data and target.animation_data.action:
        target.animation_data.action["ean_index"] = anim.index

    comp = _get_component(node, ComponentType.Rotation) or _get_component(
        node, ComponentType.Position
    )
    if comp:
        for keyframe in comp.keyframes:
            target.location = _map_vec(keyframe.x, keyframe.y, keyframe.z)
            target.keyframe_insert(data_path="location", frame=keyframe.frame_index)


def import_cam_ean(path: str) -> list[bpy.types.Object]:
    ean: EANFile = read_ean(path, link_skeleton=False)
    created: list[bpy.types.Object] = []

    if not ean.is_camera:
        return created

    target_name = "CameraTarget"
    camera_name = "Node"

    rig = bpy.data.objects.new("CameraRig", None)
    rig.empty_display_type = "PLAIN_AXES"
    rig.empty_display_size = 0.01
    rig["ean_source"] = os.path.abspath(path)
    bpy.context.collection.objects.link(rig)
    created.append(rig)

    target = bpy.data.objects.new(target_name, None)
    target.empty_display_type = "PLAIN_AXES"
    target.empty_display_size = 0.25
    target["ean_source"] = os.path.abspath(path)
    bpy.context.collection.objects.link(target)
    target.parent = rig
    created.append(target)

    cam_data = bpy.data.cameras.new(camera_name)
    cam_obj = bpy.data.objects.new(camera_name, cam_data)
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_width = 32.0
    cam_data.sensor_height = 32.0
    cam_obj["ean_source"] = os.path.abspath(path)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.parent = rig
    created.append(cam_obj)

    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    track.use_target_z = True

    for anim in ean.animations:
        cam_action_name = f"Node_{anim.name}"
        target_action_name = f"Target_{anim.name}"
        _apply_camera_keyframes(cam_obj, anim, cam_action_name)
        _apply_target_keyframes(target, anim, target_action_name)

    if cam_obj.data:
        fcurves = cam_obj.data.animation_data.drivers if cam_obj.data.animation_data else []
        for fc in list(fcurves):
            if fc.data_path == "lens":
                cam_obj.data.animation_data.drivers.remove(fc)
        lens_fc = cam_obj.data.driver_add("lens")
        lens_driver = lens_fc.driver
        lens_driver.type = "SCRIPTED"
        lens_driver.expression = "(sensor*0.5)/tan(max(1e-6, radians(fov)*0.5))"
        while lens_driver.variables:
            lens_driver.variables.remove(lens_driver.variables[0])
        var_fov = lens_driver.variables.new()
        var_fov.name = "fov"
        var_fov.targets[0].id_type = "CAMERA"
        var_fov.targets[0].id = cam_obj.data
        var_fov.targets[0].data_path = "xv2_fov"
        var_sensor = lens_driver.variables.new()
        var_sensor.name = "sensor"
        var_sensor.targets[0].id_type = "CAMERA"
        var_sensor.targets[0].id = cam_obj.data
        var_sensor.targets[0].data_path = "sensor_height"

        if target.animation_data is None:
            target.animation_data_create()
        drv_curves = target.animation_data.drivers
        for fc in list(drv_curves):
            if fc.data_path == "rotation_euler" and fc.array_index == 1:
                drv_curves.remove(fc)
        roll_fc = target.driver_add("rotation_euler", 1)
        roll_driver = roll_fc.driver
        roll_driver.type = "SCRIPTED"
        roll_driver.expression = "radians(roll)"
        while roll_driver.variables:
            roll_driver.variables.remove(roll_driver.variables[0])
        var_roll = roll_driver.variables.new()
        var_roll.name = "roll"
        var_roll.targets[0].id_type = "CAMERA"
        var_roll.targets[0].id = cam_obj.data
        var_roll.targets[0].data_path = "xv2_roll"

    return created


def import_ean_animations(
    path: str,
    target_armature: bpy.types.Object | None = None,
    replace_armature: bool = False,
) -> bpy.types.Object | None:
    ean: EANFile = read_ean(path, link_skeleton=True)
    if ean.is_camera:
        import_cam_ean(path)
        return None

    def _relink_armature(old_arm: bpy.types.Object, new_arm: bpy.types.Object) -> None:
        for coll in getattr(old_arm, "users_collection", []):
            if new_arm.name not in {obj.name for obj in coll.objects}:
                coll.objects.link(new_arm)

        new_arm.parent = old_arm.parent
        new_arm.matrix_parent_inverse = old_arm.matrix_parent_inverse.copy()

        for child in list(old_arm.children):
            child.parent = new_arm

        for obj in bpy.data.objects:
            for mod in obj.modifiers:
                if mod.type == "ARMATURE" and getattr(mod, "object", None) is old_arm:
                    mod.object = new_arm
            for constraint in obj.constraints:
                if hasattr(constraint, "target") and getattr(constraint, "target", None) is old_arm:
                    constraint.target = new_arm

    def _find_scd_sources(target_obj: bpy.types.Object) -> list[bpy.types.Object]:
        sources: list[bpy.types.Object] = []
        seen: set[str] = set()
        for obj in bpy.data.objects:
            if obj.type != "ARMATURE":
                continue
            for pbone in obj.pose.bones:
                for constraint in pbone.constraints:
                    if getattr(constraint, "target", None) is target_obj:
                        if obj.name not in seen:
                            sources.append(obj)
                            seen.add(obj.name)
                        break
                else:
                    continue
                break
        return sources

    old_arm = target_armature if replace_armature else None
    ean_arm_name = ean.skeleton.bones[0].name if ean.skeleton and ean.skeleton.bones else "Armature"

    (
        arm_obj,
        _target_abs,
        _ean_abs,
        ean_local,
        _ean_parents,
    ) = _get_rest_matrices(
        ean,
        None if replace_armature else target_armature,
        path,
        transform_ref=target_armature if replace_armature else None,
    )
    if arm_obj is None:
        print("import_ean_animations: select an armature before importing.")
        return None

    if replace_armature and old_arm and arm_obj is not None:
        scd_sources = _find_scd_sources(old_arm)
        arm_data = old_arm.data
        # Preserve the EAN armature naming instead of the previous armature name.
        arm_obj.name = ean_arm_name
        if arm_obj.data:
            arm_obj.data.name = ean_arm_name
        _relink_armature(old_arm, arm_obj)
        for scd_arm in scd_sources:
            with contextlib.suppress(Exception):
                # Relink any SCD armatures that were targeting the old armature to the new one.
                link_scd_armatures(scd_arm, arm_obj)
        with contextlib.suppress(Exception):
            bpy.data.objects.remove(old_arm, do_unlink=True)
        if arm_data and arm_data.users == 0:
            with contextlib.suppress(Exception):
                bpy.data.armatures.remove(arm_data, do_unlink=True)

    action_prefix = arm_obj.name or ean_arm_name
    for anim in sorted(ean.animations, key=lambda a: a.index):
        anim_base = anim.name or f"Anim_{anim.index}"
        action_name = f"{action_prefix}|{anim.index}|{anim_base}"
        action = bpy.data.actions.new(action_name)
        action.use_fake_user = True
        arm_obj.animation_data_create()
        arm_obj.animation_data.action = action

        anim_frames: set[int] = set()

        for node in anim.nodes:
            pose_bone = arm_obj.pose.bones.get(node.bone_name)
            if pose_bone is None:
                continue

            rest_local_ean = ean_local.get(node.bone_name, mathutils.Matrix.Identity(4))

            pos_comp = _get_component(node, ComponentType.Position)
            rot_comp = _get_component(node, ComponentType.Rotation)
            scale_comp = _get_component(node, ComponentType.Scale)

            frames: set[int] = {0}
            if pos_comp:
                frames.update(keyframe.frame_index for keyframe in pos_comp.keyframes)
            if rot_comp:
                frames.update(keyframe.frame_index for keyframe in rot_comp.keyframes)
            if scale_comp:
                frames.update(keyframe.frame_index for keyframe in scale_comp.keyframes)

            rest_loc, rest_rot, rest_scale = rest_local_ean.decompose()
            default_pos = (rest_loc.x, rest_loc.y, rest_loc.z, 1.0)
            default_rot = (rest_rot.x, rest_rot.y, rest_rot.z, rest_rot.w)
            default_scale = (rest_scale.x, rest_scale.y, rest_scale.z, 1.0)

            pose_bone.rotation_mode = "QUATERNION"

            for frame in sorted(frames):
                anim_frames.add(frame)
                pos_vals = _interp_component(pos_comp, frame, default_pos)
                rot_vals = _interp_component(rot_comp, frame, default_rot)
                scale_vals = _interp_component(scale_comp, frame, default_scale)

                baked_local = mathutils.Matrix.LocRotScale(
                    mathutils.Vector(pos_vals[:3]),
                    mathutils.Quaternion((rot_vals[3], rot_vals[0], rot_vals[1], rot_vals[2])),
                    mathutils.Vector(scale_vals[:3]),
                )

                delta = rest_local_ean.inverted_safe() @ baked_local

                loc, quat, scl = delta.decompose()

                pose_bone.location = loc
                pose_bone.keyframe_insert(data_path="location", frame=frame)
                pose_bone.rotation_quaternion = quat
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
                pose_bone.scale = scl
                pose_bone.keyframe_insert(data_path="scale", frame=frame)

        if anim_frames:
            action.frame_range = (min(anim_frames), max(anim_frames))

    return arm_obj


__all__ = ["import_cam_ean", "import_ean_animations"]
