import contextlib

import bpy
from bpy.props import EnumProperty, FloatProperty, StringProperty


def _find_camera_object(context):
    if context.object and context.object.type == "CAMERA":
        return context.object
    if context.camera:
        for obj in bpy.data.objects:
            if obj.type == "CAMERA" and obj.data == context.camera:
                return obj
    return None


def _ean_anim_items(self, context):
    items = []
    seen = set()
    for action in bpy.data.actions:
        name = action.name
        if not name.startswith("Node_") or name.endswith("_data"):
            continue
        base = name[len("Node_") :]
        if base in seen:
            continue
        seen.add(base)
        items.append((base, base, ""))
    if not items:
        items.append(("NONE", "None", "No EAN actions found"))
    return items


def _bone_items(self, context):
    arm = getattr(self, "link_armature", None)
    if arm is None or arm.type != "ARMATURE" or not arm.data:
        return [("NONE", "None", "No armature")]
    return [(bone.name, bone.name, "") for bone in arm.data.bones]


def _apply_selected_animation(self, context):
    cam_obj = _find_camera_object(context)
    if cam_obj is None:
        return
    base = self.xv2_cam_anim
    if base == "NONE":
        return
    node_action = bpy.data.actions.get(f"Node_{base}")
    target_action = bpy.data.actions.get(f"Target_{base}")
    data_action = bpy.data.actions.get(f"Node_{base}_data")

    cam_obj.animation_data_create()
    if node_action:
        cam_obj.animation_data.action = node_action
    if data_action:
        cam_obj.data.animation_data_create()
        cam_obj.data.animation_data.action = data_action
        frame = context.scene.frame_current
        for name in ("xv2_fov", "xv2_roll"):
            fc = data_action.fcurves.find(name)
            if fc is None:
                continue
            value = fc.evaluate(frame)
            setattr(cam_obj.data, name, value)

    if target_action:
        for obj in context.collection.objects:
            if obj.type == "EMPTY" and obj.name.lower().startswith("cameratarget"):
                obj.animation_data_create()
                obj.animation_data.action = target_action
                break


def _update_fov_roll(self, context):
    return


def _check_parent(obj: bpy.types.Object, parent: bpy.types.Object) -> None:
    if obj.parent is parent:
        return
    mw = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_world = mw


def _find_or_create_rig(cam_obj: bpy.types.Object, target: bpy.types.Object) -> bpy.types.Object:
    if cam_obj.parent and cam_obj.parent.type == "EMPTY":
        return cam_obj.parent
    if target.parent and target.parent.type == "EMPTY":
        return target.parent
    rig = bpy.data.objects.new("CameraRig", None)
    rig.empty_display_type = "PLAIN_AXES"
    rig.empty_display_size = 0.25
    rig["ean_source"] = cam_obj.get("ean_source", "")
    bpy.context.collection.objects.link(rig)
    _check_parent(cam_obj, rig)
    _check_parent(target, rig)
    return rig


class XV2_OT_cam_link_bone(bpy.types.Operator):
    bl_idname = "xv2_cam.link_bone"
    bl_label = "Link to Bone"
    bl_description = "Attach camera rig to selected armature/bone via Child Of"

    @classmethod
    def poll(cls, context):
        return _find_camera_object(context) is not None

    def execute(self, context):
        cam_obj = _find_camera_object(context)
        if cam_obj is None:
            self.report({"ERROR"}, "No camera selected")
            return {"CANCELLED"}

        props = context.scene.xv2_cam_props
        arm_obj = props.link_armature
        bone_name = props.link_bone

        if arm_obj is None or arm_obj.type != "ARMATURE" or bone_name in (None, "", "NONE"):
            self.report({"ERROR"}, "Select an armature and bone")
            return {"CANCELLED"}

        target_obj = None
        for obj in context.collection.objects:
            if obj.type == "EMPTY" and obj.name.lower().startswith("cameratarget"):
                target_obj = obj
                break
        if target_obj is None:
            self.report({"ERROR"}, "CameraTarget not found in this collection")
            return {"CANCELLED"}

        rig = _find_or_create_rig(cam_obj, target_obj)
        _check_parent(cam_obj, rig)
        _check_parent(target_obj, rig)

        constraint = rig.constraints.get("XV2_BoneLink")
        if constraint and constraint.type != "CHILD_OF":
            rig.constraints.remove(constraint)
            constraint = None
        if constraint is None:
            constraint = rig.constraints.new(type="CHILD_OF")
        constraint.name = "XV2_BoneLink"
        constraint.target = arm_obj
        constraint.subtarget = bone_name

        if constraint.target:
            ctx = {
                "constraint": constraint,
                "object": rig,
                "active_object": rig,
                "selected_objects": [rig],
                "selected_editable_objects": [rig],
            }
            with contextlib.suppress(Exception):
                bpy.ops.constraint.childof_clear_inverse(
                    ctx, constraint=constraint.name, owner="OBJECT"
                )
            with contextlib.suppress(Exception):
                bpy.ops.constraint.childof_set_inverse(
                    ctx, constraint=constraint.name, owner="OBJECT"
                )

            target_mat = (
                constraint.target.matrix_world @ constraint.target.pose.bones[bone_name].matrix
            )
            constraint.inverse_matrix = target_mat.inverted() @ rig.matrix_world
        for attr in (
            "use_rotation_x",
            "use_rotation_y",
            "use_rotation_z",
            "use_scale_x",
            "use_scale_y",
            "use_scale_z",
        ):
            if hasattr(constraint, attr):
                setattr(constraint, attr, False)
        for attr in ("use_location_x", "use_location_y", "use_location_z"):
            if hasattr(constraint, attr):
                setattr(constraint, attr, True)

        self.report({"INFO"}, f"Linked camera rig to {arm_obj.name}:{bone_name}")
        return {"FINISHED"}


class CameraEANProperties(bpy.types.PropertyGroup):
    xv2_cam_anim: EnumProperty(
        name="Animation",
        description="Select a camera/target EAN animation",
        items=_ean_anim_items,
        update=_apply_selected_animation,
    )  # type: ignore
    xv2_cam_status: StringProperty(name="Status", default="")  # type: ignore
    link_armature: bpy.props.PointerProperty(  # type: ignore
        name="Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj and obj.type == "ARMATURE",
    )
    link_bone: EnumProperty(  # type: ignore
        name="Bone",
        description="Bone to link the camera rig to",
        items=_bone_items,
    )


class CameraFOVRollProperties(bpy.types.PropertyGroup):
    xv2_fov: FloatProperty(
        name="XV2 FOV (deg)",
        description="XV2 vertical FOV in degrees",
        default=40.0,
        update=_update_fov_roll,
    )  # type: ignore
    xv2_roll: FloatProperty(
        name="XV2 Roll (deg)",
        description="XV2 roll in degrees",
        default=0.0,
        update=_update_fov_roll,
    )  # type: ignore


class DATA_PT_xv2_camera_actions(bpy.types.Panel):
    bl_label = "XV2 Camera"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        cam = context.camera
        return cam is not None and isinstance(cam, bpy.types.Camera)

    def draw(self, context):
        layout = self.layout
        props = context.scene.xv2_cam_props
        layout.prop(props, "xv2_cam_anim", text="Animation")
        if props.xv2_cam_anim == "NONE":
            layout.label(text="No EAN actions found.", icon="INFO")
        else:
            layout.label(text="Sets actions on Node/Target (if found).")

        layout.separator()
        cam_obj = _find_camera_object(context)
        if cam_obj:
            layout.prop(cam_obj.data, "xv2_fov", text="FOV (deg)")
            layout.prop(cam_obj.data, "xv2_roll", text="Roll (deg)")
            layout.separator()
            layout.label(text="Bone Link:")
            layout.prop(props, "link_armature", text="Armature")
            layout.prop(props, "link_bone", text="Bone")
            layout.operator("xv2_cam.link_bone", icon="CONSTRAINT")
        else:
            layout.label(text="Select a camera object.", icon="INFO")


classes = [
    CameraEANProperties,
    CameraFOVRollProperties,
    DATA_PT_xv2_camera_actions,
    XV2_OT_cam_link_bone,
]


__all__ = [
    "CameraEANProperties",
    "CameraFOVRollProperties",
    "DATA_PT_xv2_camera_actions",
    "XV2_OT_cam_link_bone",
    "classes",
]
