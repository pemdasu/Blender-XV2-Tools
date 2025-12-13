import bpy
from bpy.props import EnumProperty, FloatProperty, StringProperty
from mathutils import Matrix


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


def _current_base_from_actions(context) -> str | None:
    cam_obj = _find_camera_object(context)
    if cam_obj and cam_obj.animation_data and cam_obj.animation_data.action:
        name = cam_obj.animation_data.action.name
        if name.startswith("Node_") and not name.endswith("_data"):
            return name[len("Node_") :]
    return None


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
        constraint.inverse_matrix = Matrix.Identity(4)

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
    xv2_cam_new_name: StringProperty(
        name="New Action Name",
        description="Base name for new Node/Target camera actions",
        default="NewCam",
    )  # type: ignore
    xv2_cam_rename_name: StringProperty(
        name="Rename Action",
        description="New base name for the currently assigned Node/Target actions",
        default="",
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


class XV2_OT_cam_create_actions(bpy.types.Operator):
    bl_idname = "xv2_cam.create_actions"
    bl_label = "Create new action"
    bl_description = "Create Node/Target/FOV actions and assign to camera and target"

    def execute(self, context):
        cam_obj = _find_camera_object(context)
        if cam_obj is None:
            self.report({"ERROR"}, "No camera selected")
            return {"CANCELLED"}

        base = context.scene.xv2_cam_props.xv2_cam_new_name.strip() or "NewCam"
        node_name = f"Node_{base}"
        data_name = f"{node_name}_data"
        target_name = f"Target_{base}"

        target_obj = None
        for obj in context.collection.objects:
            if obj.type == "EMPTY" and obj.name.lower().startswith("cameratarget"):
                target_obj = obj
                break
        if target_obj is None:
            self.report({"ERROR"}, "CameraTarget not found in this collection")
            return {"CANCELLED"}

        node_action = bpy.data.actions.get(node_name) or bpy.data.actions.new(node_name)
        node_action.use_fake_user = True
        data_action = bpy.data.actions.get(data_name) or bpy.data.actions.new(data_name)
        data_action.use_fake_user = True
        target_action = bpy.data.actions.get(target_name) or bpy.data.actions.new(target_name)
        target_action.use_fake_user = True

        cam_obj.animation_data_create()
        cam_obj.animation_data.action = node_action
        cam_obj.data.animation_data_create()
        cam_obj.data.animation_data.action = data_action
        target_obj.animation_data_create()
        target_obj.animation_data.action = target_action

        # Make sure FOV/Roll curves exist on data action for the current frame
        frame = context.scene.frame_current
        for prop_name in ("xv2_fov", "xv2_roll"):
            try:
                value = getattr(cam_obj.data, prop_name)
            except Exception:
                continue
            fc = data_action.fcurves.find(prop_name)
            if fc is None:
                fc = data_action.fcurves.new(prop_name)
            fc.keyframe_points.insert(frame, value, options={"REPLACE"})

        # Update selection
        context.scene.xv2_cam_props.xv2_cam_anim = base
        self.report({"INFO"}, f"Assigned actions {node_name}, {target_name}")
        return {"FINISHED"}


class XV2_OT_cam_rename_actions(bpy.types.Operator):
    bl_idname = "xv2_cam.rename_actions"
    bl_label = "Rename action"
    bl_description = "Rename currently assigned Node/Target/Data actions to a new base name"

    def execute(self, context):
        cam_obj = _find_camera_object(context)
        if cam_obj is None:
            self.report({"ERROR"}, "No camera selected")
            return {"CANCELLED"}

        current_base = (
            _current_base_from_actions(context) or context.scene.xv2_cam_props.xv2_cam_anim
        )
        if not current_base or current_base == "NONE":
            self.report({"ERROR"}, "No active camera action to rename")
            return {"CANCELLED"}

        new_base = context.scene.xv2_cam_props.xv2_cam_rename_name.strip()
        if not new_base:
            self.report({"ERROR"}, "Provide a new name")
            return {"CANCELLED"}

        node_old = f"Node_{current_base}"
        data_old = f"{node_old}_data"
        target_old = f"Target_{current_base}"
        node_new = f"Node_{new_base}"
        data_new = f"{node_new}_data"
        target_new = f"Target_{new_base}"

        # Rename actions if they exist
        for old, new in ((node_old, node_new), (data_old, data_new), (target_old, target_new)):
            act = bpy.data.actions.get(old)
            if act:
                act.name = new
                act.use_fake_user = True

        # Update assignments
        cam_obj.animation_data_create()
        cam_obj.animation_data.action = bpy.data.actions.get(node_new)
        if cam_obj.data:
            cam_obj.data.animation_data_create()
            cam_obj.data.animation_data.action = bpy.data.actions.get(data_new)

        target_obj = None
        for obj in context.collection.objects:
            if obj.type == "EMPTY" and obj.name.lower().startswith("cameratarget"):
                target_obj = obj
                break
        if target_obj:
            target_obj.animation_data_create()
            target_obj.animation_data.action = bpy.data.actions.get(target_new)

        context.scene.xv2_cam_props.xv2_cam_anim = new_base
        self.report({"INFO"}, f"Renamed actions to base '{new_base}'")
        return {"FINISHED"}


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
        current_base = _current_base_from_actions(context)
        if current_base and props.xv2_cam_anim != current_base:
            box = layout.box()
            box.label(text=f"Active node action: {current_base}", icon="INFO")

        box_anim = layout.box()
        box_anim.label(text="Animations", icon="ACTION")
        box_anim.prop(props, "xv2_cam_anim", text="Select")
        if props.xv2_cam_anim == "NONE":
            box_anim.label(text="No EAN actions found.", icon="INFO")
        else:
            box_anim.label(text="Sets actions on Node/Target (if found).")

        box_create = layout.box()
        box_create.label(text="Create / Rename", icon="FILE_NEW")
        cam_obj = _find_camera_object(context)
        if cam_obj:
            fov_box = layout.box()
            fov_box.label(text="Lens", icon="CAMERA_DATA")
            fov_box.prop(cam_obj.data, "xv2_fov", text="FOV (deg)")
            fov_box.prop(cam_obj.data, "xv2_roll", text="Roll (deg)")

            box_create.prop(props, "xv2_cam_new_name", text="New action base")
            box_create.operator("xv2_cam.create_actions", icon="ADD")
            box_create.prop(props, "xv2_cam_rename_name", text="Rename action")
            box_create.operator("xv2_cam.rename_actions", icon="GREASEPENCIL")

            box_link = layout.box()
            box_link.label(text="Bone Link", icon="CONSTRAINT")
            box_link.prop(props, "link_armature", text="Armature")
            box_link.prop(props, "link_bone", text="Bone")
            box_link.operator("xv2_cam.link_bone", icon="CONSTRAINT")
        else:
            layout.label(text="Select a camera object.", icon="INFO")


classes = [
    CameraEANProperties,
    CameraFOVRollProperties,
    DATA_PT_xv2_camera_actions,
    XV2_OT_cam_create_actions,
    XV2_OT_cam_rename_actions,
    XV2_OT_cam_link_bone,
]


__all__ = [
    "CameraEANProperties",
    "CameraFOVRollProperties",
    "DATA_PT_xv2_camera_actions",
    "XV2_OT_cam_create_actions",
    "XV2_OT_cam_rename_actions",
    "XV2_OT_cam_link_bone",
    "classes",
]
