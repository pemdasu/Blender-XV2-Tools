from __future__ import annotations

from collections.abc import Callable

from .EMO import parse_emo


def import_emo(
    path: str,
    import_normals: bool = False,
    import_tangents: bool = False,
    merge_by_distance: bool = False,
    merge_distance: float = 0.0001,
    tris_to_quads: bool = False,
    split_submeshes: bool = True,
    return_armature: bool = False,
    preserve_structure: bool = True,
    reuse_materials: bool = True,
    warn: Callable[[str], None] | None = None,
):
    from ..EMD.importer import import_emd

    emo = parse_emo(path)
    if emo.emd_file is None:
        raise ValueError("Failed to convert EMO data to in-memory EMD data.")

    arm_obj, esk = import_emd(
        path,
        esk_override="",
        import_normals=import_normals,
        import_tangents=import_tangents,
        merge_by_distance=merge_by_distance,
        merge_distance=merge_distance,
        tris_to_quads=tris_to_quads,
        split_submeshes=split_submeshes,
        shared_armature=None,
        return_armature=True,
        preserve_structure=preserve_structure,
        dyt_entry_index=0,
        warn=warn,
        preloaded_emd=emo.emd_file,
        preloaded_esk=emo.skeleton,
        source_format="EMO",
        preserve_bone_axes=True,
        disable_dyt=False,
        force_shader_template=None,
        reuse_materials=reuse_materials,
    )

    if arm_obj is not None:
        arm_obj["emo_source_path"] = path
        arm_obj["emo_version"] = int(emo.version)
        arm_obj["emo_i24"] = int(emo.i_24)

    if return_armature:
        return arm_obj, esk
    return None


__all__ = ["import_emo"]
