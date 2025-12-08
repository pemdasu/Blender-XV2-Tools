# Blender XV2 Tools

Blender add-on for importing and exporting Dragon Ball Xenoverse 2 assets. Built and tested with **Blender 4.0.2**; other versions may work but are not officially supported.

## Features & Status

| Format | Import | Export | Notes |
| --- | --- | --- | --- |
| EMD | ✅ | ✅ | scds not fully supported as of right now |
| DYT.EMB | ✅ | ❌ ||
| EMB | ✅ | ❌ ||
| ESK | ✅ | ⏳ | scds not fully supported as of right now |
| BCS | ⏳ | ❌ ||
| EMM | ⏳ | ⏳ ||
| CAM.EAN | ⏳ | ⏳ ||
| EAN | ⏳ | ⏳ ||

✅ supported • ❌ not supported/not planned yet • ⏳ planned

## Installation

1. Download the latest zip from the GitHub **Releases** tab (or from `build/` if you built locally).
2. In Blender: `Edit > Preferences > Add-ons > Install…` and select the zip.
3. Enable the add-on named **DRAGON BALL XENOVERSE 2 Tools**.

## Editing EMD texture samplers

- After importing, sampler data is written to the active material (and mirrored on the object) as `emd_texture_sampler_def_*` custom properties and an editable `emd_texture_samplers` collection.
- In the 3D View sidebar (`N`), open the **EMD** tab > **EMD Texture Samplers** panel to add/remove samplers and edit fields (texture index, address modes, filters, scales). The same UI exists in the Material properties context as **EMD Texture Samplers**.
- Use the `+/-` buttons to manage rows, then tweak values; the active row shows editable details in the panel.
- Export prefers material samplers when present; if none are set, default samplers for texture slots 0 and 1 are written.

## Building

Local build:
```bash
python build_addon.py
```
- Outputs a versioned zip to `build/`.
- Pass `--version X.Y.Z` to override the version stamped in the filename.

## Known Issues / Notes

- Blender 4.0.2 is the target version; older/newer builds may work but aren’t guaranteed.
- If hard edges look off after export, re-verify split normals/sharp edges before exporting.
- Export EMD sometimes doesn't detect you have the meshes selected
- Export EMD asks you for a filename despite always taking whatever the mesh name is

## Contributing

- Open issues/PRs are welcome. Please keep changes Blender-4.0.2-compatible unless otherwise discussed.
- Include a brief note in `CHANGELOG.md` for user-visible changes.
