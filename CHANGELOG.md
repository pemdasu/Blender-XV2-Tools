# Changelog

## 1.2.0

### Added

- Support for blender versions 4.0.0 through 5.2.1.
- Shader previews for `TOON_UNIF_SCROLL` and its animated UV scrolling.
- Rim-light overwrite support for `_OWR` shader variants, including `TOON_UNIF_ENV_OWR`.
- Standard alpha blending for supported character shaders, including transparent materials such as glasses.
- Camera animation export support for old Unleashed camera add-on rigs.

### Fixed

- EMO and EMA export issues that caused missing parts, broken animation, incorrect rotation origins, and shading problems in-game.
- Import issues with models that use very small bone scales like `XBF_000_Face_forehead`.
- Skeleton export now uses edited armature bone transforms.
- Animation baking now has an option to bake with "Visual Keying" on export.
- SCD imports now keep shared materials and textures when no separate EMM exists.
- Camera export no longer repeatedly appends `.cam.ean` to the filename.
- Menu icons sometimes not loading properly.

### Changed

- Imports now set the scene’s color-management view to Standard.

### Notes

- Additive and subtractive blending are unsupported.
- SCD files without their own materials require the main model to be imported first, or selected in the same import.

## 1.1.0

- Added EMO import and export support
- Added EMA import and export support for `.obj.ema` files

## 1.0.9

- Added support for NSK and MAP/FMP import and export (MAP export options `Export collision meshes` and `Export linked NSK files` are experimental)
- Added import/export category dropdown menus with format icons
- Added character shader support for EMM shader names that include `UNIF_ENV` (like frieza orb) (not 100% accurate)
- Added a `Reuse Materials` import option for EMD/NSK/MAP
- Internal code clean up and refactoring
- Readded "Auto Merge by Distance" option

## 1.0.8

- Fixed addon enable error caused by EAN/ESK circular imports (Reported by CSD59ALL on GitHub)
- Fixed ESK export writing corrupt ESK files
- Added DYT Index option on EMD import (e.g. 2 = DATA002)
- Cleaned imported EMB/DYT texture names
- Fixed a bug where importing multiple character parts would re-use textures from other characters with same material name
- Added warnings for unsupported DDS imports
- Removed "Auto Merge by Distance" import option (Now always on)

## 1.0.7

- Fixed thumb import for ESKs
- Exporting EANs now properly keeps the ean index sorting
- Added option to add "dummy" keyframes on ean export

## 1.0.6

- Added auto bake actions on export for EAN and CAM.EAN

## 1.0.5

- Fixed EAN export not properly exporting multiple actions

## 1.0.4

- Fixed CAM.EAN roll being inverted
- Fixed CAM.EAN export EAN index not being preserved
- Improved Camera control UI panel
- Added create/rename action option **Note:** When creating a new action, make sure to do it through this to make sure the FOV/Roll are stored properly
- Added auto basic shader assignment
- Fixed EMB parser not being able to read some newer EMB files

## 1.0.3

- Added support for importing and exporting CAM.EAN and EAN files

## 1.0.2

- Added support for importing SCDs with proper bone linking

## 1.0.1

- Fix EMD export ignoring sharp edges

## 1.0.0

- Initial release
- Import/export for EMD models.
- Import support for EMB textures, including DYT pack splitting into per-line slices (Shader not implemented yet).
- ESK skeleton import.
