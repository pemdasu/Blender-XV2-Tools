# Changelog

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
