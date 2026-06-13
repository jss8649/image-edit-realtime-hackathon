"""
Blender headless USDZ/USD -> GLB converter.

three.js can't read binary USDC (most .usdz), and usd2gltf is fragile on materials,
so the server shells out to Blender, whose USD importer + glTF exporter are robust:

    blender -b -P usdz_convert.py -- <input.usdz> <output.glb>
"""
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
in_path, out_path = argv[0], argv[1]

# Start from an empty scene (no default cube/camera/light in the export).
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import the USD/USDZ (textures inside the archive are handled by the importer).
bpy.ops.wm.usd_import(filepath=in_path, import_textures_mode='IMPORT_PACK')

# Export a single self-contained GLB.
bpy.ops.export_scene.gltf(
    filepath=out_path,
    export_format='GLB',
    export_yup=True,
)
