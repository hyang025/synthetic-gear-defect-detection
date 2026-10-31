"""Encode the already-rendered PNG sequence with Blender's bundled FFmpeg.

./tools/blender --background --factory-startup --python healthy/encode_video.py -- outputs/full_revolution
"""

import json
from pathlib import Path
import sys

import bpy

output = Path(sys.argv[sys.argv.index("--")+1]).resolve()
config = json.loads((output / "config.json").read_text())
manifest = json.loads((output / "manifest.json").read_text())
files = [f"frame_{frame['index']:03d}.png" for frame in manifest["frames"]]
holds = config.get("animation", {}).get("extra_hold_frames", {})
files = [filename for frame, filename in zip(manifest["frames"], files)
         for _ in range(1+int(holds.get(str(frame["index"]), 0)))]
frame_dir = output / "frames"
if not files or any(not (frame_dir / name).is_file() for name in files):
    raise SystemExit("Numbered PNG sequence is missing or incomplete in frames/.")
(output / "animations").mkdir(exist_ok=True)
scene = bpy.context.scene
scene.render.resolution_x = config.get("animation", {}).get("width_px", 640)
scene.render.resolution_y = config.get("animation", {}).get("height_px", 448)
scene.render.resolution_percentage = 100
scene.render.fps = config.get("animation", {}).get("fps", 25)
scene.frame_start = 1
scene.frame_end = len(files)
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "LOW"
scene.render.ffmpeg.ffmpeg_preset = "GOOD"
scene.render.ffmpeg.audio_codec = "NONE"
scene.render.filepath = str(output / "animations" / config.get("animation", {}).get("filename", "healthy_rotation.mp4"))
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1
scene.sequencer_colorspace_settings.name = "sRGB"
sequence = scene.sequence_editor_create()
strip = sequence.strips.new_image("Rendered sequence", str(frame_dir / files[0]), channel=1, frame_start=1)
for filename in files[1:]:
    strip.elements.append(filename)
strip.frame_final_duration = len(files)
bpy.ops.render.render(animation=True)
print(f"Encoded {len(files)} frames: {scene.render.filepath}")
