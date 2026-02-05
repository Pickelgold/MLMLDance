#!/bin/bash

python extract_audio.py pv_636.wav
blender --background --python extract_blend.py -- pv_636.blend  # You must have blender installed and use blender python in order to import bpy
python align_data.py ./audio_output/ ./blend_output/ ./aligned_dataset

