"""
Blender Animation Data Extractor
Run this script through Blender to extract animation data from a .blend file

Usage:
    blender -b -P extract_blend.py -- /path/to/animation.blend ./output_dir
"""

import bpy
import sys
import json
import numpy as np
from pathlib import Path
from mathutils import Vector, Euler, Quaternion


class BlenderAnimationExtractor:
    """Extract animation data from a .blend file"""
    
    def __init__(self, fps=None):
        self.fps = fps
    
    def extract(self, blend_path, output_dir):
        """Extract all pertinent animation data from Blender file"""
        print(f"\n{'='*60}")
        print(f"Loading Blender file: {blend_path}")
        print(f"{'='*60}\n")
        
        # Open the blend file
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        
        scene = bpy.context.scene
        fps = self.fps if self.fps else scene.render.fps
        frame_start = scene.frame_start
        frame_end = scene.frame_end
        frame_count = frame_end - frame_start + 1
        duration = frame_count / fps
        
        print(f"Scene Information:")
        print(f"  FPS: {fps}")
        print(f"  Frame range: {frame_start} to {frame_end} ({frame_count} frames)")
        print(f"  Duration: {duration:.2f} seconds")
        print()
        
        # Collect all data
        extraction_data = {
            'scene_info': {
                'fps': fps,
                'frame_start': frame_start,
                'frame_end': frame_end,
                'frame_count': frame_count,
                'duration': duration
            },
            'objects': [],
            'cameras': [],
            'lights': [],
            'armatures': []
        }
        
        # Scan all objects
        print(f"Found {len(bpy.data.objects)} objects in scene")
        print()
        
        for obj in bpy.data.objects:
            print(f"Processing: {obj.name} (Type: {obj.type})")
            
            # Categorize and extract based on type
            if obj.type == 'CAMERA':
                camera_data = self._extract_camera(obj, frame_start, frame_end)
                extraction_data['cameras'].append(camera_data)
                
            elif obj.type == 'LIGHT':
                light_data = self._extract_light(obj, frame_start, frame_end)
                extraction_data['lights'].append(light_data)
                
            elif obj.type == 'ARMATURE':
                armature_data = self._extract_armature(obj, frame_start, frame_end)
                extraction_data['armatures'].append(armature_data)
                
            else:
                # Regular object (MESH, EMPTY, etc.)
                obj_data = self._extract_object(obj, frame_start, frame_end)
                extraction_data['objects'].append(obj_data)
        
        print()
        
        # Save all data
        self._save_data(extraction_data, output_dir)
        
        # Print summary
        self._print_summary(extraction_data)
        
        return extraction_data
    
    def _extract_object(self, obj, frame_start, frame_end):
        """Extract data for a regular object"""
        obj_data = {
            'name': obj.name,
            'type': obj.type,
            'has_animation': False,
            'transforms': [],
            'fcurves': []
        }
        
        # Check for animation data
        has_fcurves = obj.animation_data and obj.animation_data.action
        
        if has_fcurves:
            obj_data['has_animation'] = True
            # Extract F-Curves
            for fcurve in obj.animation_data.action.fcurves:
                fcurve_info = {
                    'data_path': fcurve.data_path,
                    'array_index': fcurve.array_index,
                    'num_keyframes': len(fcurve.keyframe_points)
                }
                obj_data['fcurves'].append(fcurve_info)
                print(f"  - FCurve: {fcurve.data_path}[{fcurve.array_index}] ({len(fcurve.keyframe_points)} keyframes)")
        
        # Always extract transforms frame-by-frame
        scene = bpy.context.scene
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            
            # World space transform
            matrix = obj.matrix_world
            loc = matrix.translation
            rot_euler = matrix.to_euler('XYZ')
            rot_quat = matrix.to_quaternion()
            scale = matrix.to_scale()
            
            transform = {
                'frame': frame,
                'location': [loc.x, loc.y, loc.z],
                'rotation_euler': [rot_euler.x, rot_euler.y, rot_euler.z],
                'rotation_quat': [rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z],
                'scale': [scale.x, scale.y, scale.z]
            }
            
            obj_data['transforms'].append(transform)
        
        if not has_fcurves:
            print(f"  - No animation curves, extracted {len(obj_data['transforms'])} transform frames")
        
        return obj_data
    
    def _extract_camera(self, camera_obj, frame_start, frame_end):
        """Extract camera-specific data"""
        camera_data = self._extract_object(camera_obj, frame_start, frame_end)
        
        # Add camera-specific properties
        cam = camera_obj.data
        camera_data['camera_properties'] = {
            'type': cam.type,
            'lens': cam.lens if cam.type == 'PERSP' else None,
            'sensor_width': cam.sensor_width,
            'clip_start': cam.clip_start,
            'clip_end': cam.clip_end
        }
        
        print(f"  - Camera lens: {cam.lens}mm")
        
        return camera_data
    
    def _extract_light(self, light_obj, frame_start, frame_end):
        """Extract light-specific data"""
        light_data = self._extract_object(light_obj, frame_start, frame_end)
        
        # Add light-specific properties
        light = light_obj.data
        light_data['light_properties'] = {
            'type': light.type,
            'energy': light.energy,
            'color': [light.color.r, light.color.g, light.color.b]
        }
        
        print(f"  - Light type: {light.type}, Energy: {light.energy}")
        
        return light_data
    
    def _extract_armature(self, armature_obj, frame_start, frame_end):
        """Extract armature (skeleton) data including bone animations"""
        armature_data = self._extract_object(armature_obj, frame_start, frame_end)
        
        armature = armature_obj.data
        armature_data['bones'] = []
        
        print(f"  - Armature with {len(armature.bones)} bones")
        
        # Extract bone data
        for bone in armature.bones:
            bone_info = {
                'name': bone.name,
                'parent': bone.parent.name if bone.parent else None,
                'head': [bone.head_local.x, bone.head_local.y, bone.head_local.z],
                'tail': [bone.tail_local.x, bone.tail_local.y, bone.tail_local.z],
                'poses': []
            }
            
            # Extract pose bone animation
            if armature_obj.pose:
                pose_bone = armature_obj.pose.bones.get(bone.name)
                if pose_bone:
                    scene = bpy.context.scene
                    for frame in range(frame_start, frame_end + 1):
                        scene.frame_set(frame)
                        
                        # Get pose bone transform
                        matrix = pose_bone.matrix
                        loc = matrix.to_translation()
                        rot_quat = matrix.to_quaternion()
                        scale = matrix.to_scale()
                        
                        pose = {
                            'frame': frame,
                            'location': [loc.x, loc.y, loc.z],
                            'rotation': [rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z],
                            'scale': [scale.x, scale.y, scale.z]
                        }
                        
                        bone_info['poses'].append(pose)
            
            armature_data['bones'].append(bone_info)
            print(f"    - Bone: {bone.name}")
        
        return armature_data
    
    def _save_data(self, data, output_dir):
        """Save extracted data in multiple formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nSaving data to: {output_dir}")
        
        # Save complete JSON
        with open(output_dir / 'animation_complete.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ animation_complete.json")
        
        # Save as numpy arrays for ML
        self._save_numpy_arrays(data, output_dir)
        
        # Save metadata separately
        metadata = {
            'scene_info': data['scene_info'],
            'object_count': len(data['objects']),
            'camera_count': len(data['cameras']),
            'light_count': len(data['lights']),
            'armature_count': len(data['armatures']),
            'object_names': [obj['name'] for obj in data['objects']],
            'camera_names': [cam['name'] for cam in data['cameras']],
            'armature_names': [arm['name'] for arm in data['armatures']]
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ metadata.json")
    
    def _save_numpy_arrays(self, data, output_dir):
        """Convert animation data to numpy arrays for ML training"""
        
        # Extract all object transforms into a single array
        all_transforms = []
        object_info = []
        
        for obj in data['objects']:
            if obj['transforms']:
                # Stack location, rotation_euler, scale for each frame
                obj_array = []
                for transform in obj['transforms']:
                    frame_data = (
                        transform['location'] + 
                        transform['rotation_euler'] + 
                        transform['scale']
                    )  # 3 + 3 + 3 = 9 values per frame
                    obj_array.append(frame_data)
                
                all_transforms.append(obj_array)
                object_info.append({
                    'name': obj['name'],
                    'type': obj['type'],
                    'index': len(object_info)
                })
        
        if all_transforms:
            # Shape: (num_objects, num_frames, 9)
            transforms_array = np.array(all_transforms)
            np.save(output_dir / 'object_transforms.npy', transforms_array)
            print(f"  ✓ object_transforms.npy - Shape: {transforms_array.shape}")
            
            with open(output_dir / 'object_info.json', 'w') as f:
                json.dump(object_info, f, indent=2)
        
        # Extract armature bone poses
        for arm_idx, armature in enumerate(data['armatures']):
            bone_poses = []
            bone_info = []
            
            for bone in armature['bones']:
                if bone['poses']:
                    bone_array = []
                    for pose in bone['poses']:
                        frame_data = (
                            pose['location'] + 
                            pose['rotation'] +  # quaternion (4 values)
                            pose['scale']
                        )  # 3 + 4 + 3 = 10 values per frame
                        bone_array.append(frame_data)
                    
                    bone_poses.append(bone_array)
                    bone_info.append({
                        'name': bone['name'],
                        'parent': bone['parent'],
                        'index': len(bone_info)
                    })
            
            if bone_poses:
                # Shape: (num_bones, num_frames, 10)
                poses_array = np.array(bone_poses)
                np.save(output_dir / f'armature_{arm_idx}_poses.npy', poses_array)
                print(f"  ✓ armature_{arm_idx}_poses.npy - Shape: {poses_array.shape}")
                
                with open(output_dir / f'armature_{arm_idx}_bone_info.json', 'w') as f:
                    json.dump(bone_info, f, indent=2)
        
        # Extract camera transforms if any
        if data['cameras']:
            cam_transforms = []
            for camera in data['cameras']:
                if camera['transforms']:
                    cam_array = []
                    for transform in camera['transforms']:
                        frame_data = (
                            transform['location'] + 
                            transform['rotation_euler'] + 
                            transform['scale']
                        )
                        cam_array.append(frame_data)
                    cam_transforms.append(cam_array)
            
            if cam_transforms:
                cam_array = np.array(cam_transforms)
                np.save(output_dir / 'camera_transforms.npy', cam_array)
                print(f"  ✓ camera_transforms.npy - Shape: {cam_array.shape}")
    
    def _print_summary(self, data):
        """Print extraction summary"""
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Scene Duration: {data['scene_info']['duration']:.2f}s @ {data['scene_info']['fps']} fps")
        print(f"Total Frames: {data['scene_info']['frame_count']}")
        print()
        print(f"Objects Extracted:")
        print(f"  - Regular objects: {len(data['objects'])}")
        print(f"  - Cameras: {len(data['cameras'])}")
        print(f"  - Lights: {len(data['lights'])}")
        print(f"  - Armatures: {len(data['armatures'])}")
        
        # Count animated vs static
        animated = sum(1 for obj in data['objects'] if obj['has_animation'])
        print()
        print(f"Animation Status:")
        print(f"  - Animated objects: {animated}")
        print(f"  - Static objects: {len(data['objects']) - animated}")
        
        # Bone count
        total_bones = sum(len(arm['bones']) for arm in data['armatures'])
        if total_bones > 0:
            print(f"  - Total bones: {total_bones}")
        
        print(f"{'='*60}\n")


def main():
    """Main extraction function"""
    
    # Parse arguments
    try:
        dash_index = sys.argv.index('--')
        args = sys.argv[dash_index + 1:]
    except ValueError:
        args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if len(args) < 1:
        print("Usage: blender -b -P extract_blend.py -- <blend_file> [output_dir] [--fps 24]")
        print()
        print("Examples:")
        print("  blender -b -P extract_blend.py -- animation.blend ./output")
        print("  blender -b -P extract_blend.py -- animation.blend ./output --fps 30")
        sys.exit(1)
    
    blend_path = Path(args[0])
    output_dir = Path(args[1]) if len(args) > 1 else Path('./blend_output')
    
    # Check for fps override
    fps = None
    if '--fps' in args:
        fps_index = args.index('--fps')
        if fps_index + 1 < len(args):
            fps = int(args[fps_index + 1])
    
    if not blend_path.exists():
        print(f"ERROR: Blend file not found: {blend_path}")
        sys.exit(1)
    
    # Extract data
    extractor = BlenderAnimationExtractor(fps=fps)
    extractor.extract(blend_path, output_dir)
    
    print("Extraction complete!")


if __name__ == '__main__':
    main()
