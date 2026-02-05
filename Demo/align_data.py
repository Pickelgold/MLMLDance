"""
Align Audio and Animation Data
Combines extracted audio features and blend animation data into aligned dataset

Usage:
    python align_data.py ./audio_output ./blend_output ./aligned_output
"""

import numpy as np
import json
from pathlib import Path
import argparse


class DataAligner:
    """Align audio and animation data for ML training"""
    
    def __init__(self):
        pass
    
    def load_audio_data(self, audio_dir):
        """Load extracted audio features"""
        audio_dir = Path(audio_dir)
        
        print("Loading audio features...")
        mel_spec = np.load(audio_dir / 'mel_spectrogram.npy')
        mfcc = np.load(audio_dir / 'mfcc.npy')
        
        with open(audio_dir / 'audio_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"  - Mel Spectrogram: {mel_spec.shape}")
        print(f"  - MFCC: {mfcc.shape}")
        print(f"  - FPS: {metadata['fps']}")
        print(f"  - Duration: {metadata['duration']:.2f}s")
        print(f"  - Frames: {metadata['n_frames']}")
        
        return {
            'mel_spectrogram': mel_spec,
            'mfcc': mfcc,
            'metadata': metadata
        }
    
    def load_blend_data(self, blend_dir):
        """Load extracted blend animation data"""
        blend_dir = Path(blend_dir)
        
        print("\nLoading animation data...")
        
        with open(blend_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"  - FPS: {metadata['scene_info']['fps']}")
        print(f"  - Duration: {metadata['scene_info']['duration']:.2f}s")
        print(f"  - Frames: {metadata['scene_info']['frame_count']}")
        
        # Load armature poses (main animation data)
        armature_files = list(blend_dir.glob('armature_*_poses.npy'))
        armatures = []
        
        for arm_file in sorted(armature_files):
            poses = np.load(arm_file)
            armatures.append(poses)
            print(f"  - {arm_file.name}: {poses.shape}")
        
        # Load object transforms if present
        objects = None
        if (blend_dir / 'object_transforms.npy').exists():
            objects = np.load(blend_dir / 'object_transforms.npy')
            print(f"  - object_transforms.npy: {objects.shape}")
        
        return {
            'armatures': armatures,
            'objects': objects,
            'metadata': metadata
        }
    
    def align(self, audio_data, blend_data, output_dir):
        """Align audio and animation data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("ALIGNING DATA")
        print("="*60)
        
        audio_fps = audio_data['metadata']['fps']
        audio_frames = audio_data['metadata']['n_frames']
        audio_duration = audio_data['metadata']['duration']
        
        anim_fps = blend_data['metadata']['scene_info']['fps']
        anim_frames = blend_data['metadata']['scene_info']['frame_count']
        anim_duration = blend_data['metadata']['scene_info']['duration']
        
        print(f"\nAudio:     {audio_frames} frames @ {audio_fps:.2f} fps ({audio_duration:.2f}s)")
        print(f"Animation: {anim_frames} frames @ {anim_fps} fps ({anim_duration:.2f}s)")
        
        # Check if fps match
        if abs(audio_fps - anim_fps) > 0.1:
            print(f"\n⚠️  WARNING: FPS mismatch!")
            print(f"   Audio: {audio_fps:.2f} fps")
            print(f"   Animation: {anim_fps} fps")
            print(f"   Please re-extract audio with --fps {anim_fps}")
            return None
        
        # Use the shorter duration
        min_frames = min(audio_frames, anim_frames)
        duration = min(audio_duration, anim_duration)
        
        print(f"\nAligned:   {min_frames} frames @ {anim_fps} fps ({duration:.2f}s)")
        
        if audio_frames != anim_frames:
            print(f"\n⚠️  Frame count mismatch - truncating to {min_frames} frames")
        
        # Truncate to matching length
        mel_spec_aligned = audio_data['mel_spectrogram'][:, :min_frames]
        mfcc_aligned = audio_data['mfcc'][:, :min_frames]
        
        # Truncate animation data
        armatures_aligned = []
        for armature in blend_data['armatures']:
            # armature shape: (n_bones, n_frames, 10)
            armatures_aligned.append(armature[:, :min_frames, :])
        
        objects_aligned = None
        if blend_data['objects'] is not None:
            objects_aligned = blend_data['objects'][:, :min_frames, :]
        
        # Save aligned data
        print("\nSaving aligned dataset...")
        
        # Audio features
        np.save(output_dir / 'audio_mel_spectrogram.npy', mel_spec_aligned)
        np.save(output_dir / 'audio_mfcc.npy', mfcc_aligned)
        print(f"  ✓ audio_mel_spectrogram.npy: {mel_spec_aligned.shape}")
        print(f"  ✓ audio_mfcc.npy: {mfcc_aligned.shape}")
        
        # Animation targets
        for idx, armature in enumerate(armatures_aligned):
            filename = f'target_armature_{idx}.npy'
            np.save(output_dir / filename, armature)
            print(f"  ✓ {filename}: {armature.shape}")
        
        if objects_aligned is not None:
            np.save(output_dir / 'target_objects.npy', objects_aligned)
            print(f"  ✓ target_objects.npy: {objects_aligned.shape}")
        
        # Create combined metadata
        alignment_metadata = {
            'fps': anim_fps,
            'duration': duration,
            'n_frames': min_frames,
            'audio_features': {
                'mel_spectrogram_shape': list(mel_spec_aligned.shape),
                'mfcc_shape': list(mfcc_aligned.shape),
                'n_mels': mel_spec_aligned.shape[0],
                'n_mfcc': mfcc_aligned.shape[0]
            },
            'animation_targets': {
                'n_armatures': len(armatures_aligned),
                'armature_shapes': [list(arm.shape) for arm in armatures_aligned],
                'has_objects': objects_aligned is not None
            },
            'source_metadata': {
                'audio': audio_data['metadata'],
                'animation': blend_data['metadata']
            }
        }
        
        with open(output_dir / 'alignment_metadata.json', 'w') as f:
            json.dump(alignment_metadata, f, indent=2)
        print(f"  ✓ alignment_metadata.json")
        
        # Print summary
        self._print_summary(alignment_metadata)
        
        return alignment_metadata
    
    def _print_summary(self, metadata):
        """Print alignment summary"""
        print("\n" + "="*60)
        print("ALIGNMENT SUMMARY")
        print("="*60)
        print(f"Dataset Duration: {metadata['duration']:.2f}s")
        print(f"Frame Rate: {metadata['fps']} fps")
        print(f"Total Frames: {metadata['n_frames']}")
        print()
        print("Input Features (Audio):")
        print(f"  - Mel Spectrogram: {metadata['audio_features']['mel_spectrogram_shape']}")
        print(f"  - MFCC: {metadata['audio_features']['mfcc_shape']}")
        print()
        print("Output Targets (Animation):")
        for idx, shape in enumerate(metadata['animation_targets']['armature_shapes']):
            print(f"  - Armature {idx}: {shape}")
            print(f"    → {shape[0]} bones × {shape[1]} frames × {shape[2]} features")
        print()
        print("Ready for ML training!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Align audio and animation data for ML training'
    )
    parser.add_argument('audio_dir', type=str, help='Directory with extracted audio features')
    parser.add_argument('blend_dir', type=str, help='Directory with extracted blend data')
    parser.add_argument('output_dir', type=str, help='Output directory for aligned data')
    
    args = parser.parse_args()
    
    # Create aligner
    aligner = DataAligner()
    
    # Load data
    audio_data = aligner.load_audio_data(args.audio_dir)
    blend_data = aligner.load_blend_data(args.blend_dir)
    
    # Align and save
    aligner.align(audio_data, blend_data, args.output_dir)
    
    print("\nAlignment complete!")
    print(f"Aligned dataset saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
