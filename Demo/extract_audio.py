"""
Simplified Audio Feature Extraction - Can run without Blender
Use this to test audio processing independently
"""

import numpy as np
import torchaudio
import torch
import json
from pathlib import Path


class AudioFeatureExtractor:
    """Extract features from a WAV file using torchaudio"""
    
    def __init__(self, sample_rate=48000, hop_length=512, n_mels=128, n_mfcc=13, target_fps=None):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.target_fps = target_fps  # If set, downsample to this fps
        
        # Initialize transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 2048,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
    
    def extract(self, wav_path):
        """Extract audio features from WAV file"""
        print(f"Loading audio from: {wav_path}")
        
        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            print(f"Resampling from {sr} Hz to {self.sample_rate} Hz")
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            print("Converting stereo to mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        duration = waveform.shape[1] / self.sample_rate
        print(f"Waveform shape: {waveform.shape}")
        print(f"Duration: {duration:.2f} seconds")
        
        # Extract features
        mel_spec = self.mel_transform(waveform)
        mfcc = self.mfcc_transform(waveform)
        
        # Convert to log scale for mel spectrogram
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Calculate energy/amplitude envelope
        amplitude = torch.sqrt(torch.mean(waveform ** 2, dim=0))
        
        # Get shapes before downsampling
        original_fps = self.sample_rate / self.hop_length
        n_frames_original = mel_spec.shape[-1]
        
        # Downsample to target fps if specified
        if self.target_fps is not None:
            print(f"\nDownsampling from {original_fps:.2f} fps to {self.target_fps} fps")
            mel_spec_db = self._downsample_features(mel_spec_db, original_fps, self.target_fps)
            mfcc = self._downsample_features(mfcc, original_fps, self.target_fps)
            n_frames = mel_spec_db.shape[-1]
            effective_fps = self.target_fps
        else:
            n_frames = n_frames_original
            effective_fps = original_fps
        
        features = {
            'waveform': waveform.squeeze().numpy(),
            'mel_spectrogram': mel_spec_db.squeeze().numpy(),  # (n_mels, time)
            'mfcc': mfcc.squeeze().numpy(),  # (n_mfcc, time)
            'amplitude': amplitude.numpy(),
            'sample_rate': self.sample_rate,
            'hop_length': self.hop_length,
            'duration': duration,
            'n_frames': n_frames,
            'fps': effective_fps,
            'original_n_frames': n_frames_original,
            'original_fps': original_fps
        }
        
        print(f"\nExtracted features:")
        if self.target_fps is not None:
            print(f"  - Original frames: {n_frames_original} @ {original_fps:.2f} fps")
            print(f"  - Downsampled frames: {n_frames} @ {effective_fps} fps")
        print(f"  - Mel Spectrogram: {features['mel_spectrogram'].shape}")
        print(f"  - MFCC: {features['mfcc'].shape}")
        
        return features
    
    def _downsample_features(self, features, original_fps, target_fps):
        """Downsample features to target fps using interpolation"""
        # features shape: (channels, time) for 2D or (time,) for 1D
        
        if isinstance(features, torch.Tensor):
            features_np = features.squeeze().numpy()
        else:
            features_np = features
        
        # Handle both 1D and 2D features
        is_2d = len(features_np.shape) == 2
        
        if is_2d:
            n_channels, n_frames_original = features_np.shape
        else:
            n_frames_original = features_np.shape[0]
        
        # Calculate target number of frames
        duration = n_frames_original / original_fps
        n_frames_target = int(duration * target_fps)
        
        # Interpolate
        if is_2d:
            downsampled = np.zeros((n_channels, n_frames_target))
            for i in range(n_channels):
                downsampled[i] = np.interp(
                    np.linspace(0, n_frames_original - 1, n_frames_target),
                    np.arange(n_frames_original),
                    features_np[i]
                )
        else:
            downsampled = np.interp(
                np.linspace(0, n_frames_original - 1, n_frames_target),
                np.arange(n_frames_original),
                features_np
            )
        
        return torch.from_numpy(downsampled) if isinstance(features, torch.Tensor) else downsampled
    
    def visualize_features(self, features, output_path=None):
        """Optional: Create a visualization of the extracted features"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot waveform
            time = np.arange(len(features['waveform'])) / features['sample_rate']
            axes[0].plot(time, features['waveform'])
            axes[0].set_title('Waveform')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Amplitude')
            
            # Plot mel spectrogram
            img1 = axes[1].imshow(
                features['mel_spectrogram'],
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            axes[1].set_title('Mel Spectrogram (dB)')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Mel Frequency')
            plt.colorbar(img1, ax=axes[1])
            
            # Plot MFCC
            img2 = axes[2].imshow(
                features['mfcc'],
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            axes[2].set_title('MFCC')
            axes[2].set_xlabel('Time')
            axes[2].set_ylabel('MFCC Coefficient')
            plt.colorbar(img2, ax=axes[2])
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                print(f"\nVisualization saved to: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for visualization")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract audio features from WAV file')
    parser.add_argument('wav_file', type=str, help='Path to WAV file')
    parser.add_argument('--output_dir', type=str, default='./audio_output', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--fps', type=int, default=None, help='Target FPS for downsampling (e.g., 30 to match animation)')
    parser.add_argument('--sample_rate', type=int, default=48000, help='Audio sample rate (default: 48000)')
    
    args = parser.parse_args()
    
    wav_path = Path(args.wav_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Audio Feature Extraction Demo")
    if args.fps:
        print(f"Target FPS: {args.fps}")
    print("=" * 60)
    print()
    
    # Extract features
    extractor = AudioFeatureExtractor(
        sample_rate=args.sample_rate,
        target_fps=args.fps
    )
    features = extractor.extract(wav_path)
    
    # Save features
    print("\nSaving features...")
    np.save(output_dir / 'mel_spectrogram.npy', features['mel_spectrogram'])
    np.save(output_dir / 'mfcc.npy', features['mfcc'])
    np.save(output_dir / 'waveform.npy', features['waveform'])
    
    # Save metadata
    metadata = {
        'duration': float(features['duration']),
        'sample_rate': features['sample_rate'],
        'hop_length': features['hop_length'],
        'n_frames': features['n_frames'],
        'fps': float(features['fps']),
        'mel_shape': list(features['mel_spectrogram'].shape),
        'mfcc_shape': list(features['mfcc'].shape)
    }
    
    if 'original_n_frames' in features:
        metadata['original_n_frames'] = features['original_n_frames']
        metadata['original_fps'] = float(features['original_fps'])
    
    with open(output_dir / 'audio_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nFeatures saved to: {output_dir}")
    
    # Visualize if requested
    if args.visualize:
        extractor.visualize_features(features, output_dir / 'audio_features.png')
    
    print("\nDone!")


if __name__ == '__main__':
    main()
