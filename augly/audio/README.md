# Audio

## Installation

If you would like to use the audio augmentations, please install AugLy using the following command:
```bash
pip install augly[audio]
```

This ensures that not only the base dependencies, but also the heavier dependencies required for audio & video processing, are installed.

## Augmentations

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/AugLy/blob/main/examples/AugLy_audio.ipynb)

Try running some AugLy audio augmentations in Colab! For a full list of available augmentations, see [here](__init__.py).

Our audio augmentations use `librosa`, `torchaudio`, and NumPy as their backend. All functions accept an audio path or an audio array to be augmented as input and return the augmented audio array. If an output path is specified, the audio will also be saved to a file.

### Function-based

You can call the functional augmentations like so:
```python
import augly.audio as audaugs

audio_path = "your_audio_path.flac"
output_path = "your_output_path.flac"

# Augmentation functions can accept audio paths as input and
# always return the resulting augmented audio array & sample rate
aug_audio, sample_rate = audaugs.change_volume(audio_path, volume_db=10.0)

# Augmentation functions can also accept np.ndarray as input
# (but then you have to provide the sample rate, too, since it
# won't be inferred when loading the audio file)
aug_audio, sample_rate = audaugs.low_pass_filter(
    aug_audio,
    sample_rate=sample_rate,
    cutoff_hz=500,
)

# If an output path is specified, the audio will also be saved to a file
aug_audio, sample_rate = audaugs.normalize(
    aug_audio,
    sample_rate=sample_rate,
    output_path=output_path,
)
```

### Class-based

You can also call any augmentation as a Transform class, including composing them together and applying them with a given probability:
```python
TRANSFORMS = audaugs.Compose([
    audaugs.Clip(duration_factor=0.25),
    audaugs.ChangeVolume(volume_db=10.0, p=0.5),
    audaugs.OneOf(
        [audaugs.Speed(factor=3.0), audaugs.TimeStretch(rate=3.0)]
    ),
])

# aug_audio is a NumPy array with your augmentations applied!
audio_array = librosa.load("your_audio_path.flac", sr=None, mono=False)
aug_audio = TRANSFORMS(audio_array)
```

## Unit Tests

You can run our audio unit tests if you have cloned `augly` (see [here](../../README.md)) by running the following:
```bash
python -m unittest discover -s augly/tests/audio_tests/ -p "*"
```
