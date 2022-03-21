# Video

## Installation
If you would like to use the video augmentations, please install AugLy using the following command:
```bash
pip install augly[video]
```

This ensures that not only the base dependencies, but also the heavier dependencies required for audio & video processing, are installed.

In order to run the video tests and/or use the augmentations, you will also need to install `ffmpeg`. If you're using conda you can do this with:
```bash
conda install -c conda-forge ffmpeg
```

If you aren't using conda, you can run:
```bash
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
apt install ffmpeg
```

AugLy Video uses `distutils.spawn.find_executable()` to automatically find your `ffmpeg` & `ffprobe` paths. If you would prefer to use another specific version of `ffmpeg`, export the `AUGLY_FFMPEG_PATH` and `AUGLY_FFPROBE_PATH` environment variables such that we can access the intended `ffmpeg` version:
```bash
which ffmpeg
export AUGLY_FFMPEG_PATH='<ffmpeg_path>'
which ffprobe
export AUGLY_FFPROBE_PATH='<ffprobe_path>'
```

## Augmentations

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/AugLy/blob/main/examples/AugLy_video.ipynb)

Try running some AugLy video augmentations in Colab! For a full list of available augmentations, see [here](__init__.py).

Our video augmentations use `ffmpeg` & OpenCV as their backend. All functions accept a path to the video to be augmented as input and output a file containing the augmented video. If no output path is specified, the original video input path will be overwritten. The file path to which the augmented video was written will also be returned by all augmentations.

### Function-based

You can call the functional augmentations like so:
```python
import augly.video as vidaugs

video_path = "your_vid_path.mp4"
output_path = "your_output_path.mp4"

vidaugs.add_dog_filter(video_path, output_path)
vidaugs.rotate(output_path, degrees=30)   # output_path will be overwritten
```

### Class-based

We have also defined class-based versions of all augmentations, as well as a Compose operator used to combine transforms.
```python
import augly.video as vidaugs

COLOR_JITTER_PARAMS = {
    "brightness_factor": 0.15,
    "contrast_factor": 1.3,
    "saturation_factor": 2.0,
}

AUGMENTATIONS = [
    vidaugs.ColorJitter(**COLOR_JITTER_PARAMS),
    vidaugs.HorizontalFlip(),
    vidaugs.OneOf(
        [
            vidaugs.RandomEmojiOverlay(),
            vidaugs.RandomIGFilter(),
            vidaugs.Shift(x_factor=0.25, y_factor=0.25),
        ]
    ),
]

TRANSFORMS = vidaugs.Compose(AUGMENTATIONS)

video_path = "your_vid_path.mp4"
out_video_path = "your_output_path.mp4"

TRANSFORMS(video_path, out_video_path)  # transformed video now stored in `out_video_path`
```

## Unit Tests

You can run our video unit tests if you have cloned `augly` (see [here](../../README.md)) by running the following:
```bash
python -m unittest discover -s augly/tests/video_tests/ -p "*"
```

Note: some of the video tests take a while to run (up to a few minutes). If you want to run the 4 test suites individually, you can run any of the following commands (listed in order of increasing runtime):
```bash
python -m unittest augly.tests.video_tests.transforms.composite_test
python -m unittest augly.tests.video_tests.transforms.cv2_test
python -m unittest augly.tests.video_tests.transforms.ffmpeg_test
python -m unittest augly.tests.video_tests.transforms.image_based_test
```
