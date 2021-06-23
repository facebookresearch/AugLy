# New Augmentation (video)

## Related Issue (if applicable)
Fixes #{issue number}

## Summary
What does your new augmentation do at a high level? Are you sure it is adding new behavior to AugLy and is not covered by any of the existing augmentations?

## Changes checklist
If you are adding a new augmentation, please ensure that you have made the necessary changes in all of the following files:
- `augly/video/functional.py`
- `augly/video/transforms.py`
- `augly/video/intensity.py`
- `augly/video/__init__.py`
- `augly/tests/video_tests/transforms/<relevant_test_file>.py`
- `augly/utils/expected_output/video_tests/expected_metadata.json`

You may also need to add an augmenter file in `augly/video/augmenters/` if your augmentation uses `cv2` or `ffmpeg`.

Please implement your new augmentation using AugLy's existing dependencies (i.e. `PIL`, `numpy`, `nlpaug`, `librosa`, etc) if possible, and avoid adding new dependencies as these will make AugLy heavier and slower to download. However, if you feel it's necessary in order to implement your new augmentation and that the new augmentation is really worth having, it may be fine; in this case, add your new dependency to `requirements.txt`, and then make sure you can install `augly` in a fresh conda environment and the new unit test passes.

Note: You should choose which test file to add the unit test for your new augmentation to based on which category it best fits in to. If the augmentation is implemented using `cv2` or `ffmpeg`, you should add the unit test to `cv2_tests.py` or `ffmpeg_tests.py` respectively. If the augmentation is implemented by applying a function to each frame of the video, you should add the unit test to `image_based_tests.py`. If the augmentation does some kind of overlay or spatial distortion, you should add the unit test to `composite_tests.py`.

If you want to see an example of what the changes should look like in each file, search all of the above files for one of the existing augmentations, e.g. `overlay_text`.

## Unit Tests
Please run the video unit test file where you added the test for your new augmentation, one of the following, and paste the output here.

```bash
python -m unittest augly.tests.video_tests.transforms.composite_tests
python -m unittest augly.tests.video_tests.transforms.cv2_tests
python -m unittest augly.tests.video_tests.transforms.ffmpeg_tests
python -m unittest augly.tests.video_tests.transforms.image_based_tests
```