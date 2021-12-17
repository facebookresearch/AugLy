# Contributing to AugLy
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Run `black` formatting.
7. If you haven't already, complete the Contributor License Agreement ("CLA").

## Adding a New Augmentation

Before adding a new augmentation, please open an associated issue such that we could ensure that (a) this functionality doesn't already exist in the library (b) it is in line with the kinds of augmentations we'd like to support.

Please implement your new augmentation using AugLy's existing dependencies (i.e. `PIL`, `numpy`, `nlpaug`, `librosa`, etc.) if possible, and **avoid adding new dependencies** as these will make AugLy heavier and slower to download. However, if you feel it's necessary in order to implement your new augmentation and that the new augmentation is really worth having, it may be fine; in this case, add your new dependency to `requirements.txt`, and then make sure you can install `augly` in a fresh conda environment and the new unit tests pass.

Before submitting a PR with your new augmentation, please ensure that you have made all the necessary changes in the listed files below for the corresponding module where you're adding the augmentation. Whenever adding a new function, class, etc please make sure to insert it in **alphabetical order** in the file!

### Audio
- `augly/audio/functional.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/audio/functional.py#L929-L981) for `pitch_shift`)
- `augly/audio/transforms.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/audio/transforms.py#L687-L716) for `pitch_shift`)
- `augly/audio/intensity.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/audio/intensity.py#L142-L145) for `pitch_shift`)
- `augly/audio/__init__.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/audio/__init__.py#L24) for `pitch_shift`)
- `augly/tests/audio_tests/functional_unit_test.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/tests/audio_tests/functional_unit_test.py#L61-L62) for `pitch_shift`)
- `augly/tests/audio_tests/transforms_unit_test.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/tests/audio_tests/transforms_unit_test.py#L125-L126) for `pitch_shift`)
- `augly/utils/expected_output/audio/expected_metadata.json` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/utils/expected_output/audio_tests/expected_metadata.json#L515-L540) for `pitch_shift`)
- `augly/assets/tests/audio/speech_commands_expected_output/{mono, stereo}/test_<aug_name>.wav` (e.g. [mono](https://github.com/facebookresearch/AugLy/blob/main/augly/assets/tests/audio/speech_commands_expected_output/mono/test_pitch_shift.wav) & [stereo](https://github.com/facebookresearch/AugLy/blob/main/augly/assets/tests/audio/speech_commands_expected_output/stereo/test_pitch_shift.wav) files for `pitch_shift`)
  - These test files should be the result of running the new augmentation with the args as specified in the new unit tests. The two new unit tests should specify the same args for the new augmentation so they can use the same output file.

### Image
- `augly/image/functional.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/image/functional.py#L568-L638) for `crop`)
- `augly/image/transforms.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/image/transforms.py#L666-L729) for `crop`)
- `augly/image/intensity.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/image/intensity.py#L103-L104) for `crop`)
- `augly/image/__init__.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/image/__init__.py#L19) for `crop`)
- `augly/tests/image_tests/functional_unit_test.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/tests/image_tests/functional_unit_test.py#L44-L45) for `crop`)
- `augly/tests/image_tests/transforms_unit_test.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/tests/image_tests/transforms_unit_test.py#L81-L82) for `crop`)
- `augly/utils/expected_output/image/expected_metadata.json` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/utils/expected_output/image_tests/expected_metadata.json#L192-L209) for `crop`)
- `augly/assets/tests/image/dfdc_expected_output/test_<aug_name>.png` (e.g. [test file](https://github.com/facebookresearch/AugLy/blob/main/augly/assets/tests/image/dfdc_expected_output/test_crop.png) for `crop`)
  - This test file should be the result of running the new augmentation with the args as specified in the new unit tests. The two new unit tests should specify the same args for the new augmentation so they can use the same output file.
- `augly/image/utils/bboxes.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/image/utils/bboxes.py#L16-L31) for `crop`)
  - You should ensure that bounding boxes will be correctly transformed with the images for your new augmentation by either (1) adding a new helper function to `bboxes.py` which computes how a given bounding box will be changed by your augmentation (see the one for [`crop`](https://github.com/facebookresearch/AugLy/blob/main/augly/image/utils/bboxes.py#L16-L31) as an example); (2) using the `spatial_bbox_helper` if your augmentation is spatial and does not cause any color changes or overlay non-black content (see [`skew`](https://github.com/facebookresearch/AugLy/blob/main/augly/image/functional.py#L2430) call it as an example); or (3) doing nothing if your new augmentation does not move or occlude pixels at all from the original image content (e.g. [`color_jitter`](https://github.com/facebookresearch/AugLy/blob/main/augly/image/functional.py#L374) or [`sharpen`](https://github.com/facebookresearch/AugLy/blob/main/augly/image/functional.py#L2246)).

### Text
- `augly/text/functional.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/text/functional.py#L819-L867) for `split_words`)
- `augly/text/transforms.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/text/transforms.py#L972-L1027) for `split_words`)
- `augly/text/intensity.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/text/intensity.py#L103-L104) for `split_words`)
- `augly/text/__init__.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/text/__init__.py#L25) for `split_words`)
- `augly/tests/text_tests/functional_unit_test.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/tests/text_tests/functional_unit_test.py#L445-L463) for `split_words`)
- `augly/tests/text_tests/transforms_unit_test.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/tests/text_tests/transforms_unit_test.py#L324-L335) for `split_words`)
- `augly/utils/expected_output/text/expected_metadata.json` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/utils/expected_output/text_tests/expected_metadata.json#L262-L276) for `split_words`)

### Video
- `augly/video/functional.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/video/functional.py#L1111-L1179) for `overlay`)
- `augly/video/transforms.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/video/transforms.py#L1071-L1137) for `overlay`)
- `augly/video/__init__.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/video/__init__.py#L31) for `overlay`)
- `augly/video/helpers/intensity.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/video/helpers/intensity.py#L177-L193) for `overlay`)
- `augly/video/helpers/__init__.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/video/helpers/__init__.py#L40) for `overlay`)
- `augly/tests/video_tests/transforms/<aug_type>.py` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/tests/video_tests/transforms/ffmpeg_test.py#L96-L98) for `overlay`)
  - Instead of `functional_unit_test.py` or `transforms_unit_test.py`, you should choose which test file to add the unit test for your new augmentation to based on which category it best fits in to. If the augmentation is implemented using `cv2` or `ffmpeg`, you should add the unit test to `cv2_test.py` or `ffmpeg_test.py` respectively. If the augmentation is implemented by applying a function to each frame of the video, you should add the unit test to `image_based_test.py`. If the augmentation is a combination of multiple video augmentations, you should add the unit test to `composite_test.py`.
- `augly/utils/expected_output/image/expected_metadata.json` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/utils/expected_output/video_tests/expected_metadata.json#L691-L721) for `overlay`
- You may also need to add an augmenter file in `augly/video/augmenters/` if your augmentation uses `cv2` or `ffmpeg` (e.g. [changes](https://github.com/facebookresearch/AugLy/blob/main/augly/video/augmenters/ffmpeg/overlay.py) for `overlay`)

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
* 4 spaces for indentation rather than tabs
* 90 character line length
* 2 newlines before a function or class definition. For functions within classes, use a single newline
* Add typing to your function parameters and add a return type
* Add a newline after an if/else block

## License
By contributing to AugLy, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
