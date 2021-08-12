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

If you're adding a new augmentation, please ensure that you have made all the necessary changes in the following files, where `<module>` is the module (`audio`, `text`, or `image`) where you added the new augmentation.
- `augly/<module>/functional.py`
- `augly/<module>/transforms.py`
- `augly/<module>/intensity.py`
- `augly/<module>/__init__.py`
- `augly/tests/<module>_tests/functional_unit_tests.py`
- `augly/tests/<module>_tests/transforms_unit_tests.py`
- `augly/utils/expected_output/<module>/expected_metadata.json`
- for audio: `augly/assets/tests/audio/speech_commands_expected_output/{mono, stereo}/test_<aug_name>.wav`
- for image: `augly/assets/tests/image/dfdc_expected_output/test_<aug_name>.png`

Note about test files: The final file will be added, and should be the result of running the new augmentation with the args as specified in the new unit tests. The two new unit tests in `functional_unit_tests.py` & `transforms_unit_tests.py` should specify the same `args` for the new augmentation so they can use the same output file.

Note about new video augmentations: You may also need to add an augmenter file in `augly/video/augmenters/` if your augmentation uses `cv2` or `ffmpeg`. Instead of `functional_unit_tests.py` or `transforms_unit_tests.py`, you should choose which test file to add the unit test for your new augmentation to based on which category it best fits in to. If the augmentation is implemented using `cv2` or `ffmpeg`, you should add the unit test to `cv2_tests.py` or `ffmpeg_tests.py` respectively. If the augmentation is implemented by applying a function to each frame of the video, you should add the unit test to `image_based_tests.py`. If the augmentation is a combination of multiple video augmentations, you should add the unit test to `composite_tests.py`.

Please implement your new augmentation using AugLy's existing dependencies (i.e. `PIL`, `numpy`, `nlpaug`, `librosa`, etc.) if possible, and avoid adding new dependencies as these will make AugLy heavier and slower to download. However, if you feel it's necessary in order to implement your new augmentation and that the new augmentation is really worth having, it may be fine; in this case, add your new dependency to `requirements.txt`, and then make sure you can install `augly` in a fresh conda environment and the new unit tests pass.

If you want to see an example of what the changes should look like in each file, search all of the above files for one of the existing augmentations, e.g. `overlay_text`.

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
