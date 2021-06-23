## Related Issue (if applicable)
Fixes #{issue number}

## Summary
What are you trying to achieve in this PR? Please summarize what changes you made and how they acheive the desired result.

## Changes checklist (if adding new augmentation)
If you are adding a new augmentation, please ensure that you have made the necessary changes in all of the following files, where `<module>` is the module (`audio`, `text`, or `image`) where you added the new augmentation.
- `augly/<module>/functional.py`
- `augly/<module>/transforms.py`
- `augly/<module>/intensity.py`
- `augly/<module>/__init__.py`
- `augly/tests/<module>_tests/functional_unit_tests.py`
- `augly/tests/<module>_tests/transforms_unit_tests.py`
- `augly/utils/expected_output/<module>/expected_metadata.json`
- for audio: `augly/assets/tests/audio/speech_commands_expected_output/{mono, stereo}/test_<aug_name>.wav`
- for image: `augly/assets/tests/image/dfdc_expected_output/test_<aug_name>.png`

Note about test files: The final file will be added, and should be the result of running the new augmentation with the args as specified in the new unit tests. The two new unit tests in `functional_unit_tests.py` & `transforms_unit_tests.py` should specify the same arguments for the new augmentation so they can use the same output file.

Note about new video augmentations: You may also need to add an augmenter file in `augly/video/augmenters/` if your augmentation uses `cv2` or `ffmpeg`. Instead of `functional_unit_tests.py` or `transforms_unit_tests`, you should choose which test file to add the unit test for your new augmentation to based on which category it best fits in to. If the augmentation is implemented using `cv2` or `ffmpeg`, you should add the unit test to `cv2_tests.py` or `ffmpeg_tests.py` respectively. If the augmentation is implemented by applying a function to each frame of the video, you should add the unit test to `image_based_tests.py`. If the augmentation is a combination of multiple video augmentations, you should add the unit test to `composite_tests.py`.

Please implement your new augmentation using AugLy's existing dependencies (i.e. `PIL`, `numpy`, `nlpaug`, `librosa`, etc.) if possible, and avoid adding new dependencies as these will make AugLy heavier and slower to download. However, if you feel it's necessary in order to implement your new augmentation and that the new augmentation is really worth having, it may be fine; in this case, add your new dependency to `requirements.txt`, and then make sure you can install `augly` in a fresh conda environment and the new unit tests pass.

If you want to see an example of what the changes should look like in each file, search all of the above files for one of the existing augmentations, e.g. `overlay_text`.

## Unit Tests
If your changes touch the `audio` module, please run all of the `audio` tests and paste the output here. Likewise for `image`, `text`, & `video`. If your changes could affect behavior in multiple modules, please run the tests for all potentially affected modules. If you are unsure of which modules might be affected by your changes, please just run all the unit tests.

### Audio
```bash
python -m unittest augly.tests.audio_tests.functional_unit_tests
python -m unittest augly.tests.audio_tests.transforms_unit_tests
```

### Image
```bash
python -m unittest augly.tests.image_tests.functional_unit_tests
python -m unittest augly.tests.image_tests.transforms_unit_tests
python -m unittest augly.tests.image_tests.pytorch_test  # Note: must have torchvision installed
```

### Text
```bash
python -m unittest augly.tests.text_tests.functional_unit_tests
python -m unittest augly.tests.text_tests.transforms_unit_tests
```

### Video
```bash
python -m unittest augly.tests.video_tests.transforms.composite_tests
python -m unittest augly.tests.video_tests.transforms.cv2_tests
python -m unittest augly.tests.video_tests.transforms.ffmpeg_tests
python -m unittest augly.tests.video_tests.transforms.image_based_tests
```

## Other testing

If applicable, test your changes and paste the output here. For example, if your changes affect the requirements/installation, then test installing augly in a fresh conda env, then make sure you are able to import augly & run the unit test
