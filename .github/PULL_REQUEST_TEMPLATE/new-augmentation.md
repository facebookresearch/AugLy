# New Augmentation (audio/image/text)

## Related Issue (if applicable)
Fixes #{issue number}

## Summary
What does your new augmentation do at a high level? Are you sure it is adding new behavior to AugLy and is not covered by any of the existing augmentations?

## Changes checklist
If you are adding a new augmentation, please ensure that you have made the necessary changes in all of the following files, where `<module>` is the module (`audio`, `text`, or `image`) where you added the new augmentation.
- `augly/<module>/functional.py`
- `augly/<module>/transforms.py`
- `augly/<module>/intensity.py`
- `augly/<module>/__init__.py`
- `augly/tests/<module>_tests/functional_unit_tests.py`
- `augly/tests/<module>_tests/transforms_unit_tests.py`
- `augly/utils/expected_output/<module>/expected_metadata.json`
- `augly/assets/tests/<module>_tests/test_<aug_name>.py`

Note: The final file will be added, and should be the result of running the new augmentation with the args as specified in the new unit tests. The two new unit tests in `functional_unit_tests.py` & `transforms_unit_tests.py` should specify the same arguments for the new augmentation so they can use the same output file.

Please implement your new augmentation using AugLy's existing dependencies (i.e. `PIL`, `numpy`, `nlpaug`, `librosa`, etc) if possible, and avoid adding new dependencies as these will make AugLy heavier and slower to download. However, if you feel it's necessary in order to implement your new augmentation and that the new augmentation is really worth having, it may be fine; in this case, add your new dependency to `requirements.txt`, and then make sure you can install `augly` in a fresh conda environment and the new unit test passes.

If you want to see an example of what the changes should look like in each file, search all of the above files for one of the existing augmentations, e.g. `overlay_text`.

## Unit Tests
Please run all the unit tests for the module where you added your new augmentation, and paste the output here. Likewise for `image` or `text`. If your changes could affect behavior in multiple modules, please run the tests for all potentially affected modules. If you are unsure of which modules might be affected by your changes, please just run all the unit tests.

### Audio
```bash
python -m unittest augly.tests.audio_tests.functional_unit_tests
python -m unittest augly.tests.audio_tests.transforms_unit_tests
```

### Image
```bash
python -m unittest augly.tests.image_tests.functional_unit_tests
python -m unittest augly.tests.image_tests.transforms_unit_tests
```

### Text
```bash
python -m unittest augly.tests.text_tests.functional_unit_tests
python -m unittest augly.tests.text_tests.transforms_unit_tests
```