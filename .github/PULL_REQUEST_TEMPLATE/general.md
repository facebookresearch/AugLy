## Related Issue (if applicable)
Fixes #{issue number}

## Summary
What are you trying to achieve in this PR? Please summarize what changes you made and how they acheive the desired result.

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
python -m unittest augly.tests.image_tests.pytorch_test  # Note: must have torch installed
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