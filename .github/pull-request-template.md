## Related Issue
Fixes #{issue number}

## Summary
- [ ] I have read CONTRIBUTING.md to understand how to contribute to this repository :)

<Please summarize what you are trying to achieve, what changes you made, and how they acheive the desired result.>

## Unit Tests
If your changes touch the `audio` module, please run all of the `audio` tests and paste the output here. Likewise for `image`, `text`, & `video`. If your changes could affect behavior in multiple modules, please run the tests for all potentially affected modules. If you are unsure of which modules might be affected by your changes, please just run all the unit tests.

### Audio
```bash
python -m unittest discover -s augly/tests/audio_tests/ -p "*"
```

### Image
```bash
python -m unittest discover -s augly/tests/image_tests/ -p "*_test.py"
# Or `python -m unittest discover -s augly/tests/image_tests/ -p "*.py"` to run pytorch test too (must install `torchvision` to run)
```

### Text
```bash
python -m unittest discover -s augly/tests/text_tests/ -p "*"
```

### Video
```bash
python -m unittest discover -s augly/tests/video_tests/ -p "*"
```

### All
```bash
python -m unittest discover -s augly/tests/ -p "*"
```

## Other testing

If applicable, test your changes and paste the output here. For example, if your changes affect the requirements/installation, then test installing augly in a fresh conda env, then make sure you are able to import augly & run the unit test
