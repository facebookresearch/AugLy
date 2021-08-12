# Image

## Installation
If you would like to use the image augmentations, please install AugLy using the following command:
```bash
pip install augly
```

## Augmentations

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/AugLy/blob/main/examples/AugLy_image.ipynb)

Try running some AugLy image augmentations! For a full list of available augmentations, see [here](__init__.py).

Our image augmentations use `PIL` as their backend. All functions accept a path to the image or a PIL Image object to be augmented as input and return the augmented PIL Image object. If an output path is specified, the image will also be saved to a file.

### Function-based

You can call the functional augmentations like so:
```python
import augly.image as imaugs

image_path = "your_img_path.png"
output_path = "your_output_path.png"

# Augmentation functions can accept image paths as input and
# always return the resulting augmented PIL Image
aug_image = imaugs.overlay_emoji(image_path, opacity=1.0, emoji_size=0.15)

# Augmentation functions can also accept PIL Images as input
aug_image = imaugs.pad_square(aug_image)

# If an output path is specified, the image will also be saved to a file
aug_image = imaugs.overlay_onto_screenshot(aug_image, output_path=output_path)
```

### Class-based

You can also call any augmentation as a Transform class, including composing them together and applying them with a given probability. This also means that you can easily integrate with PyTorch transformations if needed:
```python
import torchvision.transforms as transforms
import augly.image as imaugs

COLOR_JITTER_PARAMS = {
    "brightness_factor": 1.2,
    "contrast_factor": 1.2,
    "saturation_factor": 1.4,
}

AUGMENTATIONS = [
    imaugs.Blur(),
    imaugs.ColorJitter(**COLOR_JITTER_PARAMS),
    imaugs.OneOf(
        [imaugs.OverlayOntoScreenshot(), imaugs.OverlayEmoji(), imaugs.OverlayText()]
    ),
]

TRANSFORMS = imaugs.Compose(AUGMENTATIONS)
TENSOR_TRANSFORMS = transforms.Compose(AUGMENTATIONS + [transforms.ToTensor()])

# aug_image is a PIL image with your augs applied!
# aug_tensor_image is a Tensor with your augs applied!
image = Image.open("your_img_path.png")
aug_image = TRANSFORMS(image)
aug_tensor_image = TENSOR_TRANSFORMS(image)
```

### Numpy Wrapper
If your image is currently in the form of a NumPy array and you don't want to save the image as a file before using the augmentation functions, you can use our NumPy wrapper:
```python
from augly.image import aug_np_wrapper, overlay_emoji

np_image = np.zeros((300, 300))
# pass in function arguments as kwargs
np_aug_img = aug_np_wrapper(np_image, overlay_emoji, **{'opacity': 0.5, 'y_pos': 0.45})
```

## Unit Tests

You can run our normal image unit tests if you have cloned `augly` (see [here](../../README.md)) by running the following:
```bash
python -m unittest discover -s augly/tests/image_tests/ -p "*_test.py"
```

Note: If you want to additionally run the pytorch unit test (you must have torchvision installed), you can run:
```bash
python -m unittest discover -s augly/tests/image_tests/ -p "*"
```
