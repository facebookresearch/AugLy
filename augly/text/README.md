# Text

## Installation   
If you plan to use only the text augmentations from AugLy, you can install using the following command:   
```bash 
pip install augly[text]
```

Otherwise if you want to use other modealities as well, you can install AugLy with the required dependencies for all modalities, which just means the installation will be a bit heavier:
```bash
pip install augly[all]
```

## Augmentations

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/AugLy/blob/main/examples/AugLy_text.ipynb)

Try running some AugLy text augmentations in Colab! For a full list of available augmentations, see [here](__init__.py).

Our text augmentations use `nlpaug` as their backbone. All functions accept a list of original texts to be augmented as input and return the list of augmented texts.

### Function-based

You can call the functional augmentations like so:
```python
import augly.text as txtaugs

texts = ["hello world", "bye planet"]

augmented_synonyms = txtaugs.insert_punctuation_chars(
    texts,
    granularity="all",
    cadence=5.0,
    vary_chars=True,
)
```

### Class-based

You can also call any augmentation as a Transform class with a given probability:
```python
import augly.text as txtaugs

texts = ["hello world", "bye planet"]
transform = InsertPunctuationChars(granularity="all", p=0.5)
aug_texts = transform(texts)
```

## Unit Tests

You can run our text unit tests if you have cloned `augly` (see [here](../../README.md)) by running the following:
```
python -m unittest discover -s augly/tests/text_tests/ -p "*"
```
