# AugLy

AugLy is a data augmentations library that currently supports four modalities ([audio](augly/audio), [image](augly/image), [text](augly/text) & [video](augly/video)) and over 100 augmentations. Each modality’s augmentations are contained within its own sub-library. These sub-libraries include both function-based and class-based transforms, composition operators, and have the option to provide metadata about the transform applied, including its intensity.

AugLy is a great library to utilize for augmenting your data for model training, or to evaluate the robustness gaps of your model!

The library is Python-based and requires at least Python 3.7, as we use dataclasses.

## Authors
[**Joanna Bitton**](https://www.linkedin.com/in/joanna-bitton/) — Software Engineer at Facebook AI

[**Zoe Papakipos**](https://www.linkedin.com/in/zoe-papakipos-8637155b/) — Research Engineer at FAIR

## Installation

```bash
pip install augly
```

Or clone augly if you want to be able to run our unit tests, contribute a pull request, etc:
```bash
git clone git@github.com:facebookresearch/AugLy.git
[Optional] conda create -n augly && conda activate augly && conda install pip
pip install -r requirements.txt
```

After installing the pip requirements, there are a few additional dependencies you'll need to install using conda:
```bash
conda install -c conda-forge python-magic
conda install -c conda-forge opencv=4.5.2
export PYTHONPATH="${PYTHONPATH}:/<absolute_path_to_AugLy>/augly"
```

## Documentation

To find documentation about each sub-library, please review the READMEs in their respective directories.

## Assets

We provide various media assets to use with some of our augmentations. These assets include:
1. [Emojis](assets/twemojis/) ([Twemoji](https://twemoji.twitter.com/)) - Copyright 2020 Twitter, Inc and other contributors. Code licensed under the MIT License. Graphics licensed under CC-BY 4.0.
2. [Fonts](assets/fonts/) ([Noto fonts](https://www.google.com/get/noto/)) - Noto is a trademark of Google Inc. Noto fonts are open source. All Noto fonts are published under the SIL Open Font License, Version 1.1.
3. [Screenshot Templates](assets/screenshot_templates/) - Images created by a designer at Facebook specifically to use with AugLy. You can use these with the `overlay_onto_screenshot` augmentation in both the image and video libraries to make it look like your source image/video was screenshotted in a social media feed similar to Facebook or Instagram.

## Citation

If you use AugLy in your work, please cite:

```bibtex
@misc{bitton2020augly,
  author =       {Bitton, Joanna and Papakipos, Zoe},
  title =        {AugLy: A data augmentations library for audio, image, text, and video.},
  howpublished = {\url{https://github.com/facebookresearch/AugLy}},
  year =         {2021}
}
```

## License

AugLy is MIT licensed, as found in the [LICENSE](LICENSE) file. Please note that some of the dependencies AugLy uses may be licensed under different terms.
