/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
module.exports = {
  title: 'AugLy',
  tagline: 'A data augmentations library for audio, image, text, and video.',
  url: 'https://augly.io',
  baseUrl: '/',
  organizationName: 'facebookresearch',
  projectName: 'AugLy',
  favicon: 'img/favicon.ico',
  presets: [['@docusaurus/preset-classic', {
    theme: {
      customCss: require.resolve('./src/css/custom.css')
    }
  }]],
  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      disableSwitch: true
    },
    navbar: {
      title: 'AugLy',
      logo: {
        alt: 'AugLy Logo',
        src: 'img/logo.png',
      },
      items: [{
        to: 'http://augly.rtfd.io',
        label: 'Docs',
        position: 'right'
      }, {
        href: 'https://github.com/facebook/draft-js',
        label: 'GitHub',
        position: 'right'
      }]
    },
    footer: {
      style: 'dark',
      links: [{
        title: 'Docs',
        items: [{
          label: 'Audio',
          to: 'https://augly.rtfd.io/en/latest/augly.audio.html'
        }, {
          label: 'Image',
          to: 'https://augly.rtfd.io/en/latest/augly.image.html'
        }, {
          label: 'Video',
          to: 'https://augly.rtfd.io/en/latest/augly.video.html'
        }, {
          label: 'Text',
          to: 'https://augly.rtfd.io/en/latest/augly.text.html'
        }]
      }, {
        title: 'Examples',
        items: [{
          label: 'Audio',
          to: 'https://colab.research.google.com/github/facebookresearch/AugLy/blob/main/examples/AugLy_audio.ipynb'
        }, {
          label: 'Image',
          to: 'https://colab.research.google.com/github/facebookresearch/AugLy/blob/main/examples/AugLy_image.ipynb'
        }, {
          label: 'Video',
          to: 'https://colab.research.google.com/github/facebookresearch/AugLy/blob/main/examples/AugLy_video.ipynb'
        }, {
          label: 'Text',
          to: 'https://colab.research.google.com/github/facebookresearch/AugLy/blob/main/examples/AugLy_text.ipynb'
        }]
      }, {
        title: 'More',
        items: [{
          label: 'GitHub',
          href: 'https://github.com/facebookresearch/AugLy'
        }]
      }],
      logo: {
        alt: 'Facebook Open Source Logo',
        src: '/img/oss_logo.png',
        href: 'https://opensource.facebook.com/'
      },
      copyright: `Copyright © ${new Date().getFullYear()} Facebook, Inc.`
    }
  }
};
