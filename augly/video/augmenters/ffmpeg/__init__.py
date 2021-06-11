#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from augly.video.augmenters.ffmpeg.base_augmenter import BaseFFMPEGAugmenter
from augly.video.augmenters.ffmpeg.aspect_ratio import VideoAugmenterByAspectRatio
from augly.video.augmenters.ffmpeg.audio_swap import VideoAugmenterByAudioSwap
from augly.video.augmenters.ffmpeg.blur import VideoAugmenterByBlur
from augly.video.augmenters.ffmpeg.brightness import VideoAugmenterByBrightness
from augly.video.augmenters.ffmpeg.color_jitter import VideoAugmenterByColorJitter
from augly.video.augmenters.ffmpeg.concat import VideoAugmenterByConcat
from augly.video.augmenters.ffmpeg.contrast import VideoAugmenterByContrast
from augly.video.augmenters.ffmpeg.crop import VideoAugmenterByCrop
from augly.video.augmenters.ffmpeg.fps import VideoAugmenterByFPSChange
from augly.video.augmenters.ffmpeg.grayscale import VideoAugmenterByGrayscale
from augly.video.augmenters.ffmpeg.hflip import VideoAugmenterByHFlip
from augly.video.augmenters.ffmpeg.loops import VideoAugmenterByLoops
from augly.video.augmenters.ffmpeg.noise import VideoAugmenterByNoise
from augly.video.augmenters.ffmpeg.overlay import VideoAugmenterByOverlay
from augly.video.augmenters.ffmpeg.pad import VideoAugmenterByPadding
from augly.video.augmenters.ffmpeg.no_audio import VideoAugmenterByRemovingAudio
from augly.video.augmenters.ffmpeg.resize import VideoAugmenterByResize
from augly.video.augmenters.ffmpeg.resolution import VideoAugmenterByResolution
from augly.video.augmenters.ffmpeg.rotate import VideoAugmenterByRotation
from augly.video.augmenters.ffmpeg.quality import VideoAugmenterByQuality
from augly.video.augmenters.ffmpeg.speed import VideoAugmenterBySpeed
from augly.video.augmenters.ffmpeg.stack import VideoAugmenterByStack
from augly.video.augmenters.ffmpeg.trim import VideoAugmenterByTrim
from augly.video.augmenters.ffmpeg.vflip import VideoAugmenterByVFlip

__all__ = [
    "BaseFFMPEGAugmenter",
    "VideoAugmenterByAspectRatio",
    "VideoAugmenterByAudioSwap",
    "VideoAugmenterByBlur",
    "VideoAugmenterByBrightness",
    "VideoAugmenterByColorJitter",
    "VideoAugmenterByConcat",
    "VideoAugmenterByContrast",
    "VideoAugmenterByCrop",
    "VideoAugmenterByFPSChange",
    "VideoAugmenterByGrayscale",
    "VideoAugmenterByHFlip",
    "VideoAugmenterByLoops",
    "VideoAugmenterByNoise",
    "VideoAugmenterByOverlay",
    "VideoAugmenterByPadding",
    "VideoAugmenterByQuality",
    "VideoAugmenterByRemovingAudio",
    "VideoAugmenterByResize",
    "VideoAugmenterByResolution",
    "VideoAugmenterByRotation",
    "VideoAugmenterBySpeed",
    "VideoAugmenterByStack",
    "VideoAugmenterByTrim",
    "VideoAugmenterByVFlip",
]
