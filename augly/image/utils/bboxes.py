#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import math
import numpy as np
from typing import List, Optional, Tuple

import augly.image.utils as imutils


def crop_bboxes_helper(
    bbox: Tuple, x1: float, y1: float, x2: float, y2: float, **kwargs
) -> Tuple:
    """
    If part of the bbox was cropped out in the x-axis, the left/right side will now be
    0/1 respectively; otherwise the fraction x1 is cut off from the left & x2 from the
    right and we renormalize with the new width. Analogous for the y-axis
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    new_w, new_h = x2 - x1, y2 - y1
    return (
        max(0, (left_factor - x1) / new_w),
        max(0, (upper_factor - y1) / new_h),
        min(1, 1 - (x2 - right_factor) / new_w),
        min(1, 1 - (y2 - lower_factor) / new_h),
    )


def hflip_bboxes_helper(bbox: Tuple, **kwargs) -> Tuple:
    """
    When the src image is horizontally flipped, the bounding box also gets horizontally
    flipped
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    return (1 - right_factor, upper_factor, 1 - left_factor, lower_factor)


def meme_format_bboxes_helper(
    bbox: Tuple, src_w: int, src_h: int, caption_height: int, **kwargs
) -> Tuple:
    """
    The src image is offset vertically by caption_height pixels, so we normalize that to
    get the y offset, add that to the upper & lower coordinates, & renormalize with the
    new height. The x dimension is unaffected
    """
    left_f, upper_f, right_f, lower_f = bbox
    y_off = caption_height / src_h
    new_h = 1.0 + y_off
    return left_f, (upper_f + y_off) / new_h, right_f, (lower_f + y_off) / new_h


def overlay_onto_background_image_bboxes_helper(
    bbox: Tuple, overlay_size: float, x_pos: float, y_pos: float, **kwargs
) -> Tuple:
    """
    The src image is overlaid on the dst image offset by (`x_pos`, `y_pos`) & with a
    size of `overlay_size` (all relative to the dst image dimensions). So the bounding
    box is also offset by (`x_pos`, `y_pos`) & scaled by `overlay_size`. It is also
    possible that some of the src image will be cut off, so we take the max with 0/min
    with 1 in order to crop the bbox if needed
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    return (
        max(0, left_factor * overlay_size + x_pos),
        max(0, upper_factor * overlay_size + y_pos),
        min(1, right_factor * overlay_size + x_pos),
        min(1, lower_factor * overlay_size + y_pos),
    )


def overlay_image_bboxes_helper(
    bbox: Tuple,
    opacity: float,
    overlay_size: float,
    x_pos: float,
    y_pos: float,
    max_visible_opacity: float,
    **kwargs,
) -> Tuple:
    """
    We made a few decisions for this augmentation about how bboxes are defined:
    1. If `opacity` < `max_visible_opacity` (default 0.75, can be specified by the user),
       the bbox stays the same because it is still considered "visible" behind the
       overlaid image
    2. If the entire bbox is covered by the overlaid image, the bbox is no longer valid
       so we return it as (0, 0, 0, 0), which will be turned to None in `check_bboxes()`
    3. If the entire bottom of the bbox is covered by the overlaid image
       (i.e. `x_pos < left_factor` & `x_pos + overlay_size > right_factor` &
       `y_pos + overlay_size > lower_factor`), we crop out the lower part of the bbox
       that is covered. The analogue is true for the top/left/right being occluded
    4. If just the middle of the bbox is covered or a rectangle is sliced out of the
       bbox, we consider that the bbox is unchanged, even though part of it is occluded.
       This isn't ideal but otherwise it's very complicated; we could split the
       remaining area into smaller visible bboxes, but then we would have to return
       multiple dst bboxes corresponding to one src bbox
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    if opacity >= max_visible_opacity:
        occluded_left = x_pos < left_factor
        occluded_upper = y_pos < upper_factor
        occluded_right = x_pos + overlay_size > right_factor
        occluded_lower = y_pos + overlay_size > lower_factor

        if occluded_left and occluded_right:
            # If the bbox is completely covered, it's no longer valid so return zeros
            if occluded_upper and occluded_lower:
                return (0.0, 0.0, 0.0, 0.0)

            if occluded_lower:
                lower_factor = y_pos
            elif occluded_upper:
                upper_factor = y_pos + overlay_size
        elif occluded_upper and occluded_lower:
            if occluded_right:
                right_factor = x_pos
            elif occluded_left:
                left_factor = x_pos + overlay_size

    return left_factor, upper_factor, right_factor, lower_factor


def overlay_onto_screenshot_bboxes_helper(
    bbox: Tuple,
    src_w: int,
    src_h: int,
    template_filepath: str,
    template_bboxes_filepath: str,
    resize_src_to_match_template: bool,
    max_image_size_pixels: int,
    crop_src_to_fit: bool,
    **kwargs,
) -> Tuple:
    """
    We transform the bbox by applying all the same transformations as are applied in the
    `overlay_onto_screenshot` function, each of which is mentioned below in comments
    """
    left_f, upper_f, right_f, lower_f = bbox
    template, tbbox = imutils.get_template_and_bbox(
        template_filepath, template_bboxes_filepath
    )

    # Either src image or template image is scaled
    if resize_src_to_match_template:
        tbbox_w, tbbox_h = tbbox[2] - tbbox[0], tbbox[3] - tbbox[1]
        src_scale_factor = min(tbbox_w / src_w, tbbox_h / src_h)
    else:
        template, tbbox = imutils.scale_template_image(
            src_w, src_h, template, tbbox, max_image_size_pixels, crop_src_to_fit,
        )
        tbbox_w, tbbox_h = tbbox[2] - tbbox[0], tbbox[3] - tbbox[1]
        src_scale_factor = 1

    template_w, template_h = template.size
    x_off, y_off = tbbox[:2]

    # Src image is scaled (if resize_src_to_match_template)
    curr_w, curr_h = src_w * src_scale_factor, src_h * src_scale_factor
    left, upper, right, lower = (
        left_f * curr_w, upper_f * curr_h, right_f * curr_w, lower_f * curr_h
    )

    # Src image is cropped to (tbbox_w, tbbox_h)
    if crop_src_to_fit:
        dx, dy = (curr_w - tbbox_w) // 2, (curr_h - tbbox_h) // 2
        x1, y1, x2, y2 = dx, dy, dx + tbbox_w, dy + tbbox_h
        left_f, upper_f, right_f, lower_f = crop_bboxes_helper(
            bbox, x1 / curr_w, y1 / curr_h, x2 / curr_w, y2 / curr_h
        )
        left, upper, right, lower = (
            left_f * tbbox_w, upper_f * tbbox_h, right_f * tbbox_w, lower_f * tbbox_h
        )
    # Src image is resized to (tbbox_w, tbbox_h)
    else:
        resize_f = min(tbbox_w / curr_w, tbbox_h / curr_h)
        left, upper, right, lower = (
            left * resize_f, upper * resize_f, right * resize_f, lower * resize_f
        )
        curr_w, curr_h = curr_w * resize_f, curr_h * resize_f

        # Padding with black
        padding_x = max(0, (tbbox_w - curr_w) // 2)
        padding_y = max(0, (tbbox_h - curr_h) // 2)
        left, upper, right, lower = (
            left + padding_x, upper + padding_y, right + padding_x, lower + padding_y
        )

    # Src image is overlaid onto template image
    left, upper, right, lower = left + x_off, upper + y_off, right + x_off, lower + y_off

    return left / template_w, upper / template_h, right / template_w, lower / template_h


def pad_bboxes_helper(bbox: Tuple, w_factor: float, h_factor: float, **kwargs) -> Tuple:
    """
    The src image is padded horizontally with w_factor * src_w, so the bbox gets shifted
    over by w_factor and then renormalized over the new width. Vertical padding is
    analogous
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    new_w = 1 + 2 * w_factor
    new_h = 1 + 2 * h_factor
    return (
        (left_factor + w_factor) / new_w,
        (upper_factor + h_factor) / new_h,
        (right_factor + w_factor) / new_w,
        (lower_factor + h_factor) / new_h,
    )


def pad_square_bboxes_helper(bbox: Tuple, src_w: int, src_h: int, **kwargs) -> Tuple:
    """
    In pad_square, pad is called with w_factor & h_factor computed as follows, so we can
    use the `pad_bboxes_helper` function to transform the bbox
    """
    w_factor, h_factor = 0, 0

    if src_w < src_h:
        w_factor = (src_h - src_w) / (2 * src_w)
    else:
        h_factor = (src_w - src_h) / (2 * src_h)

    return pad_bboxes_helper(bbox, w_factor=w_factor, h_factor=h_factor)


def perspective_transform_bboxes_helper(
    bbox: Tuple,
    src_w: int,
    src_h: int,
    sigma: float,
    dx: float,
    dy: float,
    crop_out_black_border: bool,
    seed: Optional[int],
    **kwargs,
) -> Tuple:
    """
    Computes the bbox that encloses the bbox in the perspective transformed image. Also
    uses the `crop_bboxes_helper` function since the image is cropped if
    `crop_out_black_border` is True.
    """

    def transform(x: float, y: float, a: List[float]) -> Tuple:
        """
        Transforms a point in the image given the perspective transform matrix; we will
        use this to transform the bounding box corners. Based on PIL source code:
        https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Geometry.c#L399
        """
        return (
            (a[0] * x + a[1] * y + a[2]) / (a[6] * x + a[7] * y + a[8]),
            (a[3] * x + a[4] * y + a[5]) / (a[6] * x + a[7] * y + a[8]),
        )

    def get_perspective_transform(
        src_coords: List[Tuple[int, int]], dst_coords: List[Tuple[int, int]]
    ) -> List[float]:
        """
        Computes the transformation matrix used for the perspective transform with
        the given src & dst corner coordinates. Based on OpenCV source code:
        https://github.com/opencv/opencv/blob/master/modules/imgproc/src/imgwarp.cpp#L3277-L3304
        """
        a = np.zeros((8, 8), dtype=np.float)
        dst_x, dst_y = zip(*dst_coords)
        b = np.asarray(list(dst_x) + list(dst_y))

        for i, (sc, dc) in enumerate(zip(src_coords, dst_coords)):
            a[i][0] = a[i + 4][3] = sc[0]
            a[i][1] = a[i + 4][4] = sc[1]
            a[i][2] = a[i + 4][5] = 1
            a[i][6] = -sc[0] * dc[0]
            a[i][7] = -sc[1] * dc[0]
            a[i + 4][6] = -sc[0] * dc[1]
            a[i + 4][7] = -sc[1] * dc[1]

        A = np.matrix(a, dtype=np.float)
        B = np.array(b).reshape(8)
        res = np.linalg.solve(A, B)
        return np.array(res).reshape(8).tolist() + [1.0]

    assert (
        seed is not None
    ), "Cannot transform bbox for perspective_transform if seed is not provided"

    rng = np.random.RandomState(seed)
    src_coords = [(0, 0), (src_w, 0), (src_w, src_h), (0, src_h)]
    dst_coords = [
        (rng.normal(point[0], sigma) + dx, rng.normal(point[1], sigma) + dy)
        for point in src_coords
    ]

    perspective_transform_coeffs = get_perspective_transform(src_coords, dst_coords)

    left_f, upper_f, right_f, lower_f = bbox
    left, upper, right, lower = (
        left_f * src_w, upper_f * src_h, right_f * src_w, lower_f * src_h
    )
    bbox_coords = [(left, upper), (right, upper), (right, lower), (left, lower)]

    transformed_bbox_coords = [
        transform(x + 0.5, y + 0.5, perspective_transform_coeffs) for x, y in bbox_coords
    ]

    transformed_xs, transformed_ys = zip(*transformed_bbox_coords)
    transformed_bbox = (
        max(0, min(transformed_xs) / src_w),
        max(0, min(transformed_ys) / src_h),
        min(1, max(transformed_xs) / src_w),
        min(1, max(transformed_ys) / src_h),
    )

    # This is copy-pasted from `functional.py`, exactly how the crop coords are computed
    if crop_out_black_border:
        top_left, top_right, bottom_right, bottom_left = dst_coords
        new_left = max(0, top_left[0], bottom_left[0])
        new_right = min(src_w, top_right[0], bottom_right[0])
        new_top = max(0, top_left[1], top_right[1])
        new_bottom = min(src_h, bottom_left[1], bottom_right[1])

        transformed_bbox = crop_bboxes_helper(
            transformed_bbox,
            x1=new_left / src_w,
            y1=new_top / src_h,
            x2=new_right / src_w,
            y2=new_bottom / src_h,
        )

    return transformed_bbox


def rotate_bboxes_helper(
    bbox: Tuple, src_w: int, src_h: int, degrees: float, **kwargs
) -> Tuple:
    """
    Computes the bbox that encloses the rotated bbox in the rotated image. This code was
    informed by looking at the source code for PIL.Image.rotate
    (https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html#Image.rotate).
    Also uses the `crop_bboxes_helper` function since the image is cropped after being
    rotated.
    """
    left_f, upper_f, right_f, lower_f = bbox
    left, upper, right, lower = (
        left_f * src_w, upper_f * src_h, right_f * src_w, lower_f * src_h
    )
    # Top left, upper right, lower right, & lower left corner coefficients (in pixels)
    bbox_corners = [(left, upper), (right, upper), (right, lower), (left, lower)]

    def transform(x: int, y: int, matrix: List[float]) -> Tuple[float, float]:
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    def get_enclosing_bbox(
        corners: List[Tuple[int, int]], rotation_matrix: List[float]
    ) -> Tuple[int, int, int, int]:
        rotated_corners = [transform(x, y, rotation_matrix) for x, y in corners]
        xs, ys = zip(*rotated_corners)
        return (
            math.floor(min(xs)),
            math.floor(min(ys)),
            math.ceil(max(xs)),
            math.ceil(max(ys)),
        )

    # Get rotated bbox corner coefficients
    rotation_center = (src_w // 2, src_h // 2)
    angle_rad = -math.radians(degrees)
    rotation_matrix = [
        round(math.cos(angle_rad), 15),
        round(math.sin(angle_rad), 15),
        0.0,
        round(-math.sin(angle_rad), 15),
        round(math.cos(angle_rad), 15),
        0.0,
    ]
    rotation_matrix[2], rotation_matrix[5] = transform(
        -rotation_center[0], -rotation_center[1], rotation_matrix
    )
    rotation_matrix[2] += rotation_center[0]
    rotation_matrix[5] += rotation_center[1]

    # Get rotated image dimensions
    src_img_corners = [(0, 0), (src_w, 0), (src_w, src_h), (0, src_h)]
    rotated_img_min_x, rotated_img_min_y, rotated_img_max_x, rotated_img_max_y = (
        get_enclosing_bbox(src_img_corners, rotation_matrix)
    )
    rotated_img_w = rotated_img_max_x - rotated_img_min_x
    rotated_img_h = rotated_img_max_y - rotated_img_min_y

    # Get enclosing box corners around rotated bbox (on rotated image)
    new_bbox_left, new_bbox_upper, new_bbox_right, new_bbox_lower = get_enclosing_bbox(
        bbox_corners, rotation_matrix
    )
    bbox_enclosing_bbox = (
        new_bbox_left / rotated_img_w,
        new_bbox_upper / rotated_img_h,
        new_bbox_right / rotated_img_w,
        new_bbox_lower / rotated_img_h,
    )

    # Crop bbox as src image is cropped inside `rotate`
    cropped_w, cropped_h = imutils.rotated_rect_with_max_area(src_w, src_h, degrees)
    cropped_img_left, cropped_img_upper, cropped_img_right, cropped_img_lower = (
        (rotated_img_w - cropped_w) // 2 + rotated_img_min_x,
        (rotated_img_h - cropped_h) // 2 + rotated_img_min_y,
        (rotated_img_w + cropped_w) // 2 + rotated_img_min_x,
        (rotated_img_h + cropped_h) // 2 + rotated_img_min_y,
    )
    return crop_bboxes_helper(
        bbox_enclosing_bbox,
        x1=cropped_img_left / rotated_img_w,
        y1=cropped_img_upper / rotated_img_h,
        x2=cropped_img_right / rotated_img_w,
        y2=cropped_img_lower / rotated_img_h,
    )


def vflip_bboxes_helper(bbox: Tuple, **kwargs) -> Tuple:
    """
    Analogous to hflip, when the src image is vertically flipped, the bounding box also
    gets vertically flipped
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    return (left_factor, 1 - lower_factor, right_factor, 1 - upper_factor)
