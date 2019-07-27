import cv2
import numpy as np


from typing import List

from catalyst.dl.core import MultiMetricCallback
from catalyst.dl.utils import criterion

def multindex_iloc(df, index, level=0):
    label = df.index.levels[level][index]
    return df.iloc[df.index.get_loc(label)]


def generate_pixel_list(inp_image):

    pixel_dict = inp_image[np.all(inp_image != 0, axis=-1)]
    # print(pixel_dict)
    if len(pixel_dict) == 0:
        pixel_dict = [[0, 0, 0]]
    return pixel_dict


def pad_with_random_pixel(img, pad_size):

    if pad_size != 0:
        # get pad dims
        h, w, c = img.shape
        top = (pad_size - h) // 2
        bottom = pad_size - top - h
        left = (pad_size - w) // 2
        right = pad_size - left - w

        # pad image with zeros
        out_img = cv2.copyMakeBorder(img, bottom, top, left, right, cv2.BORDER_CONSTANT, 0)
    else:
        out_img = img

    # get pad list
    px_dict = generate_pixel_list(img)
    np.random.shuffle(px_dict)

    # replace zeros with list values
    np.place(out_img, out_img == 0, px_dict)
    return out_img


def pad_with_wrap(img, pad_size):

    # get pad dims
    h, w, c = img.shape
    top = (pad_size - h) // 2
    bottom = pad_size - top - h
    left = (pad_size - w) // 2
    right = pad_size - left - w
    print(img.shape)
    # get pad list
    px_dict = generate_pixel_list(img)
    np.random.shuffle(px_dict)
    # replace zeros with list values
    np.place(img, img == 0, px_dict)

    # bad an image
    out_img = cv2.copyMakeBorder(img, bottom, top, left, right, cv2.BORDER_WRAP)
    return out_img


class AerageMeanAccuracyCallback(MultiMetricCallback):
    """
    Accuracy metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        accuracy_args: List[int] = None,
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`.
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`.
            accuracy_args: specifies which accuracy@K to log.
                [1] - accuracy
                [1, 3] - accuracy at 1 and 3
                [1, 3, 5] - accuracy at 1, 3 and 5
        """
        super().__init__(
            prefix=prefix,
            metric_fn=criterion.mean_average_accuracy,
            list_args=accuracy_args or [1],
            input_key=input_key,
            output_key=output_key
        )