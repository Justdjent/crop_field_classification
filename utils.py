import cv2
import numpy as np


def multindex_iloc(df, index, level=0):
    label = df.index.levels[level][index]
    return df.iloc[df.index.get_loc(label)]


def generate_pixel_list(inp_image):

    pixel_dict = inp_image[np.all(inp_image != 0, axis=-1)]
    return pixel_dict


def pad_with_random_pixel(img, pad_size):

    # get pad dims
    h, w, c = img.shape
    top = (pad_size - h) // 2
    bottom = pad_size - top - h
    left = (pad_size - w) // 2
    right = pad_size - left - w

    # pad image with zeros
    out_img = cv2.copyMakeBorder(img,bottom, top, left, right, cv2.BORDER_CONSTANT, 0)

    # get pad list
    px_dict = generate_pixel_list(img)
    np.random.shuffle(px_dict)

    # replace zeros with list values
    np.place(out_img, out_img == 0, px_dict)
    return out_img



