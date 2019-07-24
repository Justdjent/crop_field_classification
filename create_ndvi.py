import rasterio
import os
import argparse
from tqdm import tqdm
import numpy as np
# norm_values = {
#     "TCI": {"min": 0, "max": 255}
#     "B02": {"min": 0, "max": 65536},
#     "B01": {"min": 0, "max": 65536},
#     "B03": {"min": 0, "max": 65536},
#     "B04": {"min": 0, "max": 65536},
#     "B05": {"min": 0, "max": 65536},
#     "B06": {"min": 0, "max": 65536},
#     "B07": {"min": 0, "max": 65536},
#     "B08": {"min": 0, "max": 65536},
#     "B8A": {"min": 0, "max": 65536},
#     "B09": {"min": 0, "max": 65536},
#     "B10": {"min": 0, "max": 65536},
#     "B11": {"min": 0, "max": 65536},
#     "B12": {"min": 0, "max": 65536},
# }

def save_raster_path(save_path, meta, raster_array):
    """
    Save raster having array metadata and save path
    :param save_path: path to save raster
    :param meta: metadata
    :param raster_array: raster_array
    :return:
    """
    if len(raster_array.shape) < 3:
        raster_array = np.expand_dims(raster_array, axis=0)
    with rasterio.open(save_path, 'w', **meta) as dst:
        for i in range(1, meta['count'] + 1):
            src_array = raster_array[i - 1]
            dst.write(src_array, i)


def create_ndvi(nir_path, red_path):
    # ndvi = (B08 - B04) / (B08 + B04)
    with rasterio.open(nir_path, "r") as nir_src:
        nir = nir_src.read(1).astype(np.float32)
        meta = nir_src.meta
    with rasterio.open(red_path, "r") as red_src:
        red = red_src.read(1).astype(np.float32)

    meta.update({"dtype":"float32",
                 "driver": "GTiff",
                 "nodata": -10000})
    ndvi = (nir - red) / (nir + red)
    ndvi_path = nir_path.replace("B08.jp2", "ndvi.tif")
    save_raster_path(ndvi_path, meta, ndvi.astype(np.float32))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run alignment descriptor')
    parser.add_argument('--folder', help='farmID', default='/home/user/projects/africa_data/Africa/data/images')
    args = parser.parse_args()
    dates = os.listdir(args.folder)

    for date in tqdm(dates):
        print(date)
        # for band in tqdm(norm_values.keys()):
        nir_path = os.path.join(args.folder, date, "full", "B08.jp2")
        red_path = os.path.join(args.folder, date, "full", "B04.jp2")
        create_ndvi(nir_path, red_path)
        # break