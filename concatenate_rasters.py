import rasterio
from rasterio.merge import merge
import os
from tqdm import tqdm

IMAGES_PATH = "data\\images\\"
CRS = rasterio.crs.CRS().from_string("EPSG:32734")

dates = os.listdir(IMAGES_PATH)

# for every date
for date in tqdm(dates):
    bands = os.listdir(os.path.join(IMAGES_PATH, date, "1"))

    # for every band
    for band in bands:

        # open and combine two rasters
        with rasterio.open(
            os.path.join(IMAGES_PATH, date, "1", band), "r", driver="JP2OpenJPEG"
        ) as original:

            with rasterio.open(
                os.path.join(IMAGES_PATH, date, "2", band.replace("E", "F")),
                "r",
                driver="JP2OpenJPEG",
            ) as extra:

                os.makedirs(os.path.join(IMAGES_PATH, date, "full"), exist_ok=True)
                full, full_transform = merge([original, extra])
                profile = {
                    "width": full.shape[2],
                    "height": full.shape[1],
                    "count": full.shape[0],
                    "crs": CRS,
                    "dtype": original.dtypes[0],
                    "nodata": original.nodata,
                    "transform": full_transform,
                }
                with rasterio.open(
                    os.path.join(IMAGES_PATH, date, "full", band[-7:]),
                    "w",
                    driver="JP2OpenJPEG",
                    **profile
                ) as dataset:
                    dataset.write(full)
