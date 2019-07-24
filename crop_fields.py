import rasterio
import pandas
import geopandas
from rasterio.mask import mask
import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def normalize(x, x_min, x_max, a=0, b=255):
    x[x < x_min] = x_min
    x[x > x_max] = x_max
    x_norm = (b - a) * ((x - x_min) / (x_max - x_min)) + a
    x_norm[x_norm < a] = a
    x_norm[x_norm > b] = b
    return x_norm


# norm_values = {
#     "B01": {"min": 1019.7272727272727, "max": 1417.9090909090908},
#     "B02": {"min": 671.9090909090909, "max": 1332.6363636363635},
#     "B03": {"min": 561.3181818181818, "max": 1371.1363636363637},
#     "B04": {"min": 92.81818181818187, "max": 1820.818181818182},
#     "B05": {"min": 569.5454545454546, "max": 1991.909090909091},
#     "B06": {"min": 1009.8181818181818, "max": 3514.181818181818},
#     "B07": {"min": 1051.6363636363637, "max": 4326.909090909091},
#     "B08": {"min": 993.454545454545, "max": 4308.0},
#     "B8A": {"min": 1227.8181818181818, "max": 4767.090909090909},
#     "B09": {"min": 458.6363636363637, "max": 1414.5454545454545},
#     "B10": {"min": -22.545454545454547, "max": 44.36363636363636},
#     "B11": {"min": 949.4545454545457, "max": 3425.090909090909},
#     "B12": {"min": 159.909090909091, "max": 2475.5454545454545},
# }

def run(dataroot):
    norm_values = {
        "TCI": {"min": 0, "max": 255}
        # "B02": {"min": 0, "max": 65536},
        # "B01": {"min": 0, "max": 65536},
        # "B03": {"min": 0, "max": 65536},
        # "B04": {"min": 0, "max": 65536},
        # "B05": {"min": 0, "max": 65536},
        # "B06": {"min": 0, "max": 65536},
        # "B07": {"min": 0, "max": 65536},
        # "B08": {"min": 0, "max": 65536},
        # "B8A": {"min": 0, "max": 65536},
        # "B09": {"min": 0, "max": 65536},
        # "B10": {"min": 0, "max": 65536},
        # "B11": {"min": 0, "max": 65536},
        # "B12": {"min": 0, "max": 65536},
    }

    set_ = "train"
    IMAGES_PATH = f"{dataroot}/images/"
    polygons_path = f"{dataroot}/{set_}/{set_}.shp"
    df_path = f"{dataroot}/{set_}_rgb.csv"
    df = pandas.DataFrame(
        columns=[
            "Field_Id",
            "Band",
            "Subregion",
            "Crop_Id_Ne",
            "Area",
            "2017-01-01",
            "2017-01-31",
            "2017-02-10",
            "2017-03-12",
            "2017-03-22",
            "2017-05-31",
            "2017-06-20",
            "2017-07-10",
            "2017-07-15",
            "2017-08-04",
            "2017-08-19",
        ]
    )
    # df.set_index('Field_Id', inplace=True)

    labels = geopandas.read_file(polygons_path)
    labels = labels.dropna()
    CRS = rasterio.crs.CRS.from_string("EPSG:32734")
    labels = labels.to_crs(CRS)

    dates = os.listdir(IMAGES_PATH)


    for date in tqdm(dates):
        for band in tqdm(norm_values.keys()):
            with rasterio.open(
                os.path.join(IMAGES_PATH, date, "full", band + ".jp2")
            ) as dataset:
                for label in labels.iterrows():
                    # Dataframe
                    label = label[1]
                    query = f"Field_Id == {label['Field_Id']} and Band == '{band}'"
                    if len(df.query(query)) == 0:
                        df.loc[len(df)] = [
                            label["Field_Id"],
                            band,
                            label["Subregion"],
                            label["Crop_Id_Ne"],
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        ]

                    ind = df.query(query).index

                    # Image
                    write_path = os.path.join(f'{dataroot}/images_cropped_rgb', set_)
                    masks_write_path = os.path.join(f'{dataroot}/masks_cropped_rgb', set_)
                    os.makedirs(write_path, exist_ok=True)
                    os.makedirs(masks_write_path, exist_ok=True)
                    image_path = os.path.join(write_path, f"{label['Field_Id']}_{band}_{date}.png")
                    mask_path = os.path.join(masks_write_path, f"{label['Field_Id']}_{band}_{date}.png")
                    df.at[ind, date] = image_path
                    polys_only, trans_masked = mask(
                        dataset, [label["geometry"]], nodata=0, crop=True, filled=False
                    )
                    # mask_ = polys_only == 0
                    df.loc[ind, 'Area'] = label["geometry"].area
                    # masked = np.ma.array(polys_only, mask=mask_)
                    # min_, max_ = norm_values[band]["min"], norm_values[band]["max"]
                    # norm_masked = normalize(masked, min_, max_, 0, 1)
                    # polys_only = normalize(polys_only, min_, max_, 0, 1)
                    polys_only = np.transpose(polys_only, axes=(1, 2, 0))
                    cv2.imwrite(image_path, cv2.cvtColor(polys_only.data, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(mask_path, np.logical_and.reduce(~polys_only.mask, axis=2).astype(np.uint8) * 255)
                    # np.save(
                    #     image_path,
                    #     polys_only,
                    # )
    df.to_csv(df_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,
                        default='data',
                        help='path to dataset')
    args = parser.parse_args()
    run(args.dataroot)