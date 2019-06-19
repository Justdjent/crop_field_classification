from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
import os
from utils import multindex_iloc


class AfricanFieldsDataset(Dataset):
    """African Fields dataset."""

    def __init__(
        self,
        csv_file,
        root_dir,
        bands=("B02", "B03", "B04"),
        dates=("2017-01-01"),
        resize_inter=cv2.INTER_NEAREST,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Image paths are relative to this dir
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.df.set_index(["Field_Id", "Band"], inplace=True)
        self.df.sort_index(inplace=True)
        self.dates = dates
        self.bands = bands
        self.resize_inter = resize_inter

        self.root_dir = root_dir

    #         self.transform = transform

    def __len__(self):
        return len(list(self.df.index.levels[0].unique()))

    def __getitem__(self, idx):
        field_info = multindex_iloc(self.df, idx)
        label = field_info["Crop_Id_Ne"][0]
        #         field_info = field_info[self.dates]

        field_id = field_info.index.get_level_values("Field_Id")[0]
        field_info = field_info.loc[
            field_info.index.get_level_values("Band").isin(self.bands)
        ]
        image_sequence = []
        for date in self.dates:
            date_info = field_info[date]
            image_names = list(date_info)
            images = []
            max_shape = [0, 0]
            for name in image_names:
                image = np.load(os.path.join(self.root_dir, name)).squeeze()
                if image.shape[0] > max_shape[0]:
                    max_shape = image.shape

                images.append(image)

            stacked_image = np.zeros(max_shape + (len(image_names),))
            for i, image in enumerate(images):
                stacked_image[:, :, i] = cv2.resize(
                    image, max_shape[::-1], interpolation=self.resize_inter
                )
            image_sequence.append(stacked_image)

        image_sequence = np.array(image_sequence)
        sample = {"image_sequence": image_sequence, "label": label, "id": field_id}

        #         if self.transform:
        #             sample = self.transform(sample)

        return sample
