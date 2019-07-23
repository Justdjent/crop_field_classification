from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
import os
from utils import multindex_iloc, pad_with_random_pixel
import torch


class AfricaPaddedDataset(Dataset):
    """African Fields dataset."""

    def __init__(
        self,
        csv_file,
        root_dir,
        folds=(0, 1),
        dates=("2017-01-01"),
        mask=False,
        transform=None,
        train=True
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Image paths are relative to this dir
            transform (callable, optional): Optional transform to be applied on a sample.
                                            WORKS ONLY WITH 3 CHANNEL IMAGES.
        """

        self.train = train
        self.df = pd.read_csv(csv_file)
        if self.train:
            self.df = self.df[self.df['Fold'].isin(folds)]
        self.dates = dates
        self.root_dir = root_dir
        self.transform = transform
        self.mask = mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        field_info = self.df.iloc[idx]
        if self.train:
            label = field_info["Crop_Id_Ne"]
        #         field_info = field_info[self.dates]

        image_sequence = []
        for date in self.dates:
            image_path = field_info[date]
            image = cv2.imread(os.path.join(self.root_dir, image_path))#np.load(os.path.join(self.root_dir, image_path))#.squeeze()
            # image = np.transpose(image, axes=(1, 2, 0))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(self.root_dir, image_path.replace('image', 'mask')))
            image = image*mask.astype(bool)
            image = pad_with_random_pixel(image, 0)
            # if self.mask:
            #     mask_ = cv2.imread()
            image_sequence.append(image)

        targets = {f'image{n}': image_sequence[n] for n, _ in enumerate(self.dates[1:])}
        targets['image'] = image_sequence[0]

        if self.transform:
            augmented = self.transform(**targets)
            # image = augmented['image']
            image_sequence = [augmented['image']]
            for n, _ in enumerate(self.dates[1:]):
                image_sequence.append(augmented[f'image{n}'])

        image_sequence = np.array(image_sequence, dtype=np.float32)
        image_sequence = np.transpose(image_sequence, axes=(0, 3, 1, 2))
        # image_sequence = image_sequence / 255

        if self.train:
            sample = (torch.from_numpy(image_sequence), label-1)
        else:
            sample = (torch.from_numpy(image_sequence), -1)

        return sample


class AfricanMultibandDataset(Dataset):
    """African Fields dataset."""

    def __init__(
        self,
        rgb_file,
        nir_file,
        root_dir,
        folds=(0, 1),
        dates=("2017-01-01"),
        mask=False,
        transform=None,
        train=True
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Image paths are relative to this dir
            transform (callable, optional): Optional transform to be applied on a sample.
                                            WORKS ONLY WITH 3 CHANNEL IMAGES.
        """

        self.train = train
        self.df_rgb = pd.read_csv(rgb_file)
        self.df_nir = pd.read_csv(nir_file)


        if self.train:
            self.df_rgb = self.df_rgb[self.df_rgb['Fold'].isin(folds)]
            self.df_nir = self.df_rgb[self.df_rgb['Fold'].isin(folds)]

        self.dates = dates
        self.root_dir = root_dir
        self.transform = transform
        self.mask = mask

    def __len__(self):
        return len(self.df_rgb)

    def __getitem__(self, idx):
        field_info_rgb = self.df_rgb.iloc[idx]
        field_info_nir = self.df_nir.iloc[idx]

        if self.train:
            label = field_info_rgb["Crop_Id_Ne"]

        image_sequence = []
        for date in self.dates:
            image_path = field_info_rgb[date]
            nir_path = field_info_nir[date]
            nir_image = cv2.imread(os.path.join(self.root_dir, image_path), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(os.path.join(self.root_dir, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image[:, :, 2] = nir_image
            image_sequence.append(image)

        targets = {f'image{n}': image_sequence[n] for n, _ in enumerate(self.dates[1:])}
        targets['image'] = image_sequence[0]

        if self.transform:
            augmented = self.transform(**targets)
            image_sequence = [augmented['image']]
            for n, _ in enumerate(self.dates[1:]):
                image_sequence.append(augmented[f'image{n}'])

        image_sequence = np.array(image_sequence, dtype=np.float32)
        # image_sequence = np.expand_dims(image_sequence, axis=3)
        image_sequence = np.transpose(image_sequence, axes=(0, 3, 1, 2))

        if self.train:
            sample = (torch.from_numpy(image_sequence), label-1)
        else:
            sample = (torch.from_numpy(image_sequence), -1)

        return sample


class AfricanNIRDataset(Dataset):
    """African Fields dataset."""

    def __init__(
        self,
        csv_file,
        root_dir,
        folds=(0, 1),
        dates=("2017-01-01"),
        mask=False,
        transform=None,
        train=True
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Image paths are relative to this dir
            transform (callable, optional): Optional transform to be applied on a sample.
                                            WORKS ONLY WITH 3 CHANNEL IMAGES.
        """

        self.train = train
        self.df = pd.read_csv(csv_file)
        if self.train:
            self.df = self.df[self.df['Fold'].isin(folds)]
        self.dates = dates
        self.root_dir = root_dir
        self.transform = transform
        self.mask = mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        field_info = self.df.iloc[idx]
        if self.train:
            label = field_info["Crop_Id_Ne"]

        image_sequence = []
        for date in self.dates:
            image_path = field_info[date]
            image = cv2.imread(os.path.join(self.root_dir, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_sequence.append(image)

        targets = {f'image{n}': image_sequence[n] for n, _ in enumerate(self.dates[1:])}
        targets['image'] = image_sequence[0]

        if self.transform:
            augmented = self.transform(**targets)
            image_sequence = [augmented['image'][:, :, 0]]
            for n, _ in enumerate(self.dates[1:]):
                image_sequence.append(augmented[f'image{n}'][:, :, 0])

        image_sequence = np.array(image_sequence, dtype=np.float32)
        image_sequence = np.expand_dims(image_sequence, axis=3)
        image_sequence = np.transpose(image_sequence, axes=(0, 3, 1, 2))

        if self.train:
            sample = (torch.from_numpy(image_sequence), label-1)
        else:
            sample = (torch.from_numpy(image_sequence), -1)

        return sample


class AfricanImageDataset(Dataset):
    """African Fields dataset."""

    def __init__(
        self,
        csv_file,
        root_dir,
        folds=(0, 1),
        dates=("2017-01-01"),
        mask=False,
        transform=None,
        train=True
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Image paths are relative to this dir
            transform (callable, optional): Optional transform to be applied on a sample.
                                            WORKS ONLY WITH 3 CHANNEL IMAGES.
        """

        self.train = train
        self.df = pd.read_csv(csv_file)
        if self.train:
            self.df = self.df[self.df['Fold'].isin(folds)]
        self.dates = dates
        self.root_dir = root_dir
        self.transform = transform
        self.mask = mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        field_info = self.df.iloc[idx]
        if self.train:
            label = field_info["Crop_Id_Ne"]
        #         field_info = field_info[self.dates]

        image_sequence = []
        for date in self.dates:
            image_path = field_info[date]
            image = cv2.imread(os.path.join(self.root_dir, image_path))#np.load(os.path.join(self.root_dir, image_path))#.squeeze()
            # image = np.transpose(image, axes=(1, 2, 0))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # if self.mask:
            #     mask_ = cv2.imread()
            image_sequence.append(image)

        targets = {f'image{n}': image_sequence[n] for n, _ in enumerate(self.dates[1:])}
        targets['image'] = image_sequence[0]

        if self.transform:
            augmented = self.transform(**targets)
            # image = augmented['image']
            image_sequence = [augmented['image']]
            for n, _ in enumerate(self.dates[1:]):
                image_sequence.append(augmented[f'image{n}'])

        image_sequence = np.array(image_sequence, dtype=np.float32)
        image_sequence = np.transpose(image_sequence, axes=(0, 3, 1, 2))
        # image_sequence = image_sequence / 255

        if self.train:
            sample = (torch.from_numpy(image_sequence), label-1)
        else:
            sample = (torch.from_numpy(image_sequence), -1)

        return sample


class AfricanRGBDataset(Dataset):
    """African Fields dataset."""

    def __init__(
        self,
        csv_file,
        root_dir,
        folds=(0, 1),
        dates=("2017-01-01"),
        transform=None,
        train=True
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Image paths are relative to this dir
            transform (callable, optional): Optional transform to be applied on a sample.
                                            WORKS ONLY WITH 3 CHANNEL IMAGES.
        """

        self.train = train
        self.df = pd.read_csv(csv_file)
        if self.train:
            self.df = self.df[self.df['Fold'].isin(folds)]
        self.dates = dates
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        field_info = self.df.iloc[idx]
        if self.train:
            label = field_info["Crop_Id_Ne"]
        #         field_info = field_info[self.dates]

        image_sequence = []
        for date in self.dates:
            image_path = field_info[date]
            image = np.load(os.path.join(self.root_dir, image_path))
            image = np.transpose(image, axes=(1, 2, 0))

            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']

            image = np.transpose(image, axes=(2, 0, 1))
            image_sequence.append(image)

        image_sequence = np.array(image_sequence, dtype=np.float32)
        if self.train:
            sample = (torch.from_numpy(image_sequence), label-1)
        else:
            sample = (torch.from_numpy(image_sequence), -1)

        return sample


class AfricanFieldsDataset(Dataset):
    """African Fields dataset."""

    def __init__(
        self,
        csv_file,
        root_dir,
        bands=("B02", "B03", "B04"),
        dates=("2017-01-01"),
        resize_inter=cv2.INTER_NEAREST,
        transform=None
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Image paths are relative to this dir
            transform (callable, optional): Optional transform to be applied on a sample.
                                            WORKS ONLY WITH 3 CHANNEL IMAGES.
        """
        self.df = pd.read_csv(csv_file)
        self.df.set_index(["Field_Id", "Band"], inplace=True)
        self.df.sort_index(inplace=True)
        self.dates = dates
        self.bands = bands
        self.resize_inter = resize_inter

        self.root_dir = root_dir

        self.transform = transform

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

            if self.transform:
                augmented = self.transform(image=stacked_image)
                stacked_image = augmented['image']

            image_sequence.append(stacked_image)

        image_sequence = np.array(image_sequence)
        sample = (torch.from_numpy(image_sequence), label)

        return sample
