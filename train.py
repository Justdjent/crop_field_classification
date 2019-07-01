import collections
import torch
import torch.nn as nn
from albumentations import (
    Compose,
    Resize,
    Normalize,
    VerticalFlip,
    Transpose,
    ShiftScaleRotate,
    RandomRotate90,
    GridDistortion,
    ElasticTransform,
)
from africa_dataset import AfricanImageDataset, AfricanRGBDataset, AfricanNIRDataset, AfricanMultibandDataset
from simple_net import (
    SimpleNetRGB,
    SimpleNetAttentionRGB,
    SimpleNetDeeperRGB,
    SimpleNet3D,
    ModerateNetRGB
)
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, LRFinder
from catalyst.contrib.criterion import FocalLossMultiClass
from ce_callback import CECallback
from adamw import AdamW

if __name__ == "__main__":
    bs = 32
    num_workers = 3
    num_epochs = 35
    dates = (
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
    )
    rgb_file_path = "data/train_rgb.csv"
    nir_file_path = "data/train_b08.csv"

    additional_targets = {f"image{n}": "image" for n, _ in enumerate(dates[:-1])}
    data_transform = Compose(
        [
            Resize(16, 16),
            VerticalFlip(p=0.3),
            Transpose(p=0.3),
            ShiftScaleRotate(p=0.3),
            RandomRotate90(p=0.3),
            GridDistortion(p=0.3),
            ElasticTransform(p=0.3),
            Normalize(),
        ],
        additional_targets=additional_targets,
    )

    val_transform = Compose(
        [Resize(16, 16), Normalize()], additional_targets=additional_targets
    )

    fold_sets = [[(0, 1, 2), (0, 1, 2)]]
    #fold_sets = [[(1, 2), (0,)], [(0, 2), (1,)], [(0, 1), (2,)]]
    # # KOSTIL' ALERT
    for i, fold_set in enumerate(fold_sets):
        logdir = f"./logs/simple_net_hsv/final"

        trainset = AfricanImageDataset(
            csv_file=rgb_file_path,
            # nir_file=nir_file_path,
            dates=dates,
            root_dir="./",
            transform=data_transform,
            folds=fold_set[0],
        )

        valset = AfricanImageDataset(
            csv_file=rgb_file_path,
            # nir_file=nir_file_path,
            dates=dates,
            root_dir="./",
            folds=fold_set[1],
            transform=val_transform,
        )

        loaders = collections.OrderedDict()
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True, num_workers=num_workers
        )

        valloader = torch.utils.data.DataLoader(
            valset, batch_size=bs, shuffle=True, num_workers=num_workers
        )

        loaders["train"] = trainloader
        loaders["valid"] = valloader

        model = SimpleNetRGB(11, channels_in=3)  # SimpleNetAttentionRGB(11)
        criterion = nn.CrossEntropyLoss()  # FocalLossMultiClass()
        optimizer = torch.optim.Adam(
            model.parameters()
        )  # SGD(model.parameters(), lr=0.01, weight_decay=0.1)

        # model runner
        runner = SupervisedRunner(device="cuda")

        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=logdir,
            num_epochs=num_epochs,
            verbose=True,
            callbacks=[
                CECallback(),
                # LRFinder(final_lr=0.1, num_steps=1000)
                # AccuracyCallback(accuracy_args=[1]),
                # EarlyStoppingCallback(patience=4, min_delta=0.001),
            ],
        )
