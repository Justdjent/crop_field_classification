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
    RandomCrop
)
from africa_dataset import AfricanImageDataset, AfricanRGBDataset, AfricanNIRDataset, AfricanMultibandDataset, \
    AfricaPaddedDataset
import argparse
from simple_net import (
    SimpleNetRGB,
    SimpleNetAttentionRGB,
    SimpleNetDeeperRGB,
    SimpleNet3D,
    ModerateNetRGB
)
from resnet import resnet18
from catalyst.dl import SupervisedRunner, CheckpointCallback, AccuracyCallback
from catalyst.dl.callbacks import EarlyStoppingCallback, LRFinder
from catalyst.contrib.criterion import FocalLossMultiClass
from ce_callback import CECallback
from adamw import AdamW
from utils import AerageMeanAccuracyCallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='data',
                        help='path to dataset')
    parser.add_argument("--bs", type=int, default=32, help='batch_size')
    parser.add_argument("--num_workers", default=3)
    parser.add_argument("--num_epochs", default=35)
    args = parser.parse_args()
    bs = args.bs
    num_workers = args.num_workers
    num_epochs = args.num_epochs
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
    rgb_file_path = f"{args.root}/train_rgb.csv"
    nir_file_path = f"{args.root}/train_b08.csv"

    additional_targets = {f"image{n}": "image" for n, _ in enumerate(dates[:-1])}
    data_transform = Compose(
        [
            RandomCrop(128, 128),
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
        [RandomCrop(128, 128), 
         Normalize()], additional_targets=additional_targets
    )

    #fold_sets = [[(0, 1, 2), (0, 1, 2)]]
    fold_sets = [[(1, 2), (0,)], [(0, 2), (1,)], [(0, 1), (2,)]]
    # # KOSTIL' ALERT
    for i, fold_set in enumerate(fold_sets):
        logdir = f"./logs/3d_resnet_random_pad/fold{i}"

        trainset = AfricaPaddedDataset(
            csv_file=rgb_file_path,
            # nir_file=nir_file_path,
            dates=dates,
            root_dir="./",
            transform=data_transform,
            folds=fold_set[0],
        )

        valset = AfricaPaddedDataset(
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

        # model = resnet18(num_classes=9, sample_size=3, sample_duration=11)
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
                AerageMeanAccuracyCallback(accuracy_args=[1, 3, 5])
                # CheckpointCallback(resume=f"pretrained/resnet-18-fixed.pth")
            ],
        )
