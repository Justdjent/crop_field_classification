import collections
import torch
import torch.nn as nn
from albumentations import Compose, Resize
from africa_dataset import AfricanRGBDataset
from simple_net import SimpleNetRGB, SimpleNetAttentionRGB, SimpleNetDeeperRGB, SimpleNet3D
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback
from catalyst.contrib.criterion import FocalLossMultiClass
from ce_callback import CECallback

if __name__ == "__main__":
    bs = 32
    num_workers = 3
    num_epochs = 50
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
    csv_file_path = "data/train_rgb.csv"

    data_transform = Compose([Resize(64, 64)])

    fold_sets = [[(0, 1), (2,)], [(1, 2), (0,)], [(0, 2), (1,)]]
    #
    # # KOSTIL' ALERT
    for i, fold_set in enumerate(fold_sets):

        # fold_set = [(0, 1, 2), (0, 1, 2)]
        logdir = f"./logs/simple_net_3D/fold{i}"

        trainset = AfricanRGBDataset(
            csv_file=csv_file_path,
            dates=dates,
            root_dir="./",
            transform=data_transform,
            folds=fold_set[0],
        )

        valset = AfricanRGBDataset(
            csv_file=csv_file_path,
            dates=dates,
            root_dir="./",
            transform=data_transform,
            folds=fold_set[1],
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

        model = SimpleNet3D(11) #SimpleNetAttentionRGB(11)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        # model runner
        runner = SupervisedRunner(device='cuda')

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
                CECallback(),  # AccuracyCallback(accuracy_args=[1]),
                EarlyStoppingCallback(patience=4, min_delta=0.001),
            ],
        )
