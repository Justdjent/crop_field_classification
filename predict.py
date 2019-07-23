from catalyst.dl.callbacks import InferCallback
import collections
import torch
import torch.nn as nn
from albumentations import Compose, Resize, Normalize
from africa_dataset import AfricanImageDataset, AfricanNIRDataset
from simple_net import SimpleNetRGB
from catalyst.dl import SupervisedRunner, CheckpointCallback
import pandas as pd
import numpy as np
from scipy.special import softmax


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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
model_name = "simple_net_nocrop_16"
logdir = f"./logs/simple_net_nocrop_16/fold2"

ids = pd.read_csv("data/train_rgb.csv")
ids = ids[ids['Fold'] == 2]["Field_Id"].values

additional_targets = {f"image{n}": "image" for n, _ in enumerate(dates[:-1])}
data_transform = Compose(
    [Resize(16, 16), Normalize()], additional_targets=additional_targets
)

testset = AfricanImageDataset(
    csv_file=csv_file_path,
    dates=dates,
    root_dir="./",
    transform=data_transform,
    folds=(2,),
)

dataset_length = len(testset)

loaders = collections.OrderedDict()
testloader = torch.utils.data.DataLoader(testset, shuffle=False)

model = SimpleNetRGB(11)  # , channels_in=1)
runner = SupervisedRunner(device="cuda")

loaders["valid"] = testloader
loaders = collections.OrderedDict([("infer", loaders["valid"])])
runner.infer(
    model=model,
    loaders=loaders,
    callbacks=[
        InferCallback(),
        CheckpointCallback(resume=f"{logdir}/checkpoints/best.pth"),
    ],
)

predictions = runner.callbacks[0].predictions["logits"].reshape(dataset_length, 9)
predictions = sigmoid(predictions)
predictions = np.argmax(predictions, axis=1) + 1

pred_frame = {"Field_Id": ids, "Predictions": predictions}

pred_frame = pd.DataFrame(pred_frame)
pred_frame.to_csv(f"data/validation/{model_name}_valid.csv", index=False)
