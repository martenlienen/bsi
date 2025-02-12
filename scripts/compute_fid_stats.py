#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import torch
import torchmetrics as tm
from hydra.utils import instantiate
from tqdm import tqdm

from bsi.utils.hydra import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("dataset")
    parser.add_argument("split")
    args = parser.parse_args()

    device = args.device
    dataset = args.dataset
    split = args.split

    if not torch.cuda.is_available():
        device = "cpu"

    config = load_config(f"data={dataset} data.eval_batch_size=512 data.batch_size=512")
    datamodule = instantiate(config.data, seed=0)

    fid = tm.image.FrechetInceptionDistance(
        feature=2048, reset_real_features=True, normalize=True
    )
    fid.to(device=device)

    datamodule.prepare_data()
    if split == "train":
        datamodule.setup("fit")
        # If the dataset does not have a fixed train split and we have split it into
        # train and val arbitrarily, compute the FID stats on the complete training data
        loader = datamodule.fid_train_dataloader()
    elif split == "val":
        datamodule.setup("validate")
        loader = datamodule.val_dataloader()
    elif split == "test":
        datamodule.setup("test")
        loader = datamodule.test_dataloader()

    # If we also val/test on train data, discard the second loader
    if isinstance(loader, list):
        loader = loader[0]

    discretization = datamodule.discretization()
    for data, _ in tqdm(loader):
        data = data.to(device=device)
        fid.update(discretization.to_unit_interval(data), real=True)

    n = fid.real_features_num_samples.numpy(force=True)
    sum = fid.real_features_sum.numpy(force=True)
    cov_sum = fid.real_features_cov_sum.numpy(force=True)

    data_name = datamodule.short_name()
    out = Path(__file__).parent.parent / "data" / "fid-stats" / data_name / f"{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, n=n, sum=sum, cov_sum=cov_sum)


if __name__ == "__main__":
    main()
