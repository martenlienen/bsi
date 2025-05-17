#!/usr/bin/env python

import argparse
import math
from pathlib import Path

import einops as eo
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_file")
    parser.add_argument("out_image")
    args = parser.parse_args()

    samples_path = Path(args.samples_file)
    out_image = args.out_image

    data = torch.load(samples_path)
    samples = data["samples"]
    n = len(samples)
    largest_divisor = max(i for i in range(1, int(math.sqrt(n)) + 2) if n % i == 0)
    grid = eo.rearrange(samples, "(a b) c h w -> (a h) (b w) c", b=largest_divisor)

    grid = (grid * 255).to(torch.uint8)

    images = Image.fromarray(grid.numpy())
    images.save(out_image)


if __name__ == "__main__":
    main()
