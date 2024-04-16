import argparse
import glob
import os
import random
import math
import numpy as np
import torch
from torch import Tensor
from typing import Any
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import RandomCrop
from class_load import LoadFiles
from pathlib import Path
import yaml


def parse_args():
    desc = "Pytorch Implementation of 'Restormer: Efficient Transformer for High-Resolution Image Restoration'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--data_path", type=str, default="/home/data")
    parser.add_argument(
        "--data_name", type=str, default="rain100L", choices=["rain100L", "rain100H"]
    )
    parser.add_argument("--save_path", type=str, default="result")
    parser.add_argument(
        "--num_blocks",
        nargs="+",
        type=int,
        default=[4, 6, 6, 8],
        help="number of transformer blocks for each level",
    )
    parser.add_argument(
        "--num_heads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="number of attention heads for each level",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=[48, 96, 192, 384],
        help="number of channels for each level",
    )
    parser.add_argument(
        "--expansion_factor",
        type=float,
        # default=2.66,
        default=1,
        help="factor of channel expansion for GDFN",
    )
    parser.add_argument(
        "--num_refinement",
        type=int,
        # default=4,
        default=2,
        help="number of channels for refinement stage",
    )
    parser.add_argument(
        "--num_iter", type=int, default=300000, help="iterations of training"
    )
    parser.add_argument(
        "--batch_size",
        nargs="+",
        type=int,
        # default=[64, 40, 32, 16, 8, 8],
        default=[1, 1, 1, 1, 1, 1],
        help="batch size of loading images for progressive learning",
    )
    parser.add_argument(
        "--patch_size",
        nargs="+",
        type=int,
        # default=[128, 160, 192, 256, 320, 384],
        default=[1, 1, 1, 1, 1, 1],
        help="patch size of each image for progressive learning",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0003, help="initial learning rate"
    )
    parser.add_argument(
        "--milestone",
        nargs="+",
        type=int,
        default=[92000, 156000, 204000, 240000, 276000],
        help="when to change patch size and batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,  # 8,
        help="number of data loading workers",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="random seed (-1 for no manual seed)"
    )
    # model_file is None means training stage, else means testing stage
    parser.add_argument(
        "--model_file", type=str, default=None, help="path of pre-trained model file"
    )

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_name = args.data_name
        self.save_path = args.save_path
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.channels = args.channels
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return Config(args)


def pad_image_needed(img, size):
    width, height = T.get_image_size(img)
    if width < size[1]:
        img = T.pad(img, [size[1] - width, 0], padding_mode="reflect")
    if height < size[0]:
        img = T.pad(img, [0, size[0] - height], padding_mode="reflect")
    return img


class RainDataset(Dataset):
    def __init__(self, data_path, data_name, data_type, patch_size=None, length=None):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = (
            data_name,
            data_type,
            patch_size,
        )
        self.rain_images = sorted(
            glob.glob("{}/{}/{}/rain/*.png".format(data_path, data_name, data_type))
        )
        self.norain_images = sorted(
            glob.glob("{}/{}/{}/norain/*.png".format(data_path, data_name, data_type))
        )
        # make sure the length of training and testing different
        self.num = len(self.rain_images)
        self.sample_num = length if data_type == "train" else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        image_name = os.path.basename(self.rain_images[idx % self.num])
        rain = T.to_tensor(Image.open(self.rain_images[idx % self.num]))
        norain = T.to_tensor(Image.open(self.norain_images[idx % self.num]))
        h, w = rain.shape[1:]

        if self.data_type == "train":
            # make sure the image could be cropped
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(
                rain, (self.patch_size, self.patch_size)
            )
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
            if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
        else:
            # padding in case images are not multiples of 8
            new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
            pad_h = new_h - h if h % 8 != 0 else 0
            pad_w = new_w - w if w % 8 != 0 else 0
            rain = F.pad(rain, (0, pad_w, 0, pad_h), "reflect")
            norain = F.pad(norain, (0, pad_w, 0, pad_h), "reflect")
        return rain, norain, image_name, h, w


def rgb_to_y(x):
    rgb_to_grey = torch.tensor(
        [0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device
    ).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = -10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords**2
    g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma**2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1**2, k2**2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_xx = (
        F.conv2d(x**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    )
    sigma_yy = (
        F.conv2d(y**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    )
    sigma_xy = (
        F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy
    )

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()


def create_dataloader(dataset: Dataset, dataset_opt: dict) -> DataLoader:
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt["phase"]
    if phase == "train":
        dataloader_args = dict(
            dataset=dataset,
            batch_size=dataset_opt["batch_size"],
            shuffle=dataset_opt["shuffle"],
            num_workers=dataset_opt["num_worker"],
            drop_last=True,
        )
        print(dataloader_args)

    elif phase in ["val", "test"]:  # validation
        dataloader_args = dict(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=0
        )
    else:
        raise ValueError(
            f"Wrong dataset phase: {phase}. "
            "Supported ones are 'train', 'val' and 'test'."
        )

    # dataloader_args["pin_memory"] = dataset_opt.get("pin_memory", False)

    #  prefetch_mode=None: Normal dataloader
    # prefetch_mode='cuda': dataloader for CUDAPrefetcher
    return DataLoader(**dataloader_args, pin_memory=True)


class CustomDataset(Dataset):
    def __init__(
        self, dataset_path: Path | str, channel: int = 3, train: bool = True
    ) -> None:
        handler_files = LoadFiles()

        if train:
            self.noisy_path = Path(dataset_path).joinpath("Noisy/train")
            self.gt_path = Path(dataset_path).joinpath("GTruth/train")
        else:
            self.noisy_path = Path(dataset_path).joinpath("Noisy/test")
            self.gt_path = Path(dataset_path).joinpath("GTruth/test")
        self._noisy_images = handler_files.search_load_files_extencion(
            path_search=self.noisy_path.as_posix(), ext=["png"]
        )["png"][1]
        self._gt_imges = handler_files.search_load_files_extencion(
            path_search=self.gt_path.as_posix(), ext=["png"]
        )["png"][1]
        self.channel = channel

    def __getitem__(self, index) -> tuple[Tensor, Tensor, str, int, int]:
        if self.channel == 3:
            img_gt: Tensor = read_image(
                self._gt_imges[index], mode=ImageReadMode.RGB
            )  # ImageReadMode.UNCHANGED
            img_lq: Tensor = read_image(
                self._noisy_images[index], mode=ImageReadMode.RGB
            )
        elif self.channel == 1:
            img_gt: Tensor = read_image(self._gt_imges[index], mode=ImageReadMode.GRAY)
            img_lq: Tensor = read_image(
                self._noisy_images[index], mode=ImageReadMode.GRAY
            )
        h = img_gt.size(dim=0)
        w = img_gt.size(dim=1)
        # d = img_gt.size(dim=2)
        return img_lq.float(), img_gt.float(), self._noisy_images[index], h, w

    def __len__(self):
        return len(self._noisy_images)


def load_dataloader(opt):
    if opt["dataloader"]["phase"] == "train":
        sar_dataset = CustomDataset(
            dataset_path=opt["preprocessing"]["output"], channel=3, train=True
        )
        dataloader = create_dataloader(
            dataset=sar_dataset, dataset_opt=opt["dataloader"]
        )
    elif opt["dataloader"]["phase"] == "val":
        sar_dataset = CustomDataset(
            dataset_path=opt["preprocessing"]["output"], channel=3, train=False
        )
        dataloader = create_dataloader(
            dataset=sar_dataset,
            dataset_opt=opt["dataloader"],
        )
    num_iter_per_epoch = math.ceil(
        len(sar_dataset) / opt["dataloader"]["batch_size_per_gpu"]
    )
    total_iters = int(opt["dataloader"]["iters"])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))

    return dataloader, total_epochs, total_iters
