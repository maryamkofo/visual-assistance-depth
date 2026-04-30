import numpy as np
import os
from tqdm import tqdm
import cv2
import h5py

import sys
sys.path.append('/Depth-Anything-V2/metric_depth')

import torch
import torch.nn.functional as F

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.dataset.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

import json


def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


val_paths = get_all_files(r'(path to DepthAnything dir)\Depth-Anything-V2\nyudepthv2\val')


class NYU(torch.utils.data.Dataset):
    def __init__(self, paths, size=(518, 518)):
        self.size = size
        self.paths = paths

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def __getitem__(self, item):
        path = self.paths[item]
        image, depth = self.h5_loader(path)
        image = image / 255.0
        sample = self.transform({'image': image, 'depth': depth})
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        return sample

    def __len__(self):
        return len(self.paths)

    def h5_loader(self, path):
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth'])
        return rgb, depth


def get_val_dataloader():
    val_dataset = NYU(val_paths)
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))
    d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    mae = torch.mean(torch.abs(diff))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.detach(), 'abs_rel': abs_rel.detach(), 'rmse': rmse.detach(), 'mae': mae.detach(), 'silog': silog.detach()}


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
model_encoder = 'vits'
max_depth = 10


def eval_only():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DepthAnythingV2(**{**model_configs[model_encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load('NYUmodel.pth', map_location=device))
    model.eval().to(device)

    val_dataloader = get_val_dataloader()

    results = {'d1': 0, 'abs_rel': 0, 'rmse': 0, 'mae': 0, 'silog': 0}

    for sample in tqdm(val_dataloader):
        img, depth = sample['image'].float().to(device), sample['depth'][0].to(device)

        with torch.no_grad():
            pred = model(img)
            pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

        valid_mask = (depth <= max_depth) & (depth >= 0.001)
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        for k in results.keys():
            results[k] += cur_results[k]

    for k in results.keys():
        results[k] = round((results[k] / len(val_dataloader)).item(), 4)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    os.makedirs('./nyu_eval_results', exist_ok=True)
    with open('./nyu_eval_results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to ./nyu_eval_results/metrics.json")


if __name__ == '__main__':
    eval_only()