import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


# def crop_center(img, mask, size, ignore_value=255):
#     w, h = img.size
#     padw = size - w if w < size else 0
#     padh = size - h if h < size else 0

#     pad_h_half = int(padh / 2)
#     pad_w_half = int(padw / 2)
    
#     img = ImageOps.expand(img, border=(pad_w_half, pad_h_half, padw - pad_w_half, padh - pad_h_half), fill=0)
#     mask = ImageOps.expand(mask, border=(pad_w_half, pad_h_half, padw - pad_w_half, padh - pad_h_half), fill=ignore_value)

#     w, h = img.size

#     h_off = (h - size) // 2
#     w_off = (w - size) // 2

#     img = img.crop((w_off, h_off, w_off + size, h_off + size))
#     mask = mask.crop((w_off, h_off, w_off + size, h_off + size))

#     return img, mask, np.array([pad_h_half, pad_w_half])

def crop_center(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0

    pad_h_half = int(padh / 2)
    pad_w_half = int(padw / 2)
    
    img = ImageOps.expand(img, border=(pad_w_half, pad_h_half, padw - pad_w_half, padh - pad_h_half), fill=0)
    mask = ImageOps.expand(mask, border=(pad_w_half, pad_h_half, padw - pad_w_half, padh - pad_h_half), fill=ignore_value)

    w, h = img.size

    h_off = (h - size) // 2
    w_off = (w - size) // 2

    img = img.crop((w_off, h_off, w_off + size, h_off + size))
    mask = mask.crop((w_off, h_off, w_off + size, h_off + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.long()


def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][
        : len(labels) // 2
    ]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


def generate_unsup_data(data, target, logits, mode="cutout"):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        if mode == "cutout":
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
            target[i][(1 - mix_mask).bool()] = 255

            new_data.append((data[i] * mix_mask).unsqueeze(0))
            new_target.append(target[i].unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue

        if mode == "cutmix":
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        if mode == "classmix":
            mix_mask = generate_class_mask(target[i]).to(device)

        new_data.append(
            (
                data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_target.append(
            (
                target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_logits.append(
            (
                logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )

    new_data, new_target, new_logits = (
        torch.cat(new_data),
        torch.cat(new_target),
        torch.cat(new_logits),
    )
    return new_data, new_target.long(), new_logits