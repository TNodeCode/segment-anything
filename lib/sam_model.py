from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import sys
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np


def show_anns(anns, ax=None, alpha=0.35):
    if len(anns) == 0:
        raise Exception("No annotations found")
    if ax is None:
        raise Exception("You must provide an axis object")
    if alpha < 0.0 or alpha > 1.0:
        raise Exception("alpha must be between 0.0 and 1.0")
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                  sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[m] = color_mask
    ax.imshow(img)


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sam(device="cpu", model_type="vit_h"):
    sam_checkpoint = os.path.expanduser("~/.pytorch/SAM/sam_vit_h_4b8939.pth")
    model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    model.to(device=device)
    return model


def get_mask_generator(model):
    mask_generator = SamAutomaticMaskGenerator(model)
    return mask_generator


def generate_masks(image, generator):
    return generator.generate(image)


def show_anns(anns, ax=None, alpha=0.35):
    if len(anns) == 0:
        raise Exception("No annotations found")
    if ax is None:
        raise Exception("You must provide an axis object")
    if alpha < 0.0 or alpha > 1.0:
        raise Exception("alpha must be between 0.0 and 1.0")
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                  sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[m] = color_mask
    ax.imshow(img)


def generate_mask_image(image, masks, file_dest, alpha_image=0.0, alpha_mask=1.0):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    display_axis = False
    ax.imshow(image)
    ax.imshow(image * alpha_image)
    show_anns(masks, ax=ax, alpha=alpha_mask)
    ax.axis('on' if display_axis else 'off')
    plt.savefig(file_dest, bbox_inches='tight')
