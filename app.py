import os
import shutil
import click
from loguru import logger
from lib import sam_model
from os.path import basename, splitext


@click.group()
def cli():
    """
    Main command group for the CLI application.
    """
    pass


@cli.command()
@click.option('--device', help='Device: "cpu", "cuda" or "auto"', default='auto')
@click.option('--save-masks', is_flag=True, help='Enable saving masks')
@click.option('--alpha-image', type=click.FloatRange(0.0, 1.0), default=0.0, help='Alpha value for image')
@click.option('--alpha-mask', type=click.FloatRange(0.0, 1.0), default=1.0, help='Alpha value for mask')
@click.argument('dir_in', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.argument('dir_out', type=click.Path(file_okay=False, resolve_path=True))
def maskgen(dir_in, dir_out, device, alpha_image, alpha_mask, save_masks):
    """
    Command for generating masks.

    Args:
        device (string): cpu or cuda
        save_masks (bool): Enable saving masks.
        alpha_image (float): Alpha value for the image.
        alpha_mask (float): Alpha value for the mask.
        dir_in (str): Input directory path.
        dir_out (str): Output directory path.
    """
    logger.add("app.log", rotation="500 MB")  # Loguru configuration

    if device == 'auto':
        device = sam_model.get_device()
    logger.info(f"Using device {device}")
    logger.info("Loading SAM model ...")
    model = sam_model.load_sam(device=device)
    logger.info("Model loaded successfully")
    logger.info("Creating mask generator ...")
    generator = sam_model.get_mask_generator(model)
    logger.info("Msk generator created successfully")

    for file_name in os.listdir(dir_in):
        name, ext = splitext(basename(file_name))
        file_path_src = os.path.join(dir_in, file_name)
        file_path_dest = os.path.join(dir_out, name + "_mask" + ext)
        if os.path.isfile(file_path_src) and file_path_src.lower().endswith(('.jpg', '.png')):
            logger.info(f"Processing file: {file_name}")
            image = sam_model.load_image(file_path_src)
            logger.info("Image loaded")
            logger.info("Generating masks ...")
            masks = sam_model.generate_masks(image, generator)
            logger.info("Masks generated successfully")
            sam_model.generate_mask_image(image, masks, file_path_dest)
            logger.info(f"Saved resulting image to {file_path_dest}")

    logger.info("Processing complete.")


@cli.command()
def check_model():
    """
    Command for checking if the model file exists.
    """
    model_path = os.path.expanduser("~/.pytorch/SAM/sam_vit_h_4b8939.pth")

    if os.path.isfile(model_path):
        logger.info("Model was found.")
    else:
        logger.error("Model was not found.")


if __name__ == '__main__':
    cli()
