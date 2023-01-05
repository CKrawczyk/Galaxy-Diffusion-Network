'''Turn images of letters into galaxies'''

import numpy as np
import torch
import os

from PIL import Image, ImageDraw, ImageFont
from torch.utils import data
from torchvision.transforms import transforms
from torchvision import utils
from denoising_diffusion_pytorch.classifier_free_guidance import Unet
from dataset_trainer import TrainerClass, GaussianDiffusionExtension
from galaxy_mnist import GalaxyMNISTHighrez as GalaxyMNISTOrigHighrez, read_dataset_file


class GalaxyMNISTHighrez(GalaxyMNISTOrigHighrez):
    # Subclass GalaxyMNIST so it returns both the training and test data as a single set
    def _load_data(self):
        # load in all training and test data into one set
        images_train, targets_train = read_dataset_file(
            os.path.join(self.raw_folder, 'train_dataset.hdf5')
        )
        images_test, targets_test = read_dataset_file(
            os.path.join(self.raw_folder, 'test_dataset.hdf5')
        )
        all_images = torch.cat([images_train, images_test], axis=0)
        all_labels = torch.cat([targets_train, targets_test], axis=0)
        return all_images, all_labels


def text_phantom(text, size):
    '''Create an image of text'''

    # Availability is platform dependent
    # point to local font file
    font = '/mnt/lustre/shared_conda/envs/ckraw/gz-torch/fonts/DejaVuSans.ttf'

    # Create font
    pil_font = ImageFont.truetype(font, size=size // len(text), encoding="unic")
    text_left, text_top, text_right, text_bottom = pil_font.getbbox(text)
    text_width = text_right - text_left
    text_height = text_bottom - text_top + 75

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [size, size], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = (
        (size - text_width) // 2,
        ((size - text_height) // 2)
    )
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    return 255 - np.asarray(canvas)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 32
milestone = 30
num_classes = 4

if __name__ == '__main__':
    # transforms for images
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )

    root = '../gzmnist_data'
    dataset = GalaxyMNISTHighrez(
        root=root,
        download=True,
        train=True,
        transform=transform
    )
    logdir = 'logs/gzmnist_high_res_x0_l2_with_classes'
    train_batch_size = 20

    # 500 is enough to fully noise up an image
    timesteps = 1500
    # use DDIP for sampling faster
    sampling_timesteps = 50

    image_size = dataset.data.shape[-1]

    dataloader = data.DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=num_classes,
        cond_drop_prob=0.5
    ).to(device=DEVICE)

    diffusion = GaussianDiffusionExtension(
        model,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=0,
        loss_type='l2',
        objective='pred_x0'
    ).to(device=DEVICE)

    trainer = TrainerClass(
        diffusion,
        num_classes,
        dataloader,
        train_batch_size=train_batch_size,
        results_folder=logdir,
        train_lr=2e-5,
        train_num_steps=30000,          # total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        save_and_sample_every=1000,
        num_samples_per_class=5,
        split_batches=False
    )
    trainer.load(milestone)

    SAMPLES_PER_CLASS = 5
    T = 1450

    transform_letter = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    for LETTER in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        print(LETTER)
        letter = text_phantom(LETTER, image_size)
        letter_tensor = transform_letter(letter)
        img = torch.stack([letter_tensor] * SAMPLES_PER_CLASS * num_classes)

        classes = torch.arange(num_classes)
        classes = classes.repeat(SAMPLES_PER_CLASS)
        all_images = trainer.ema.ema_model.q_than_p(
            img.to(DEVICE),
            classes.to(DEVICE),
            T,
            sampling_timesteps=30,
            cond_scale=6.0
        )

        utils.save_image(all_images, f'letters/{LETTER}_{T}_galaxies.png', nrow=num_classes)
