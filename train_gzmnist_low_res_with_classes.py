'''Train low res galaxyMNIST dataset with class labels'''

import torch
import os

from torch.utils import data
from torchvision.transforms import transforms
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion
from dataset_trainer import TrainerClass
from galaxy_mnist import GalaxyMNIST as GalaxyMNISTOrig, read_dataset_file


class GalaxyMNIST(GalaxyMNISTOrig):
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 32
milestone = 0
num_classes = 4


if __name__ == '__main__':
    # must be in '__main__' to use num_workers
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )

    root = '../gzmnist_data'
    dataset = GalaxyMNIST(
        root=root,
        download=True,
        train=True,
        transform=transform
    )
    logdir = 'logs/gzmnist_low_res_x0_l2_with_classes'
    train_batch_size = 128

    # 500 is enough to fully noise up an image
    timesteps = 500
    # use DDIP for sampling faster
    sampling_timesteps = 150

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

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
        objective='pred_x0'
    ).to(device=DEVICE)

    trainer = TrainerClass(
        diffusion,
        dataloader,
        train_batch_size=train_batch_size,
        results_folder=logdir,
        train_lr=2e-5,
        train_num_steps=10000,          # total training steps
        gradient_accumulate_every=1,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        save_and_sample_every=500,
        num_samples=16,
        split_batches=False
    )

    if milestone != 0:
        trainer.load(milestone)

    trainer.train(num_classes)
