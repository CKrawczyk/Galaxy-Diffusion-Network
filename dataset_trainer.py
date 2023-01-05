from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    Trainer,
    Path,
    Adam,
    EMA,
    Accelerator,
    has_int_squareroot,
    cycle,
    tqdm,
    torch,
    utils,
    num_to_groups,
    unnormalize_to_zero_to_one
)

from denoising_diffusion_pytorch.classifier_free_guidance import GaussianDiffusion as GaussianDiffusionWithClasses


# make a subclass of Trainer that uses a torch Dataloader rather than a folder of images
class TrainerDS(Trainer):
    def __init__(
        self,
        diffusion_model,
        dataloader,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results',
        amp=False,
        fp16=False,
        split_batches=True
    ):
        super(Trainer).__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        # use the dataloader that is passed in
        dl = dataloader

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(
            diffusion_model.parameters(),
            lr=train_lr,
            betas=adam_betas
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model,
                beta=ema_decay,
                update_every=ema_update_every
            )

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)


# Create a trainer for diffusion with class labels passed in
class TrainerClass(Trainer):
    def __init__(
        self,
        diffusion_model,
        num_classes,
        dataloader,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples_per_class=5,
        results_folder='./results',
        amp=False,
        fp16=False,
        split_batches=True
    ):
        super(Trainer).__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model
        self.num_classes = num_classes

        # number of samples to take from each class
        self.num_samples_per_class = num_samples_per_class
        self.num_samples = self.num_samples_per_class * self.num_classes
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        # use the dataloader passed in
        dl = dataloader

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(
            diffusion_model.parameters(),
            lr=train_lr,
            betas=adam_betas
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model,
                beta=ema_decay,
                update_every=ema_update_every
            )

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def train(self):
        # updated the training loop to use the class data
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    images = data[0].to(device)
                    # pull out the labels
                    labels = data[1].to(device)

                    with self.accelerator.autocast():
                        # pass labels into the model
                        loss = self.model(images, classes=labels)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            # samples from each class
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            # if not divisible by num_classes make more than needed
                            classes = torch.arange(self.num_classes)
                            classes = classes.repeat(self.num_samples_per_class)
                            # limit to the number needed (split into groups of batch size)
                            class_batches = []
                            start = 0
                            for b in batches:
                                end = start + b
                                class_batches.append(classes[start:end])
                                start = end
                            all_images_list = list(map(
                                lambda c: self.ema.ema_model.sample(
                                    classes=c.to(device),
                                    cond_scale=3.0
                                ),
                                class_batches
                            ))

                        all_images = torch.cat(all_images_list, dim=0)
                        utils.save_image(
                            all_images,
                            str(self.results_folder / f'sample-{milestone}.png'),
                            nrow=self.num_classes
                        )
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')


class GaussianDiffusionExtension(GaussianDiffusionWithClasses):
    @torch.no_grad()
    def ddim_sample_from_image(
        self,
        classes,
        img,
        cond_scale=3.0,
        total_timesteps=None,
        sampling_timesteps=None,
        clip_denoised=True
    ):
        # sample starting from an image
        # noise the input by total_timesteps
        # denoise it by sampling_timesteps using DDIM
        batch = img.shape[0]
        device = self.betas.device
        if total_timesteps is None:
            total_timesteps = self.num_timesteps
        if sampling_timesteps is None:
            sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                classes,
                cond_scale=cond_scale,
                clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def q_than_p(self, img, classes, t, **kwargs):
        # helper class to noise up and image by `t` and denoise it
        device = self.betas.device
        T = torch.tensor([t]).to(device)
        img_with_noise = self.q_sample(img, t=T)
        img_denoise = self.ddim_sample_from_image(
            classes,
            img_with_noise,
            total_timesteps=t,
            **kwargs
        )
        return img_denoise
