import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def visualize_vae(epoch, vae, dataset, device, n=8):
    vae.eval()
    
    x = torch.stack([dataset[i]["image"] for i in range(n)]).to(device)
    x_hat, _, _ = vae(x)

    x = x.cpu()
    x_hat = x_hat.cpu()

    fig, axes = plt.subplots(2, n, figsize=(n*1.5, 3))

    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(x_hat[i, 0], cmap="gray")
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.savefig(f"./Img/VAE_epoch_{epoch}_reconstruction.png")
    plt.close()


@torch.no_grad()
def visualize_latent(epoch, vae, dataset, device, n=8):
    vae.eval()

    x = torch.stack([dataset[i]["image"] for i in range(n)]).to(device)
    mu, _ = vae.encoder(x)

    mu = mu.cpu()

    fig, axes = plt.subplots(n, mu.shape[1], figsize=(mu.shape[1]*1.2, n*1.2))

    for i in range(n):
        for c in range(mu.shape[1]):
            axes[i, c].imshow(mu[i, c], cmap="gray")
            axes[i, c].axis("off")

    plt.suptitle("Latent channels (mu)")
    plt.savefig(f"./Img/VAE_epoch_{epoch}_latent_vis.png")
    plt.close()

@torch.no_grad()
def sample_latent(unet, scheduler, shape, device):
    z = torch.randn(shape).to(device)

    for t in reversed(range(scheduler.timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device)

        noise_pred = unet(z, t_tensor)
        alpha = scheduler.alphas[t]
        alpha_bar = scheduler.alpha_bars[t]

        z = (
            1 / torch.sqrt(alpha) *
            (z - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred)
        )

        if t > 0:
            z += torch.sqrt(scheduler.betas[t]) * torch.randn_like(z)

    return z



@torch.no_grad()
def visualize_samples(epoch, latent_unet, scheduler, vae, device):
    z = sample_latent(
        latent_unet,
        scheduler,
        shape=(16, 4, 8, 8),
        device=device,
    )

    imgs = vae.decoder(z).cpu()
    imgs = (imgs - imgs.min() ) / (imgs.max() - imgs.min())

    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i, 0], cmap="gray")
        ax.axis("off")

    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f"./Img/Resample_ep_{epoch}.png")
    plt.close()


@torch.no_grad()
def sample_latent_conditional(unet, scheduler, context, shape, device):
    z = torch.randn(shape, device=device)

    for t in reversed(range(scheduler.timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = unet(z, t_tensor, context)

        alpha = scheduler.alphas[t]
        alpha_bar = scheduler.alpha_bars[t]
        beta = scheduler.betas[t]

        z = (1 / torch.sqrt(alpha)) * (
            z - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps
        )

        if t > 0:
            z = z + torch.sqrt(beta) * torch.randn_like(z)

    return z



@torch.no_grad()
def visualize_digits(
    epoch,
    latent_unet,
    vae,
    text_encoder,
    scheduler,
    device,
    latent_shape=(1, 4, 8, 8)
):
    latent_unet.eval()
    vae.eval()
    text_encoder.eval()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()

    for digit in range(10):
        # 1️⃣ label → text embedding
        label = torch.tensor([digit], device=device)
        context = text_encoder(label)   # (1, 1, D)

        # 2️⃣ latent diffusion sampling
        z = sample_latent_conditional(
            latent_unet,
            scheduler,
            context,
            latent_shape,
            device
        )

        # 3️⃣ latent → image
        img = vae.decoder(z).cpu()
        img = (img - img.min() ) / (img.max() - img.min())

        # 4️⃣ plot
        axes[digit].imshow(img[0, 0], cmap="gray")
        axes[digit].set_title(f"{digit}")
        axes[digit].axis("off")

    plt.suptitle("MNIST Text-to-Image (Digit Conditioning)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./Img/DigitCondition_epoch_{epoch}.png")
    plt.close()


@torch.no_grad()
def sample_latent_cfg(
    unet,
    scheduler,
    text_encoder,
    labels,
    shape,
    guidance_scale=5.0,
    device="cuda"
):
    z = torch.randn(shape, device=device)

    # context 준비
    cond_context = text_encoder(labels)
    uncond_labels = torch.full_like(labels, 10)
    uncond_context = text_encoder(uncond_labels)

    for t in reversed(range(scheduler.timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        eps_uncond = unet(z, t_tensor, uncond_context)
        eps_cond = unet(z, t_tensor, cond_context)

        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        alpha = scheduler.alphas[t]
        alpha_bar = scheduler.alpha_bars[t]
        beta = scheduler.betas[t]

        z = (1 / torch.sqrt(alpha)) * (
            z - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps
        )

        if t > 0:
            z = z + torch.sqrt(beta) * torch.randn_like(z)

    return z


def visualize_digits_cfg(
    epoch,
    latent_unet,
    vae,
    text_encoder,
    scheduler,
    device,
    guidance_scale=5.0,
    latent_scale=1.0
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()

    for digit in range(10):
        labels = torch.tensor([digit], device=device)

        z = sample_latent_cfg(
            latent_unet,
            scheduler,
            text_encoder,
            labels,
            shape=(1, 4, 8, 8),
            guidance_scale=guidance_scale,
            device=device
        )

        z = z / latent_scale
        img = vae.decoder(z).cpu()

        axes[digit].imshow(img[0, 0], cmap="gray")
        axes[digit].set_title(f"{digit}")
        axes[digit].axis("off")

    plt.suptitle(f"CFG Sampling (scale={guidance_scale})")
    plt.tight_layout()
    plt.savefig(f"./Img/DigitCondition_epoch_{epoch}_guidance_scale_{guidance_scale}.png")
    plt.close()
