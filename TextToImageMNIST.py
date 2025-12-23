import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from models.VAE import MNISTVAE, reparameterize
from models.DiffusionUNet import DiffusionUNet
from models.TextEncoder import DigitTextEncoder
from losses.VAELoss import vae_loss
from utils.NoiseUtils import LatentNoiseScheduler
from utils.Resampling import visualize_samples, visualize_vae, visualize_latent, visualize_digits, visualize_digits_cfg

transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

class MNISTTextDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.data = list(mnist_dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        text = f"{label}을 그려줘"
        return {
            "image": image,          # (1, 28, 28)
            "text": text,
            "label": label
        }

dataset = MNISTTextDataset(mnist)

device = "cuda"

vae = MNISTVAE().to(device)
optimizer = torch.optim.AdamW(vae.parameters(), lr=5e-4)

loader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True
)

for epoch in range(10):
    for batch in loader:
        x = batch["image"].to(device)

        x_hat, mu, logvar = vae(x)
        loss = vae_loss(x, x_hat, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch}, loss {loss.item():.4f}")


vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# Calculate latent scaling factor
with torch.no_grad():
    sample_batch = next(iter(loader))
    sample_x = sample_batch["image"][:100].to(device)
    mu, logvar = vae.encoder(sample_x)
    sample_z = reparameterize(mu, logvar)
    latent_scale = 1.0 / sample_z.std()
    print(f"Latent scaling factor: {latent_scale:.4f}")


context_dim = 32
latent_unet = DiffusionUNet(time_dim=32, context_dim=context_dim).to(device)
text_encoder = DigitTextEncoder(embed_dim=context_dim).to(device)

optimizer = torch.optim.AdamW(
    [
        {"params": latent_unet.parameters(), "lr": 5e-4},
        {"params": text_encoder.parameters(), "lr": 5e-4},
    ]
)
scheduler = LatentNoiseScheduler()

for epoch in range(50):
    for batch in loader:
        x = batch["image"].to(device)
        

        with torch.no_grad():
            mu, logvar = vae.encoder(x)
            z0 = reparameterize(mu, logvar) * latent_scale
                        
        t = torch.randint(
            0, scheduler.timesteps, (z0.size(0),),
            device=device
        )
        zt, noise = scheduler.add_noise(z0, t)
        
        labels = batch["label"].to(device)
        drop_prob = 0.1
        mask = torch.rand(labels.shape, device=device) < drop_prob
        labels = labels.clone()
        labels[mask] = 10
        context = text_encoder(labels)

        noise_pred = latent_unet(zt, t, context)
        
        loss = ((noise - noise_pred) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[epoch {epoch}] diffusion loss: {loss.item():.4f}")
    visualize_digits_cfg(
        epoch,
        latent_unet,
        vae,
        text_encoder,
        scheduler,
        device,
        guidance_scale=3.0,
        latent_scale=latent_scale
    )