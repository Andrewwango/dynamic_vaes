import torch
import numpy as np
from pytorch_cvae import CVAE, load_bouncing_ball
from pytorch_cvae_model import ConvVAE

if __name__ == '__main__':
    print("create dataloaders")
    batch_size = 64
    train_dataset, val_dataset =  load_bouncing_ball("../bouncing_ball_data", "box", singular=True)
    train_dataset = train_dataset[np.random.choice(5000, 2560)]
    val_dataset = val_dataset[np.random.choice(1000, 512)]
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = 4)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle = True, num_workers = 4)


    model = ConvVAE(kernel_size=4,
                    init_channels=8,
                    image_channels=1,
                    latent_dim=16)

    cvae = CVAE(model=model,
                lr = 1e-3,
                epochs = 50,
                batch_size = 64)

    cvae.train(train_dataloader, val_dataloader)