import torch
import numpy as np
from kvae_model import KVAEModel
from kvae import KVAE
from auxiliary import load_bouncing_ball

if __name__ == '__main__':
    print("create dataloaders")
    batch_size = 64
    train_dataset, val_dataset =  load_bouncing_ball("nonlinear_ball_data", "elliptical", binary=False)
    train_dataset = train_dataset[np.random.choice(5000, 2560)]
    #val_dataset = val_dataset[np.random.choice(1000, 512)]
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = 2)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle = True, num_workers = 2)


    model = KVAEModel(x_dim = 32 * 32, 
                a_dim = 4,#32
                z_dim = 4, #16
                x_2d=True,

                init_kf_mat = 0.05,
                noise_transition = 0.08,
                noise_emission = 0.03,
                init_cov = 20,

                K = 1,
                dim_RNN_alpha = 50,
                num_RNN_alpha = 2,
                dropout_p = 0,
                scale_reconstruction = 0.3,
                device='cpu').to('cpu')
    model.build()
    
    kvae = KVAE(model=model,
                lr = 5e-4,
                lr_tot = 7e-4,#3e-3
                epochs = 100,
                batch_size = batch_size,
                early_stop_patience = 200,
                save_frequency = 1,
                only_vae_epochs = 0,
                kf_update_epochs = 0,
                save_dir = "results")

    kvae.train(train_dataloader, val_dataloader, verbose=False)
    
    """
    # uncomment for training with warm restarts
    kvae.train(train_dataloader, val_dataloader, verbose=False,
        state_dict_path="results/conv_test22_d_singular/model_at_49.pt",
        optimizer_vae_path="results/conv_test22_d_singular/latest_vae.pt",
        optimizer_kf_path=None,
        optimizer_all_path=None)
    """
