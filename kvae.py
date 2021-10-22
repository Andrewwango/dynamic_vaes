import torch
import numpy as np
import os
from tqdm import tqdm
import pickle

from auxiliary import plot_loss, export_vid

class KVAE:
    def __init__(self, model, lr, lr_tot, epochs, batch_size, early_stop_patience, 
                save_frequency, only_vae_epochs, kf_update_epochs,
                save_dir):
        self.model = model
        self.lr = lr
        self.lr_tot = lr_tot
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.save_frequency = save_frequency
        self.only_vae_epochs = only_vae_epochs
        self.kf_update_epochs = kf_update_epochs
        self.save_dir = save_dir

    def train(self, train_dataloader, val_dataloader, verbose=True, state_dict_path=None, optimizer_vae_path=None, optimizer_kf_path=None, optimizer_all_path=None):        
        if verbose:
            for a in self.model.named_parameters():
                print(a)
        
        if state_dict_path:
            self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
            print("loaded state.")

        train_loss, train_vae, train_lgssm, val_loss, val_vae, val_lgssm = np.zeros((6, self.epochs))
        train_num = len(train_dataloader.dataset)
        val_num = len(val_dataloader.dataset)

        best_epoch = self.epochs
        best_state_dict = self.model.state_dict()
        best_val_loss = np.inf
        patience = 0

        for epoch in range(self.epochs):
            print("epoch", epoch)

            if epoch == -1:
                print("switching to only vae")
                self.optimizer_only_vae = torch.optim.Adam(self.model.iter_vae, lr=self.lr)
                if optimizer_vae_path:
                    self.optimizer_only_vae.load_state_dict(torch.load(optimizer_vae_path, map_location='cpu'))
                    print("loaded vae optim")
            elif epoch == -1:#self.only_vae_epochs:
                print("switching to vae+kf")
                self.optimizer_vae_kf = torch.optim.Adam(self.model.iter_vae_kf, lr=self.lr_tot)
                if optimizer_kf_path:
                    self.optimizer_vae_kf.load_state_dict(torch.load(optimizer_kf_path, map_location='cpu'))
                    print("loaded kf optim")
            elif epoch == self.only_vae_epochs + self.kf_update_epochs:
                print("Switching to all")
                self.optimizer_all = torch.optim.Adam(self.model.iter_all, lr=self.lr_tot)
                if optimizer_all_path:
                    self.optimizer_all.load_state_dict(torch.load(optimizer_all_path, map_location='cpu'))
                    print("loaded all optim")


            if epoch < self.only_vae_epochs:
                print("only vae...")
                optimizer = self.optimizer_only_vae
            elif epoch < self.only_vae_epochs + self.kf_update_epochs:
                print("vae + kf...")
                optimizer = self.optimizer_vae_kf
            else:
                print("all...")
                optimizer = self.optimizer_all

            train_loss_accum = np.zeros(3)
            print("start batch training")
            for batch_idx, batch_data in tqdm(enumerate(train_dataloader)):
                self.model.train()
                batch_data = batch_data.to('cpu')
                recon_batch_data = self.model(batch_data, compute_loss=True) #forward pass

                loss_tot, loss_vae, loss_lgssm = self.model.loss
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                train_loss_accum += loss_tot.item(), loss_vae.item(), loss_lgssm.item()
                
            val_loss_accum = np.zeros(3)
            for batch_idx, batch_data in tqdm(enumerate(val_dataloader)):
                self.model.eval()

                batch_data = batch_data.to('cpu')
                recon_batch_data = self.model(batch_data, compute_loss=True) #forward pass

                loss_tot, loss_vae, loss_lgssm = self.model.loss

                val_loss_accum += loss_tot.item(), loss_vae.item(), loss_lgssm.item()
            
            train_loss[epoch], train_vae[epoch], train_lgssm[epoch] = train_loss_accum / train_num
            val_loss[epoch],   val_vae[epoch],   val_lgssm[epoch]   = val_loss_accum   / val_num
            
            if val_loss[epoch] < best_val_loss:
                best_val_loss = val_loss[epoch]
                best_state_dict = self.model.state_dict()
                best_epoch = epoch
                patience = 0
            else:
                patience += 1

            print("Training loss", train_loss[epoch], train_vae[epoch], train_lgssm[epoch], "Validation loss", val_loss[epoch], val_vae[epoch],   val_lgssm[epoch])

            if patience == self.early_stop_patience:
                print("Early stop at epoch", epoch)
                break

            if not epoch % self.save_frequency:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'model_at_{epoch}.pt'))
                if epoch < self.only_vae_epochs: # for schedule training (TODO: try without this)
                    optim_name = "vae"
                elif epoch < self.only_vae_epochs + self.kf_update_epochs:
                    optim_name = "vaekf"
                else:
                    optim_name = "all"
                torch.save(optimizer.state_dict(), os.path.join(self.save_dir, f'latest_{optim_name}.pt'))
                export_vid(self, val_dataloader, epoch=epoch)
        
        torch.save(best_state_dict, os.path.join(self.save_dir, f'best_{best_epoch}.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'last_{epoch}.pt'))
        
        train_loss = train_loss[:epoch+1]; train_vae = train_vae[:epoch+1]; train_lgssm = train_lgssm[:epoch+1]
        val_loss = val_loss[:epoch+1]; val_vae = val_vae[:epoch+1]; val_lgssm = val_lgssm[:epoch+1]
        
        with open(os.path.join(self.save_dir, 'loss_model.pckl'), 'wb') as f:
            pickle.dump([train_loss, train_vae, train_lgssm, val_loss, val_vae, val_lgssm], f)

        plot_loss(train_loss, val_loss, "", self.save_dir)
        plot_loss(train_vae, val_vae, "vae", self.save_dir)
        plot_loss(train_lgssm, val_lgssm, "lgssm", self.save_dir)
    
    def eval(self, x, x_dims=None, state_dict=None, dir=None):
        if state_dict:
            self.model.load_state_dict(torch.load(state_dict, map_location='cpu'))
            print("state_dict loaded")
        self.model.eval()
        with torch.no_grad():
            x_hat = self.model(x, compute_loss=False).to('cpu').detach().numpy()
        if x_dims:
            x_hat = x_hat.reshape(x.shape[0], *x_dims, -1)
        return x,x_hat