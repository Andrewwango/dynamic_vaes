import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
import torch
import numpy as np

def load_bouncing_ball(datafolder, dataname, eps=1e-6, static=False, singular=False, binary=True):
    """[summary]

    Args:
        datafolder ([type]): [description]
        dataname ([type]): [description]
        eps (float, optional): offset to prevent div/0 errors. Defaults to 1e-6.
        static (bool, optional): returns one video per sample consisting of one frame repeated. Defaults to False.
        singular (bool, optional): returns one image per sample (to use with classical static convolutional VAE). Defaults to False.
    """
    def load_images(path):
        npzfile = np.load(path)
        pics = (npzfile['images'].astype(np.float32) > 0).astype('float32') if binary else\
            npzfile['images'].astype(np.float32)
        pics += eps
        if static:
            pics = np.repeat(pics[:,10,:,:][:,None,:,:], pics.shape[1], axis=1)
        if singular:
            pics = pics[:, 0, :, :][:, None, :, :]
        return pics
    train_images = load_images(f"{datafolder}/{dataname}.npz") #sequences, timesteps, d1, d2 = images.shape
    test_images  = load_images(f"{datafolder}/{dataname}_test.npz")
    return np.moveaxis(train_images, 1, -1), np.moveaxis(test_images, 1, -1)

def plot_loss(train, val, title, dir):
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    plt.rcParams['font.size'] = 12
    plt.plot(train, label='training loss')
    plt.plot(val, label='validation loss')
    plt.legend(fontsize=16, title="loss_"+title, title_fontsize=20)
    plt.xlabel('epochs', fontdict={'size':16})
    plt.ylabel('loss', fontdict={'size':16})
    plt.savefig(os.path.join(dir, f'loss_{title}.png'))

def save_frames(seq, name="test.mp4"):
    fig = plt.figure()
    frames=[]
    for i in range(seq.shape[-1]):
        frames.append([plt.imshow(np.vstack([seq[b, :, :, i] for b in range(seq.shape[0])]), cmap='gray', animated=True)])
    ani = ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(name)
    plt.close()

def export_vid(model, dataloader, state_dict=None, epoch=0, j=0, both=True, ret=False):
    """[summary]

    Args:
        model ([type]): [description]
        dataloader ([type]): [description]
        state_dict ([type], optional): [description]. Defaults to None.
        epoch (int, optional): [description]. Defaults to 0.
        j (int, optional): [description]. Defaults to 0.
        both (bool, optional): [description]. Defaults to True.
        ret (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    t = torch.Tensor(dataloader.dataset[j:j+64])
    d = (32,32)
    if state_dict:
        x,x_hat = model.eval(t, x_dims=d, state_dict=state_dict)
    else:
        x,x_hat = model.eval(t, x_dims=d)

    print(x.shape, x_hat.shape)
    fig = plt.figure()
    if x.shape[-1] > 1:
        frames=[]
        for i in range(x.shape[-1]):
            x_mat = np.vstack([np.hstack([x[sample,:,:,i], x_hat[sample,:,:,i]]) for sample in (0,1)])
            frames.append([plt.imshow(x_mat if both else x_hat[0,:,:,i], cmap='gray', animated=True)])
        ani = ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
        ani.save(f"results/test_x_{epoch}.mp4")
    else:
        x_mat = np.vstack([np.hstack([x[0,:,:,0], x_hat[0,:,:,0]]), np.hstack([x[1,:,:,0], x_hat[1,:,:,0]])])
        plt.imshow(x_mat, cmap='gray')
        plt.savefig(f"results/test_x_{epoch}.jpg", bbox_inches='tight')
    plt.close()
    if ret: return x_hat


def export_latent_space_vis(mu_smooth, a, titles=["z_0-1", "z_2-3", "a_0-1", "a_2-3"], j=0):
    #mu_smooth (seq_len, batch_size, z_dim), a (seq_len, batch_size, a_dim)
    xs = [mu_smooth[:, j, 0], mu_smooth[:, j, 2], a[:, j, 0], a[:, j, 2]]
    ys = [mu_smooth[:, j, 1], mu_smooth[:, j, 3], a[:, j, 1], a[:, j, 3]]
    fig, axs = plt.subplots(nrows=1, ncols=4)#, figsize=(5, 3))

    lines = []
    for j in range(4):
        axs[j].set(xlim=(xs[j].min(), xs[j].max()), ylim=(ys[j].min(), ys[j].max()), title=titles[j])
        lines.append(axs[j].plot(xs[j][0], ys[j][0], color='k', lw=2)[0])

    def animate(i):
        for j in range(4):
            lines[j].set_data(xs[j][:i], ys[j][:i])

    anim = FuncAnimation(fig, animate, interval=50, frames=mu_smooth.shape[0]-1)
    anim.save('results/z_circle.mp4')