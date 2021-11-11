import numpy as np
from tqdm import tqdm
from PIL import Image
import imageio
import os

class GrayScottVideoGenerator:
    def __init__(self, frame_size=300, F=0.0545, k=0.062, Du=0.1, Dv=0.05):
        self.frame_size = frame_size
        self.F = F
        self.k = k
        self.Du = Du
        self.Dv = Dv
        
    def initialise(self, mask_xlim=[0.4, 0.6], mask_ylim=[0.4, 0.6]):
        n = self.frame_size
        u = np.ones((n+2,n+2))
        v = np.zeros((n+2,n+2))

        x, y = np.meshgrid(np.linspace(0, 1, n+2), np.linspace(0, 1, n+2))

        mask = (mask_xlim[0]<x) & (x<mask_xlim[1]) & (mask_ylim[0]<y) & (y<mask_ylim[1])

        u[mask] = np.random.rand(*(u[mask].shape))#0.50
        v[mask] = np.random.rand(*(v[mask].shape))#0.25

        return u, v
    
    def set_boundary_conditions(self, x):
        x[0 , :] = x[-2, :]
        x[-1, :] = x[1 , :]
        x[: , 0] = x[: ,-2]
        x[: ,-1] = x[: , 1]
        return x

    def Laplacian(self, x):
        """
        second order finite differences
        """
        return (                  x[ :-2, 1:-1] +
                 x[1:-1, :-2] - 4*x[1:-1, 1:-1] + x[1:-1, 2:] +
                              +   x[2:  , 1:-1] )
    
    def step(self, u, v):
        Lu = self.Laplacian(u)
        Lv = self.Laplacian(v)
        
        U, V = u[1:-1,1:-1], v[1:-1,1:-1]

        UVV = U*V*V
        U += self.Du*Lu - UVV + self.F*(1 - U)
        V += self.Dv*Lv + UVV - (self.F + self.k)*V

        u = self.set_boundary_conditions(u)
        v = self.set_boundary_conditions(v)
        
        return u,v
    
    def run(self, u, v, seq_len=500, save_freq=40):        
        frames = np.zeros((seq_len, *(u.shape)), dtype=np.float32)
        for i in range(save_freq*seq_len):
            u,v = self.step(u,v)
            if not i % save_freq:
                frame = np.float32(1.0*(v-v.min()) / (v.max()-v.min()))
                frames[int(i/save_freq)] = frame
                
        return frames
    

class BatchGrayScottVideoGenerator():

    def run(self, res=60, n_samples=50, seq_len=200, format=None, fp=""):
        images = np.empty((seq_len, n_samples, *res), dtype=np.float32)

        for i in tqdm(range(n_samples)):
            g = GrayScottVideoGenerator(frame_size=res[0])
            u,v = g.initialise()#mask_xlim=[0,1], mask_ylim=[0,1])
            frames = g.run(u, v, seq_len=seq_len, save_freq=40)
            images[:, i, :, :] = frames #TODO: remove implicit conversion
        
            if format=='gif':
                imageio.mimsave(f'grayscott_{i}.gif', frames, format='gif', fps=60)
        
        if format=='npz':
            np.savez(os.path.abspath(fp), images=images)

if __name__ == "__main__":
    print("Generating training sequences")
    TrainGenerator = BatchGrayScottVideoGenerator()
    TrainGenerator.run(res=60, iterations=10, seq_len=100, format='gif', fp="grayscott_data/grayscott")

    print("Generating test sequences")
    TestGenerator = BatchGrayScottVideoGenerator()
    TestGenerator.run(res=60, iterations=0, seq_len=100, format='gif', fp="grayscott_data/grayscott_test")