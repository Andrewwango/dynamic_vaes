import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from collections import OrderedDict

def debug(n, v):
    print(n, v.shape, "Nan:", torch.isnan(v).any(),"inf:", torch.isinf(v).any())

def check_bad(*v):
    return True in [torch.isnan(i).any() or torch.isinf(i).any() for i in v]

def concat_iter(*iter_list):
    for i in iter_list:
        yield from i

class KVAEModel(nn.Module):

    def __init__(self, x_dim, a_dim = 8, z_dim=4,
                 x_2d=False,
                 init_kf_mat=0.05, noise_transition=0.08, noise_emission=0.03, init_cov=20,
                 K=3, dim_RNN_alpha=50, num_RNN_alpha=1,
                 dropout_p=0, scale_reconstruction=1, device='cpu'):

        super().__init__()
        ## General parameters
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.u_dim = a_dim
        self.dropout_p = dropout_p
        self.scale_reconstruction = scale_reconstruction
        self.device = device
        # VAE
        self.x_2d = x_2d
        # LGSSM
        self.init_kf_mat = init_kf_mat
        self.noise_transition = noise_transition
        self.noise_emission = noise_emission
        self.init_cov = init_cov
        # Dynamics params (alpha)
        self.K = K
        self.dim_RNN_alpha = dim_RNN_alpha
        self.num_RNN_alpha = num_RNN_alpha

    def build(self):

        #############
        #### VAE ####
        #############
        mlp_n = 25
        conv_n = 32*3*3
        # Encoder
        mlp_x_a_dict = OrderedDict([
            ("linear0",     nn.Linear(self.x_dim, mlp_n)),
            ('activation0', nn.Tanh()),
            ('dropout0',    nn.Dropout(p=self.dropout_p)),
            ("linear1",     nn.Linear(mlp_n, mlp_n)),
            ("activation1", nn.Tanh()),
            ("dropout1",    nn.Dropout(p=self.dropout_p))
        ])
        #(seq_len, batch_size, x_dim)
        conv_x_a_dict = OrderedDict([
            ("unflatten", nn.Unflatten(-1, (1, 32, 32))),
            ("conv0", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)),
            ("activation0", nn.ReLU()),
            ("conv1", nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)),
            ("activation1", nn.ReLU()),
            ("conv2", nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)),
            ("activation2", nn.ReLU()),
            ("flatten", nn.Flatten(start_dim=1))
        ])
        self.mlp_x_a = nn.Sequential(conv_x_a_dict)

        self.inf_mean = nn.Linear(conv_n, self.a_dim)
        self.inf_logvar = nn.Sequential(OrderedDict([
            ("linear",     nn.Linear(conv_n, self.a_dim)),
            ("activation", nn.Sigmoid())
        ]))

        # 2. Decoder
        mlp_a_x_dict = OrderedDict([
            ("linear0",     nn.Linear(self.a_dim, mlp_n)),
            ('activation0', nn.Tanh()),
            ('dropout0',    nn.Dropout(p=self.dropout_p)),
            ("linear1",     nn.Linear(mlp_n, mlp_n)),
            ("activation1", nn.Tanh()),
            ("dropout1",    nn.Dropout(p=self.dropout_p))
        ])
        conv_a_x_dict = OrderedDict([
            ("linear", nn.Linear(self.a_dim, 32*3*3)),
            ("unflatten", nn.Unflatten(-1, (32, 3, 3))),
            ("conv0", nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)),
            ("activation0", nn.ReLU()),
            ("conv1", nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)),
            ("activation1", nn.ReLU()),
            ("conv2", nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, output_padding=1)),
            ("activation2", nn.ReLU())
        ])
        self.mlp_a_x = nn.Sequential(conv_a_x_dict)

        gen_mlp_dict = OrderedDict([
            ("linear", nn.Linear(mlp_n, self.x_dim)),
            ("activation", nn.Sigmoid())
        ])
        #x_mu = slim.conv2d(dec_hidden, 1, 1, stride=1, activation_fn=activation_x_mu)
        gen_conv_dict = OrderedDict([
            ("conv", nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)),
            ("flatten", nn.Flatten(start_dim=1)),
            ("activation", nn.Sigmoid())            
        ])
        #
        #TODO: add extra linear layer after/instead of final conv?
        #
        self.gen_logvar = nn.Sequential(gen_conv_dict)

        ###############
        #### LGSSM ####
        ###############
        # Initializers for LGSSM variables, torch.tensor(), enforce torch.float32 type
        # A is an identity matrix
        # B and C are randomly sampled from a Gaussian
        # Q and R are isotroipic covariance matrices
        # z = Az + Bu
        # a = Cz
        self.A = torch.tensor(np.array([np.eye(self.z_dim) for _ in range(self.K)]), dtype=torch.float32, requires_grad=True, device=self.device) # (K, z_dim. z_dim,)
        self.B = torch.tensor(np.array([self.init_kf_mat * np.random.randn(self.z_dim, self.u_dim) for _ in range(self.K)]), dtype=torch.float32, requires_grad=True, device=self.device) # (K, z_dim, u_dim)
        self.C = torch.tensor(np.array([self.init_kf_mat * np.random.randn(self.a_dim, self.z_dim) for _ in range(self.K)]), dtype=torch.float32, requires_grad=True, device=self.device) # (K, a_dim, z_dim)
        self.Q = self.noise_transition * torch.eye(self.z_dim).to(self.device) # (z_dim, z_dim)
        self.R = self.noise_emission * torch.eye(self.a_dim).to(self.device) # (a_dim, a_dim)
        self._I = torch.eye(self.z_dim).to(self.device) # (z_dim, z_dim)

        ###############
        #### Alpha ####
        ###############
        self.a_init = torch.zeros((1, self.a_dim), requires_grad=True, device=self.device) # (bs, a_dim)
        self.rnn_alpha = nn.LSTM(self.a_dim, self.dim_RNN_alpha, self.num_RNN_alpha, bidirectional=False)
        self.mlp_alpha = nn.Sequential(nn.Linear(self.dim_RNN_alpha, self.K),
                                       nn.Softmax(dim=-1))

        
        ############################
        #### Scheduler Training ####
        ############################
        self.A = nn.Parameter(self.A)
        self.B = nn.Parameter(self.B)
        self.C = nn.Parameter(self.C)
        self.a_init = nn.Parameter(self.a_init)
        kf_params = [self.A, self.B, self.C, self.a_init]
        
        self.iter_kf = (i for i in kf_params)
        self.iter_vae = concat_iter(self.mlp_x_a.parameters(),
                                         self.inf_mean.parameters(),
                                         self.inf_logvar.parameters(),
                                         self.mlp_a_x.parameters(),
                                         self.gen_logvar.parameters())
        self.iter_alpha = concat_iter(self.rnn_alpha.parameters(),
                                           self.mlp_alpha.parameters())
        self.iter_kf_alpha = concat_iter(self.iter_kf, self.iter_alpha)
        self.iter_vae_kf = concat_iter(self.iter_vae, self.iter_kf)
        self.iter_all = concat_iter(self.iter_kf, self.iter_vae, self.iter_alpha)
                

    def encode(self, x):
        seq_len, batch_size, x_len = x.shape
        x_mlp = self.mlp_x_a(x.reshape(seq_len*batch_size, x_len))
        x_a = x_mlp.view(seq_len, batch_size, -1)
        a_mean = self.inf_mean(x_a)
        a_logvar = self.inf_logvar(x_a)
        # Reparameterisation Trick
        a = torch.randn_like(a_logvar).mul(torch.exp(0.5*a_logvar)).add_(a_mean)
        return a, a_mean, a_logvar

    def decode(self, a):
        seq_len, batch_size, a_len = a.shape
        a_x = self.mlp_a_x(a.reshape(seq_len*batch_size, a_len))
        log_y = self.gen_logvar(a_x).view(seq_len, batch_size, -1)
        #y = torch.exp(log_y)
        return log_y

    def a_to_alpha(self, a): #a=None
        batch_size = a.shape[1] if a is not None else 64
        # Calculate alpha, initial observation a_init is assumed to be zero and can be learned
        a_init_expand = self.a_init.unsqueeze(1).repeat(1, batch_size, 1) # (1, bs, a_dim)
        if a is not None:
            a_tm1 = torch.cat([a_init_expand, a[:-1,:,:]], 0) # (seq_len, bs, a_dim)
        else:
            a_tm1 = torch.cat([a_init_expand], 0)
        with torch.no_grad():
            alpha = self.get_alpha(a_tm1) # (seq_len, bs, K)
        return alpha

    def kf_smoother(self, a, u, K, A, B, C, R, Q, optimal_gain=False, alpha_sq=1):
        """"
        Kalman Smoother, refer to Murphy's book (MLAPP), section 18.3
        Difference from KVAE source code: 
            - no imputation
            - only RNN for the calculation of alpha
            - different notations (rather than using same notations as Murphy's book ,we use notation from model KVAE)
            >>>> z_t = A_t * z_tm1 + B_t * u_t
            >>>> a_t = C_t * z_t
        Input:
            - a, (seq_len, bs, a_dim)
            - u, (seq_len, bs, u_dim)
            - alpha, (seq_len, bs, alpha_dim)
            - K, real number
            - A, (K, z_dim, z_dim)
            - B, (K, z_dim, u_dim)
            - C, (K, a_dim, z_dim)
            - R, (z_dim, z_dim)
            - Q , (a_dim, a_dim)
        """
        # Initialization
        seq_len = a.shape[0]
        batch_size = a.shape[1]
        self.mu = torch.zeros((batch_size, self.z_dim)).to(self.device) # (bs, z_dim), z_0
        self.Sigma = self.init_cov * torch.eye(self.z_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # (bs, z_dim, z_dim), Sigma_0
        mu_pred = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device) # (seq_len, bs, z_dim)
        mu_filter = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device) # (seq_len, bs, z_dim)
        mu_smooth = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device) # (seq_len, bs, z_dim)
        Sigma_pred = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device) # (seq_len, bs, z_dim, z_dim)
        Sigma_filter = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device) # (seq_len, bs, z_dim, z_dim)
        Sigma_smooth = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device) # (seq_len, bs, z_dim, z_dim)
        
        # Calculate alpha, initial observation a_init is assumed to be zero and can be learned
        a_init_expand = self.a_init.unsqueeze(1).repeat(1, batch_size, 1) # (1, bs, a_dim)
        a_tm1 = torch.cat([a_init_expand, a[:-1,:,:]], 0) # (seq_len, bs, a_dim)
        alpha = self.get_alpha(a_tm1) # (seq_len, bs, K)

        # Calculate the mixture of A, B and C
        A_flatten = A.view(K, self.z_dim*self.z_dim) # (K, z_dim*z_dim) 
        B_flatten = B.view(K, self.z_dim*self.u_dim) # (K, z_dim*u_dim) 
        C_flatten = C.view(K, self.a_dim*self.z_dim) # (K, a_dim*z_dim) 
        A_mix = alpha.matmul(A_flatten).view(seq_len, batch_size, self.z_dim, self.z_dim)
        B_mix = alpha.matmul(B_flatten).view(seq_len, batch_size, self.z_dim, self.u_dim)
        C_mix = alpha.matmul(C_flatten).view(seq_len, batch_size, self.a_dim, self.z_dim)

        # Forward filter
        for t in range(seq_len):
            
            # Mixture of A, B and C
            A_t = A_mix[t] # (bs, z_dim. z_dim)
            B_t = B_mix[t] # (bs, z_dim, u_dim)
            C_t = C_mix[t] # (bs, a_dim, z_dim)

            if t == 0:
                mu_t_pred = self.mu.unsqueeze(-1) # (bs, z_dim, 1)
                Sigma_t_pred = self.Sigma
            else:
                u_t = u[t,:,:] # (bs, u_dim)
                mu_t_pred = A_t.bmm(mu_t) + B_t.bmm(u_t.unsqueeze(-1)) # (bs, z_dim, 1), z_{t|t-1}
                Sigma_t_pred = alpha_sq * A_t.bmm(Sigma_t).bmm(A_t.transpose(1,2)) + self.Q # (bs, z_dim, z_dim), Sigma_{t|t-1}
                # alpha_sq (>=1) is fading memory control, which indicates how much you want to forgert past measurements, see more infos in 'FilterPy' library
            
            # Residual
            a_pred = C_t.bmm(mu_t_pred)  # (bs, a_dim, z_dim) x (bs, z_dim, 1)
            res_t = a[t, :, :].unsqueeze(-1) - a_pred # (bs, a_dim, 1)

            # Kalman gain
            S_t = C_t.bmm(Sigma_t_pred).bmm(C_t.transpose(1,2)) + self.R # (bs, a_dim, a_dim)
            S_t_inv = S_t.inverse()
            K_t = Sigma_t_pred.bmm(C_t.transpose(1,2)).bmm(S_t_inv) # (bs, z_dim, a_dim)

            # Update 
            mu_t = mu_t_pred + K_t.bmm(res_t) # (bs, z_dim, 1)
            I_KC = self._I - K_t.bmm(C_t) # (bs, z_dim, z_dim)
            if optimal_gain:
                Sigma_t = I_KC.bmm(Sigma_t_pred) # (bs, z_dim, z_dim), only valid with optimal Kalman gain
            else:
                Sigma_t = I_KC.bmm(Sigma_t_pred).bmm(I_KC.transpose(1,2)) + K_t.matmul(self.R).matmul(K_t.transpose(1,2)) # (bs, z_dim, z_dim), general case

            # Save cache
            mu_pred[t] = mu_t_pred.view(batch_size, self.z_dim)
            mu_filter[t] = mu_t.squeeze()
            Sigma_pred[t] = Sigma_t_pred
            Sigma_filter[t] = Sigma_t
  
        # Add the final state from filter to the smoother as initialization
        mu_smooth[-1] =  mu_filter[-1]
        Sigma_smooth[-1] = Sigma_filter[-1]

        # Backward smooth, reverse loop from pernultimate state
        for t in range(seq_len-2, -1, -1):
            
            # Backward Kalman gain
            J_t = Sigma_filter[t].bmm(A_mix[t+1].transpose(1,2)).bmm(Sigma_pred[t+1].inverse()) # (bs, z_dim, z_dim)

            # Backward smoothing
            dif_mu_tp1 = (mu_smooth[t+1] - mu_filter[t+1]).unsqueeze(-1) # (bs, z_dim, 1)
            mu_smooth[t] = mu_filter[t] + J_t.matmul(dif_mu_tp1).view(batch_size, self.z_dim) # (bs, z_dim)
            dif_Sigma_tp1 = Sigma_smooth[t+1] - Sigma_pred[t+1] # (bs, z_dim, z_dim)
            Sigma_smooth[t] = Sigma_filter[t] + J_t.bmm(dif_Sigma_tp1).bmm(J_t.transpose(1,2)) # (bs, z_dim, z_dim)

        # Generate a from smoothing z
        a_gen = C_mix.matmul(mu_smooth.unsqueeze(-1)).view(seq_len, batch_size, self.a_dim) # (seq_len, bs, a_dim)
        
        return a_gen, mu_smooth, Sigma_smooth, A_mix, B_mix, C_mix

    def get_alpha(self, a_tm1):        
        alpha, _ = self.rnn_alpha(a_tm1) # (seq_len, bs, dim_alpha)
        alpha = self.mlp_alpha(alpha) # (seq_len, bs, K), softmax on K dimension
        return alpha

    def forward(self, x, compute_loss=False):
        if self.x_2d: x = torch.flatten(x, -3, -2)

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(-1, 0, 1)
    
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part
        a, a_mean, a_logvar = self.encode(x)
        batch_size = a.shape[1]
        u_0 = torch.zeros(1, batch_size, self.u_dim).to(self.device)
        u = torch.cat((u_0, a[:-1]), 0)
        a_gen, mu_smooth, Sigma_smooth, A_mix, B_mix, C_mix = self.kf_smoother(a, u, self.K, self.A, self.B, self.C, self.R, self.Q)

        y = self.decode(a_gen)
        #print(x.shape, y.shape)
        # calculate loss
        if compute_loss:

            loss_tot, loss_vae, loss_lgssm = self.get_loss(x, y, u, 
                                                        a, a_mean, a_logvar, 
                                                        mu_smooth, Sigma_smooth, 
                                                        A_mix, B_mix, C_mix,
                                                        self.scale_reconstruction,
                                                        seq_len, batch_size)
            self.loss = (loss_tot, loss_vae, loss_lgssm)
        
        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        
        self.y = y.permute(1,-1,0)#.squeeze()

        return self.y

    def forward_debug(self, x, compute_loss=False):
        if self.x_2d: x = torch.flatten(x, -3, -2)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(-1, 0, 1)
    
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        print("1")
        # main part
        a, a_mean, a_logvar = self.encode(x)
        print("2")
        batch_size = a.shape[1]
        u_0 = torch.zeros(1, batch_size, self.u_dim).to(self.device)
        u = torch.cat((u_0, a[:-1]), 0)
        a_gen, mu_smooth, Sigma_smooth, A_mix, B_mix, C_mix = self.kf_smoother(a, u, self.K, self.A, self.B, self.C, self.R, self.Q)
        print("3")
        y = self.decode(a_gen)
        print("4")
        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        
        self.y = y.permute(1,-1,0)#.squeeze()

        return x.to('cpu').detach().numpy(), self.y.to('cpu').detach().numpy(), a.to('cpu').detach().numpy(), a_gen.to('cpu').detach().numpy(), mu_smooth.to('cpu').detach().numpy()


    def get_loss(self, x, y, u, a, a_mean, a_logvar, mu_smooth, Sigma_smooth,
             A, B, C, scale_reconstruction=1, seq_len=150, batch_size=32):
               
        #DVAE (IS)
        log_px_given_a = - torch.sum( x/(y+1e-15) - torch.log(x/(y+1e-15)) - 1) / (batch_size * seq_len)
        #tutorial + KVAE
        #log_px_given_a = - nn.BCELoss(reduction='sum')(y.permute(1,-1,0), x.permute(1,-1,0))/(batch_size * seq_len)

        # log q_{\phi}(a_hat | x), Gaussian
        #KVAE + DVAE (orig) (G)
        log_qa_given_x = - 0.5 * a_logvar - torch.pow(a - a_mean, 2) / (2 * torch.exp(a_logvar))
        log_qa_given_x = torch.sum(log_qa_given_x) /  (batch_size * seq_len)
        # Tutorial (K>0)
        # log_qa_given_x = -0.5 * torch.sum(1 + a_logvar - a_mean.pow(2) - a_logvar.exp()) / (batch_size * seq_len)
        # DVAE (get_vae_loss)
        #log_qa_given_x = -0.5 * torch.sum(a_logvar - a_mean.pow(2) - a_logvar.exp()) / (batch_size * seq_len)


        #loss_vae = self.new_vae_loss(x, y, a_mean, a_logvar, batch_size, seq_len, beta=1)
        loss_vae = - scale_reconstruction * log_px_given_a + log_qa_given_x

        # log p_{\gamma}(a_tilde, z_tilde | u) < in sub-comment, 'tilde' is hidden for simplification >
        # >>> log p(z_t | z_tm1, u_t), transition
        mvn_smooth = MultivariateNormal(mu_smooth, Sigma_smooth)
        z_smooth = mvn_smooth.sample() # # (seq_len, bs, z_dim)
        Az_tm1 = A[:-1].matmul(z_smooth[:-1].unsqueeze(-1)).view(seq_len-1, batch_size, -1) # (seq_len, bs, z_dim)
        Bu_t = B[:-1].matmul(u[:-1].unsqueeze(-1)).view(seq_len-1, batch_size, -1) # (seq_len, bs, z_dim)
        mu_t_transition = Az_tm1 +Bu_t
        z_t_transition = z_smooth[1:]
        mvn_transition = MultivariateNormal(z_t_transition, self.Q)
        log_prob_transition = mvn_transition.log_prob(mu_t_transition)
        # >>> log p(z_0 | z_init), init state
        z_0 = z_smooth[0]
        mvn_0 = MultivariateNormal(self.mu, self.Sigma)
        log_prob_0 = mvn_0.log_prob(z_0)
        # >>> log p(a_t | z_t), emission
        Cz_t = C.matmul(z_smooth.unsqueeze(-1)).view(seq_len, batch_size, self.a_dim)
        mvn_emission = MultivariateNormal(Cz_t, self.R)
        log_prob_emission = mvn_emission.log_prob(a)
        # >>> log p_{\gamma}(a_tilde, z_tilde | u)
        log_paz_given_u = torch.cat([log_prob_transition, log_prob_0.unsqueeze(0)], 0) + log_prob_emission

        # log p_{\gamma}(z_tilde | a_tilde, u)
        # >>> log p(z_t | a, u)
        log_pz_given_au = mvn_smooth.log_prob(z_smooth)

        # Normalization
        #log_px_given_a = torch.sum(log_px_given_a) /  (batch_size * seq_len)
        #log_qa_given_x = torch.sum(log_qa_given_x) /  (batch_size * seq_len)
        log_paz_given_u = torch.sum(log_paz_given_u) /  (batch_size * seq_len)
        log_pz_given_au = torch.sum(log_pz_given_au) /  (batch_size * seq_len)

        # Loss
        loss_lgssm =  - log_paz_given_u + log_pz_given_au
        loss_tot = loss_vae + loss_lgssm

        if check_bad(loss_vae, loss_lgssm, loss_tot):
            debug("loss_vae", loss_vae)
            #debug("log_px_given_a", log_px_given_a)
            print("Args bad", check_bad(x, y, u, a, a_mean, a_logvar, mu_smooth, Sigma_smooth, A, B, C))
            """
            debug("x", x)
            debug("y", y)
            debug("u", u)
            debug("a", a)
            debug("a_mean", a_mean)
            debug("a_logvar", a_logvar)
            debug("mu_smooth", mu_smooth)
            debug("Sigma_smooth", Sigma_smooth)
            debug("A", A)
            debug("B", B)
            debug("C", C)
            """

        return loss_tot, loss_vae, loss_lgssm

    def generate(self, z_temp, N=200):
        #z_temp list seq_len, batch_size, z_dim
        z_temp = torch.tensor(z_temp).detach()
        seq_len, batch_size, _ = z_temp.shape
        z_temp = z_temp.permute(1,-1,0)

        z_samples = torch.zeros((batch_size, self.z_dim, seq_len+N))
        y_samples = torch.zeros((batch_size, self.x_dim, seq_len+N))
        a_samples = torch.zeros((batch_size, self.a_dim, seq_len+N))

        z = z_temp[:, :, -2:-1] #batch_size, z_dim, 1
        K = 1
        A_flatten = self.A.view(K, self.z_dim*self.z_dim) # (K, z_dim*z_dim) 
        B_flatten = self.B.view(K, self.z_dim*self.u_dim) # (K, z_dim*u_dim) 
        C_flatten = self.C.view(K, self.a_dim*self.z_dim) # (K, a_dim*z_dim) 

        A_mix = A_flatten.view(1, self.z_dim, self.z_dim)
        B_mix = B_flatten.view(1, self.z_dim, self.u_dim)
        C_mix = C_flatten.view(1, self.a_dim, self.z_dim)

        A_mix = A_mix.repeat((batch_size, 1, 1))
        B_mix = B_mix.repeat((batch_size, 1, 1))
        C_mix = C_mix.repeat((batch_size, 1, 1))

        def z_to_y(model, z, C): #z in (batch_size, z_dim, seq_len)
            model.eval()
            _, _, seq_len = z.shape
            #print(C.shape, z.shape)

            a_gen = C.bmm(z)
            a_gen = a_gen.permute(-1, 0, 1)
            #print(a_gen.shape)
            with torch.no_grad():
                y = model.decode(a_gen).permute(1,-1,0) #(batch_size, dim, seq_len)
            return a_gen.permute(1,-1,0), y

        z_samples[:, :, :seq_len] = z_temp
        a_gen, y = z_to_y(self, z_temp, C_mix)
        a_samples[:, :, :seq_len] = a_gen
        y_samples[:, :, :seq_len] = y

        for i in range(seq_len, seq_len+N):
            a_gen, y = z_to_y(self, z, C_mix)
            z = A_mix.bmm(z)
            
            z_samples[:, :, i:i+1] = z
            a_samples[:, :, i:i+1] = a_gen
            y_samples[:, :, i:i+1] = y

        y_samples = y_samples.view(batch_size, 32, 32, seq_len+N)

        return z_samples, y_samples