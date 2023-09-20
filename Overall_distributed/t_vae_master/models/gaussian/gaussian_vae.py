import logging
import math
import time
import numpy as np

import torch
from torch import nn

from t_vae_master.models.utils import from_numpy, to_numpy, make_data_loader
from t_vae_master.models.utils import gaussian_nll, standard_gaussian_nll, gaussian_kl_divergence, reparameterize

from torchquad import MonteCarlo, VEGAS, set_up_backend
import scipy.stats as st

logger = logging.getLogger(__name__)

class GaussianNetwork(nn.Module):
    def __init__(self, n_in, n_latent, n_h, tau=0.5):
        super(GaussianNetwork, self).__init__()

        self.n_in = n_in
        self.n_latent = n_latent
        self.n_h = n_h

        # Encoder
        self.le1 = nn.Sequential(
            nn.Linear(n_in, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
        )
        self.le2_mu = nn.Linear(n_h, n_latent)
        self.le2_ln_var = nn.Linear(n_h, n_latent)
        self.bn_layer_mu = BN_Layer(n_latent, tau, mu=True)
        self.bn_layer_var = BN_Layer(n_latent, tau, mu=False)

        # Decoder
        self.ld1 = nn.Sequential(
            nn.Linear(n_latent, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
        )
        self.ld2_mu = nn.Linear(n_h, n_in)
        self.ld2_ln_var = nn.Linear(n_h, n_in)

    def encode(self, x):
        h = self.le1(x)
        
        mu = self.le2_mu(h)
        mu = self.bn_layer_mu(mu)
        
        ln_var = self.le2_ln_var(h)
        ln_var = self.bn_layer_var(ln_var)
        
        return mu, ln_var

    def decode(self, z):
        h = self.ld1(z)
        return self.ld2_mu(h), self.ld2_ln_var(h)
    
    def forward(self, x, k=1):
        # Compute Negative ELBO
        mu_enc, ln_var_enc = self.encode(x)

        RE = 0
        for i in range(k):
            z = reparameterize(mu=mu_enc, ln_var=ln_var_enc)
            mu_dec, ln_var_dec = self.decode(z)
            RE += gaussian_nll(x, mu=mu_dec, ln_var=ln_var_dec) / k

        KL = gaussian_kl_divergence(mu=mu_enc, ln_var=ln_var_enc)
        return RE, KL

    def evidence_lower_bound(self, x, k=1):
        RE, KL = self.forward(x, k=k)
        return -1 * (RE + KL)

    def importance_sampling(self, x, k=1):
        mu_enc, ln_var_enc = self.encode(x)
        # print("var_enc : ", torch.exp(ln_var_enc*0.5).tolist())
        lls = []
        for i in range(k):
            z = reparameterize(mu=mu_enc, ln_var=ln_var_enc)
            mu_dec, ln_var_dec = self.decode(z)
            ll = -1 * gaussian_nll(x, mu=mu_dec, ln_var=ln_var_dec, dim=1)
            ll -= standard_gaussian_nll(z, dim=1)
            ll += gaussian_nll(z, mu=mu_enc, ln_var=ln_var_enc, dim=1)
            lls.append(ll[:, None])

        return torch.cat(lls, dim=1).logsumexp(dim=1) - math.log(k)
    
class BN_Layer(nn.Module):
    def __init__(self, dim_z, tau=0.5, mu=True):
        super(BN_Layer, self).__init__()
        self.dim_z = dim_z
        self.tau = torch.tensor(tau)
        self.theta = torch.tensor(0.5, requires_grad=True)
        
        self.gamma1 = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.theta))
        self.gamma2 = torch.sqrt((1 - self.tau) * torch.sigmoid(-self.theta))
        
        self.bn = nn.BatchNorm1d(dim_z)
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = True
        
        if mu:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma1)
        else:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma2)
                
    def forward(self, x):
        x = self.bn(x)
        return x

class GaussianVAE:
    def __init__(self, n_in, n_latent, n_h, tau=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = GaussianNetwork(n_in, n_latent, n_h, tau).to(self.device)

        self.train_losses = []
        self.train_times = []
        self.reconstruction_errors = []
        self.kl_divergences = []
        self.valid_losses = []
        self.min_valid_loss = float("inf")

    def _loss_function(self, x, k=1, beta=1):
        RE, KL = self.network(x, k=k)
        RE_sum = RE.sum()
        KL_sum = KL.sum()
        loss = RE_sum + beta * KL_sum
        return loss, RE_sum, KL_sum

    def fit(self, X, k=1, batch_size=100, learning_rate=0.001, n_epoch=500,
            warm_up=False, warm_up_epoch=20,
            is_stoppable=False, X_valid=None, path=None):

        self.network.train()
        N = X.shape[0]
        data_loader = make_data_loader(X, device=self.device, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        if is_stoppable:
            X_valid = from_numpy(X_valid, self.device)

        for epoch in range(n_epoch):
            start = time.time()

            # warm-up
            beta = 1 * epoch / warm_up_epoch if warm_up and epoch <= warm_up_epoch else 1

            mean_loss = 0
            mean_RE = 0
            mean_KL = 0
            for _, batch in enumerate(data_loader):
                optimizer.zero_grad()
                loss, RE, KL = self._loss_function(batch[0], k=k, beta=beta)
                loss.backward()
                mean_loss += loss.item() / N
                mean_RE += RE.item() / N
                mean_KL += KL.item() / N
                optimizer.step()

            end = time.time()
            self.train_losses.append(mean_loss)
            self.train_times.append(end - start)
            self.reconstruction_errors.append(mean_RE)
            self.kl_divergences.append(mean_KL)

            # print(f"epoch: {epoch} / Train: {mean_loss:0.3f} / RE: {mean_RE:0.3f} / KL: {mean_KL:0.3f}", end='')

            if warm_up and epoch < warm_up_epoch:
                # print(" / Warm-up", end='')
                logger.info(f"epoch: {epoch} / Train: {mean_loss:0.3f} / RE: {mean_RE:0.3f} / KL: {mean_KL:0.3f} / Warm-up")
            elif is_stoppable:
                valid_loss, _, _ = self._loss_function(X_valid, k=k, beta=1)
                valid_loss = valid_loss.item() / X_valid.shape[0]
                # print(f" / Valid: {valid_loss:0.3f}", end='')
                self.valid_losses.append(valid_loss)
                is_early_stopping = self._early_stopping(valid_loss, path)
                if is_early_stopping:
                    logger.info(f"epoch: {epoch} / Train: {mean_loss:0.3f} / RE: {mean_RE:0.3f} / KL: {mean_KL:0.3f} / Valid: {valid_loss:0.3f} / Save")
                else:
                    logger.info(f"epoch: {epoch} / Train: {mean_loss:0.3f} / RE: {mean_RE:0.3f} / KL: {mean_KL:0.3f} / Valid: {valid_loss:0.3f}")

            # print('')

        if is_stoppable:
            self.network.load_state_dict(torch.load(path))

        self.network.eval()

    def _early_stopping(self, valid_loss, path):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            torch.save(self.network.state_dict(), path)
            return True
            # print(" ", end='')
        return False

    def encode(self, X):
        mu, ln_var = self.network.encode(from_numpy(X, self.device))
        return to_numpy(mu, self.device), to_numpy(ln_var, self.device)

    def decode(self, Z):
        mu, ln_var = self.network.decode(from_numpy(Z, self.device))
        return to_numpy(mu, self.device), to_numpy(ln_var, self.device)

    def reconstruct(self, X):
        mu_enc, ln_var_enc = self.network.encode(from_numpy(X, self.device))
        z = reparameterize(mu=mu_enc, ln_var=ln_var_enc)
        mu_dec, ln_var_dec = self.network.decode(z)
        return to_numpy(mu_dec, self.device), to_numpy(ln_var_dec, self.device)

    def evidence_lower_bound(self, X, k=1):
        return to_numpy(self.network.evidence_lower_bound(from_numpy(X, self.device), k=k), self.device)

    def importance_sampling(self, X, k=1):
        return to_numpy(self.network.importance_sampling(from_numpy(X, self.device), k=k), self.device)

    def gaussian_prob(self, left_bounds, right_bounds):
        dim = len(left_bounds)
        
        integration_domain = np.row_stack((left_bounds, right_bounds)).T
        integration_domain = from_numpy(integration_domain, self.device)
        # print(integration_domain)

        # 利用importance_sampling近似目标分布的pdf
        # 通过torchquad库求积分
        set_up_backend("torch", data_type="float32")
        def multivariate_normal(x):
            with torch.no_grad():
                prob_list = self.network.importance_sampling(x, k=1)
                prob_list = torch.exp(prob_list)
                return prob_list
        vegas = VEGAS()
        mc = MonteCarlo()
        prob = mc.integrate(
            multivariate_normal,
            dim=dim,
            N=10000,
            integration_domain=integration_domain,
            backend="torch",
            )   
        return prob
    
    def sqr_gaussian_prob(self, left_bounds, right_bounds):
        dim = len(left_bounds)
        
        integration_domain = np.row_stack((left_bounds, right_bounds)).T
        integration_domain = from_numpy(integration_domain, self.device)
        # 利用importance_sampling近似目标分布的pdf
        # 通过torchquad库求积分
        set_up_backend("torch", data_type="float32")
        def multivariate_normal(x):
            with torch.no_grad():
                prob_list = self.network.importance_sampling(x, k=1)
                prob_list = torch.exp(prob_list*2)
                return prob_list
        vegas = VEGAS()
        mc = MonteCarlo()
        prob = mc.integrate(
            multivariate_normal,
            dim=dim,
            N=10000,
            integration_domain=integration_domain,
            backend="torch",
            )   
        return prob
    
    def get_num_parameters(self):
        """
        Returns the number of trainable parameters in a model of type nets.Module
        :param model: nets.Module containing trainable parameters
        :return: number of trainable parameters in model
        """
        num_parameters = 0
        for parameter in self.network.parameters():
            num_parameters += torch.numel(parameter)
        return num_parameters
    
    def get_model_size(self):
        n_params = self.get_num_parameters()
        logger.info(f'There are {n_params} trainable parameters in this model.')
        logger.info(f'Parameters total size is {n_params * 4 / 1024 / 1024} MB')