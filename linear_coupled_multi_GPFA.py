import math
import torch
import torch.nn as nn
from sklearn.decomposition import FactorAnalysis
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy as sp

class Multi_populations_GPFA(nn.Module):
    """
    Multi_populations_GPFA is a Gaussian Process Factor Analysis model for spike trian data of multiple days.
    It tries to find shared but not exact same latent trajectories across days.
    The model assumes that the latent trajectories are Gaussian processes, and each days has its own
    day-specific loadings. And the latent dynamics of each days can be linear transformed to each other. 
    """
    
    def __init__(self, n_factors, T, bin_width=0.001, tau=0.05, jitter=1e-3, 
                 beta=1, alpha=1, lamda=1, gamma=1, delta=1,
                 min_rho_m=1e-4, min_sigma_m=1e-4, min_tau=0.002):
        """
        Parameters
        ----------
        n_factors : int
            Number of latent factors
        bin_width : float
            Width of time bins in seconds
        tau : float
            GP timescale parameter (controls smoothness)
        jitter : float
            Small constant added to diagonal for numerical stability
        T : int
            Number of time bins (length of the time series)
        lamda : float
            Weight for the smoothness loss
        gamma : float
            Weight for the latent alignment loss
        delta : float
            Weight for the low rank constraint loss
        alpha : float
            Weight for the weight decay loss
        """
        
        super().__init__()
        self.n_factors = n_factors
        self.T = T
        self.bin_width = bin_width
        self.tau_init = tau
        self.jitter = jitter
        self.lamda = lamda  # weight for smoothness loss
        self.gamma = gamma  # weight for latent alignment loss
        self.delta = delta  # weight for low rank constraint loss
        self.beta = beta  # weight for reconstruct loss
        self.alpha = alpha # weight for weight decay loss
        self.min_rho_m = min_rho_m
        self.min_sigma_m = min_sigma_m
        self.min_tau = min_tau
        
        
        
    def _build_gp_covariance(self, T, tau):
        """
        Build GP covariance matrix K for squared-exponential kernel
        K(t_i, t_j) = σf²*exp(-0.5 * (t_i - t_j)² / τ²) + σn²*δ(t_i, t_j)
        where σf² = 1-σn² is fixed, and σn is a small constant to ensure positive definiteness.
        where τ is a learnable parameter.
        δ(t_i, t_j) is the Kronecker delta function, which is 1 if i == j and 0 otherwise.
        """
        times = torch.arange(T) * self.bin_width
        K = torch.zeros((T, T))
        
        for i in range(T):
            for j in range(T):
                # RBF kernel with timescale tau
                K[i, j] = torch.exp(-0.5 * (times[i] - times[j])**2 / tau**2)
                
        # Add small diagonal term for numerical stability
        K += self.jitter * torch.eye(T)

        return K
    
    
    def make_K_big(self, n_time_bins, gamma_m):
        '''
        K_big : torch.ndarray
        GP covariance matrix with dimensions (xDim * T) x (xDim * T).
        The (t1, t2) block is diagonal, has dimensions xDim x xDim, and
        represents the covariance between the state vectors at timesteps t1 and
        t2. K_big is sparse and striped.
        '''
        K_big = torch.zeros((self.n_factors * n_time_bins, self.n_factors * n_time_bins), dtype=torch.float64)
        Tdif = torch.tile(torch.arange(0, n_time_bins), (n_time_bins, 1)).T \
                - torch.tile(torch.arange(0, n_time_bins), (n_time_bins, 1))
        for i in range(self.n_factors):
            K = (1 - self.jitter) * torch.exp(-gamma_m[i] / 2 * Tdif ** 2) \
                + self.jitter * torch.eye(n_time_bins)
            K_big[i::self.n_factors, i::self.n_factors] = K
                    
        return K_big

    def make_K_big_inv(self, K_big, n_factors, n_timebins):
    
        K_big_inv = torch.zeros((n_factors * n_timebins, n_factors * n_timebins))
        for i in range(n_factors):
            K = K_big[i::n_factors, i::n_factors]
            K_big_inv[i::n_factors, i::n_factors] = torch.linalg.inv(K)
        
        return K_big_inv
    
    
    def logdet(self, A):
        """
        log(det(A)) where A is positive-definite.
        This is faster and more stable than using log(det(A)).

        Written by Tom Minka
        (c) Microsoft Corporation. All rights reserved.
        """
        U = torch.linalg.cholesky(A)
        return 2 * (torch.log(torch.diag(U))).sum()
    
    
    def cal_logdet_K_big(self, K_big, n_factors):
        
        logdet_K_big = 0
        for i in range(n_factors):
            K = K_big[i::n_factors, i::n_factors]
            logdet_K = self.logdet(K)
            logdet_K_big = logdet_K_big + logdet_K
            
        return logdet_K_big
    
    
    def rdiv(self, a, b):
        """
        Returns the solution to x b = a. Equivalent to MATLAB right matrix
        division: a / b
        """
        return torch.linalg.solve(b.T, a.T).T


    def fill_persymm(self, p_in, blk_size, n_blocks, blk_size_vert=None):
        """
        Fills in the bottom half of a block persymmetric matrix, given the
        top half.

        Parameters
        ----------
        p_in :  (xDim*Thalf, xDim*T) torch.ndarray
            Top half of block persymmetric matrix, where Thalf = ceil(T/2)
        blk_size : int
            Edge length of one block
        n_blocks : int
            Number of blocks making up a row of Pin
        blk_size_vert : int, optional
            Vertical block edge length if blocks are not square.
            `blk_size` is assumed to be the horizontal block edge length.

        Returns
        -------
        Pout : (xDim*T, xDim*T) torch.ndarray
            Full block persymmetric matrix
        """
        if blk_size_vert is None:
            blk_size_vert = blk_size

        Nh = blk_size * n_blocks
        Nv = blk_size_vert * n_blocks
        Thalf = int(math.floor(n_blocks / 2.0))
        THalf = int(math.ceil(n_blocks / 2.0))

        Pout = torch.empty((blk_size_vert * n_blocks, blk_size * n_blocks), dtype=torch.float64)
        Pout[:blk_size_vert * THalf, :] = p_in
        for i in range(Thalf):
            for j in range(n_blocks):
                Pout[Nv - (i + 1) * blk_size_vert:Nv - i * blk_size_vert,
                    Nh - (j + 1) * blk_size:Nh - j * blk_size] \
                    = p_in[i * blk_size_vert:(i + 1) *
                        blk_size_vert,
                        j * blk_size:(j + 1) * blk_size]

        return Pout


    def inv_persymm(self, M, blk_size):
        """
        Inverts a matrix that is block persymmetric.  This function is
        faster than calling inv(M) directly because it only computes the
        top half of inv(M).  The bottom half of inv(M) is made up of
        elements from the top half of inv(M).

        WARNING: If the input matrix M is not block persymmetric, no
        error message will be produced and the output of this function will
        not be meaningful.

        Parameters
        ----------
        M : (blkSize*T, blkSize*T) torch.ndarray
            The block persymmetric matrix to be inverted.
            Each block is blkSize x blkSize, arranged in a T x T grid.
        blk_size : int
            Edge length of one block

        Returns
        -------
        invM : (blkSize*T, blkSize*T) torch.ndarray
            Inverse of M
        logdet_M : float
            Log determinant of M
        """
        T = int(M.shape[0] / blk_size)
        Thalf = int(math.ceil(T / 2.0))
        mkr = blk_size * Thalf

        invA11 = torch.linalg.inv(M[:mkr, :mkr])
        invA11 = (invA11 + invA11.T) / 2

        # Multiplication of a sparse matrix by a dense matrix is not supported by
        # SciPy. Making A12 a sparse matrix here  an error later.
        off_diag_sparse = False
        if off_diag_sparse:
            A12 = sp.sparse.csr_matrix(M[:mkr, mkr:])
        else:
            A12 = M[:mkr, mkr:]

        term = invA11 @ A12
        F22 = M[mkr:, mkr:] - A12.T @ term

        res12 = self.rdiv(-term, F22)
        res11 = invA11 - res12 @ term.T
        res11 = (res11 + res11.T) / 2

        # Fill in bottom half of invM by picking elements from res11 and res12
        invM = self.fill_persymm(torch.hstack([res11, res12]), blk_size, T)

        # logdet_M = -self.logdet(invA11) + self.logdet(F22)

        return invM
    
    
    def reshape_F(self, tensor: torch.Tensor, *shape):
        """模拟 NumPy reshape(..., order='F')。"""
        return tensor.T.contiguous().reshape(*reversed(shape)).T    

        
    def _e_step(self, params_m, params_Km, spike_data):
        """
        Extracts latent trajectories from neural data, given GPFA model parameters.

        Parameters
        ----------
        spike_data : list
            whose s-th element (corresponding to the s-th
            experimental trial) has fields:
            N : torch.ndarray of shape (#units, #bins)
                neural data
            T : int
                number of bins
        params_m : parameters dict
            GPFA model parameters whe the following fields:
            Km : torch.ndarray # (n_factors*T, n_factors*T)
            Cm : torch.ndarray # (N, n_factors)
                FA factor loadings matrix
            dm : torch.ndarray # (N,)
                FA mean vector
            Rm : torch.ndarray # (N, N)
                FA noise covariance matrix 
            taum : torch.ndarray # (n_factors, )
                GP timescale

        Returns
        -------
        x_mean : # (n_trials, n_factors, n_time_bins)
            posterior mean of latent variables at each time bin
        """
        n_trials, n_neurons, n_timebins = spike_data.shape
        n_factors = self.n_factors

        x_mean = []

        # Precomputations
        rinv = torch.diag(1.0 / torch.diag(params_m['Rm']))
        # logdet_r = (torch.log(torch.diag(params_m['Rm']))).sum()

        c_rinv = params_m['Cm'].T @ rinv
        c_rinv_c = c_rinv @ params_m['Cm']

        K_big = params_Km['Km']
        K_big_inv = self.make_K_big_inv(K_big, n_factors, n_timebins)
        # logdet_k_big = self.cal_logdet_K_big(K_big, n_factors)
        
        # K_big = sparse.csr_matrix(K_big)
        blah = [c_rinv_c for _ in range(n_timebins)]
        c_rinv_c_big = torch.block_diag(*blah)  # (n_factors*n_timebins) x (n_factors*n_timebins)
        
        minv = self.inv_persymm(K_big_inv + c_rinv_c_big, n_factors).to(torch.float64)

        # Note that posterior covariance does not depend on observations,
        # so can compute once for all trials with same T.
        # xDim x xDim posterior covariance for each timepoint
        vsm = torch.full((n_factors, n_factors, n_timebins), torch.nan)
        idx = torch.arange(0, n_factors * n_timebins + 1, n_factors)
        for i in range(n_timebins):
            vsm[:, :, i] = minv[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]

        # T x T posterior covariance for each GP
        vsm_gp = torch.full((n_timebins, n_timebins, n_factors), torch.nan)
        for i in range(n_factors):
            vsm_gp[:, :, i] = minv[i::n_factors, i::n_factors]

        # dif is yDim x sum(T)
        dif = spike_data.permute(1, 0, 2).contiguous().view(n_neurons, n_trials * n_timebins) - params_m['dm'][:, torch.newaxis]
        # term1Mat is (xDim*T) x length(n_trials)
        # term1_mat = self.reshape_F(c_rinv @ dif, n_factors * n_timebins, -1)
        term1_mat = (c_rinv @ dif).reshape(n_factors, -1, n_timebins).permute(0, 2, 1).reshape(n_factors * n_timebins, -1)

        # Compute blkProd = CRinvC_big * invM efficiently
        # blkProd is block persymmetric, so just compute top half
        t_half = math.ceil(n_timebins / 2.0)
        blk_prod = torch.zeros((n_factors * t_half, n_factors * n_timebins))
        
        idx = range(0, n_factors * t_half + 1, n_factors)
        for i in range(t_half):
            blk_prod[idx[i]:idx[i + 1], :] = c_rinv_c @ minv[idx[i]:idx[i + 1], :]
            
        temp =  self.fill_persymm(torch.eye(n_factors * t_half, n_factors * n_timebins) - blk_prod, n_factors, n_timebins)
        blk_prod = K_big[:n_factors * t_half, :] @ temp
        
        # latent_variableMat is (xDim*T) x length(n_trials)
        latent_variable_mat = self.fill_persymm(
            blk_prod, n_factors, n_timebins) @ term1_mat

        for s in range(n_trials):
            x_mean.append(latent_variable_mat[:, s].reshape(n_timebins, n_factors).T) # (n_factors, n_timebins)
            # x_mean.append( self.reshape_F(latent_variable_mat[:, s], n_factors, n_timebins))

        x_mean = torch.stack(x_mean)

        return x_mean 


    def _initialize_parameters(self, multidays_spike_data):
        
        """
        Initialize model parameters using PCA and sample statistics.
        multidays_spike_data : list of spike data tensors which has M elements.
        each has the shape of (n_trials, N, T).
        N : int
            Number of neurons (or latent dimensions)
        T : int
            Number of time bins (length of the time series)
        """
                
        M = len(multidays_spike_data) # number of days
        
        for m in range(M):
            
            spike_data = multidays_spike_data[m] 
          
            n_trials, n_neurons, n_time_bins = spike_data.shape
            
            Ym = spike_data  # (n_trials, N, T)
            
            # stack all trials to initialize parameters
            y_all = Ym.permute(1, 0, 2).contiguous().reshape(n_neurons, n_trials * n_time_bins)
            
            fa = FactorAnalysis(n_components=self.n_factors, copy=True,
                        noise_variance_init=torch.diag(torch.cov(y_all, correction=0)))
            fa.fit(y_all.T)
            
            # Innitialize baseline firing rates 
            dm = y_all.mean(axis=1)  # (N, )
            
            # Initialize loading matrix C using PCA
            Cm = fa.components_.T # (N, D )
            
            # Initialize observation noise
            Rm = torch.diag(torch.from_numpy(fa.noise_variance_)) # （N, N)
            
            # Build GP covariance matrix for latent dynamics
            tau = self.tau_init  # initial timescale
            tau_m = torch.ones(self.n_factors) * tau 
            gamma_m = (self.bin_width / self.bin_width) ** 2 * torch.ones(self.n_factors)
            
            K_big = self.make_K_big(n_time_bins, gamma_m)
            
            params_m = {
                'dm': torch.as_tensor(dm).clone(),  # (N)
                'Rm': torch.as_tensor(Rm).clone(),   # (N, N)
                'Cm': torch.as_tensor(Cm).clone(),  # (N, n_factors)
                'taum': (torch.as_tensor(torch.exp(tau_m)+ self.min_tau)).clone() # (n_factors)
            }
            
            params_Km = {
                'Km': torch.as_tensor(K_big).clone(),   # (n_factors*T, n_factors*T)
            }
            
            # Initialize sigma to explicitly let each neuron accommodate different noise levels
            sigma_m = torch.ones(n_neurons)  # (N, )
            params_m['sigma_m'] = torch.as_tensor(sigma_m).clone()
            
            # Initialize rho to explicitly let each day accommodate different noise levels
            rho_m = 1.0  
            params_m['rho_m'] = torch.as_tensor(rho_m).clone()
            
            setattr(self, f"params_m_{m}", nn.ParameterDict(params_m))
            setattr(self, f"params_Km_{m}", params_Km)

        
        # Initialize Xm by esimating latent factors given current parameters (E-step)
        for m in range(M): 
            # if m == 0:
            Xm = self._e_step(params_m, params_Km, spike_data)  # (n_trials, n_factors, n_time_bins)
            # else:
            #     Xm_0 = getattr(self, f"Xm_0")
            #     A_mi_mj = getattr(self, f"A_mi_mj_{m-1}_{m}")
            #     b_mi_mj = getattr(self, f"b_mi_mj_{m-1}_{m}")
            #     import pdb; pdb.set_trace()
                
            #     Xm = A_mi_mj @ Xm_0 + b_mi_mj
            setattr(self, f"Xm_{m}", nn.Parameter(Xm))  # Store estimated latent factors
            
        # Initialize A matrix for linear transformation between days
        # by MSE 
        for mi in range(M-1):
            mj = mi + 1
            # A_mi_mj is a linear transformation matrix from day mi to day mj
            # A_mi_mj should be a square matrix of size n_factors x n_factors
            Xmi = getattr(self, f"Xm_{mi}")
            Xmj = getattr(self, f'Xm_{mj}')
            
            Xmi = torch.mean(Xmi,axis=0) # average across trials
            Xmj = torch.mean(Xmj ,axis=0)
            # import pdb; pdb.set_trace()
            _, T = Xmi.shape
            
            ones = torch.ones(1, T, dtype=Xmi.dtype)
            Xmi_aug = torch.cat([Xmi, ones], dim=0) 
            A_aug = Xmj @ Xmi_aug.T @ torch.inverse(Xmi_aug @ Xmi_aug.T)

            A_mi_mj = A_aug[:, :-1]  # (n_factors, n_factors)
            setattr(self, f"A_mi_mj_{mi}_{mj}", nn.Parameter(A_mi_mj))
            
            b_mi_mj = A_aug[:, -1]  # (n_factors,)
            setattr(self, f"b_mi_mj_{mi}_{mj}", nn.Parameter(b_mi_mj))
            
            Xmj_pred = (A_mi_mj @ Xmi + b_mi_mj.unsqueeze(1))  
            mse = torch.norm(Xmj - Xmj_pred) / T
            print("Xmi and Xmj fitting mse:", mse.item())
                

    def reconstruct_loss(self, spike_data, params_m, m):
        """
        Compute the reconstruction loss for day m.
        The loss is computed as 
        """
        Ym = spike_data  # (n_trials, N, T)
        dm = params_m['dm'] # (N)
        Rm = params_m['Rm']  # (N, N)
        Cm = params_m['Cm']  # (N, n_factors)
        Xm = getattr(self, f"Xm_{m}")  # (n_trials, n_factors, n_time_bins)
        rho_m = torch.exp(params_m['rho_m']) + self.min_rho_m  # int
        sigma_m = torch.exp(params_m['sigma_m']) + self.min_sigma_m # (N,)
        
        # log-det term:  (T/2) log(2πρ²σ_q²)
        T = Ym.shape[2]  # number of time bins
        log_det = torch.abs((T / 2.0) * torch.log(2 * math.pi * rho_m**2 * sigma_m**2 )) # (N,)
        
        # quadratic term:  ||·||² / (2ρ²σ_q²)
        Y_hat = torch.matmul(Cm, Xm)                       # (n_trials,N,T)
        residual = Ym - (Y_hat + dm.unsqueeze(-1).unsqueeze(0))         # (n_trials,N,T)
        quad = residual.pow(2).sum(-1, keepdim=True) / (2 * rho_m**2 * sigma_m**2)
        
        # sum over neurons q = 1..N
        loss_rec = (log_det + quad).sum() # (n_trials,)
        
        # Average over trials AND neurons
        loss_rec /= (Ym.shape[0] * Ym.shape[1])  # (n_trials, N)
        

        return loss_rec
    
    
    def _gp_nll(self, Xp, Kp):
        """
        negative log GP prior for one latent row
        ½ log|2πK| + ½ xᵀK⁻¹x (batch-free)
        """
        log_det_term = 0.5 * torch.logdet(2 * torch.pi * Kp)
        Kp_inv = torch.inverse(Kp)  # Inverse of covariance matrix
        quad_term = 0.5 * torch.matmul(torch.matmul(Xp, Kp_inv), Xp.T)
        loss_gp_p = log_det_term + quad_term
        return loss_gp_p    
    
    
    def smoothness_loss(self, params_Km, m):
        """
        Compute the smoothness loss for day m.
        The loss is computed as the negative log likelihood of the GP prior on the latent factors.
        """
        Xm = getattr(self, f"Xm_{m}") # (n_trials, n_factors, T)
        Km = params_Km['Km']  # (n_factors*T, n_factors*T)
        
        n_trials, n_factors, n_time_bins = Xm.shape
        loss_gp = 0.0
        for s in range(n_trials):
            for p in range(n_factors):
                Xp = Xm[s, p, :].view(1,-1) # (T,)
                Kp = Km[p::n_factors, p::n_factors] # (T, T)
                loss_gp += self._gp_nll(Xp, Kp)
        
        # Average over trials and factors
        loss_gp /= (n_trials * n_factors)
        
        return loss_gp
    
    
    def weight_decay_loss(self, params_m):
        """
        Compute the weight decay loss.
        This is a regularization term to prevent overfitting.
        """
        Cm = params_m['Cm']  # (N, n_factors)
        loss_wd = 0.5 * torch.sum(Cm ** 2)  # sum over all loadings
        
        return loss_wd
    
    
    def latent_alignment_loss(self, Xm_list):
        """
        Compute the latent alignment loss.
        This loss encourages the latent trajectories across different days to be aligned.
        Xm_list : list of Xm_m matrices, each has the shape of (n_trials, n_factors, T).
        n_trials : int
            Number of trials (or days)
        n_factors : int
            Number of latent factors
        T : int
            Number of time bins (length of the time series)
        The loss is computed as the average squared difference between the mean latent trajectories across days.
        """
        # average across trials 
        Xm_mean_list = [torch.mean(Xm, dim=0) for Xm in Xm_list]  # (n_factors, T)
        # compute the difference between each Xm_mean
        loss_latent_alignment = 0.0
        M = len(Xm_mean_list)
        for i in range(M):
            for j in range(i + 1, M):
                Xm_i = Xm_mean_list[i]
                Xm_j = Xm_mean_list[j]
                # Compute the squared difference
                diff = Xm_i - Xm_j  # (n_factors, T)
                loss_latent_alignment += torch.sum(diff ** 2)  # sum over all factors and time bins
        # Average over pairs of days
        loss_latent_alignment /= (M * (M - 1) / 2)  # average over pairs of days
        
        return loss_latent_alignment
    
    
    
    def low_rank_constraint_loss(self, A_list):
        """
        Compute the low rank constraint loss.
        This loss encourages the A matrices to be low rank.
        A_list : list of A_mi_mj matrices, each has the shape of (n_factors, n_factors).
        """
        # let each A matrix be low rank
        loss_low_rank = 0.0
        for A_mi_mj in A_list:
            # Compute the singular values of A_mi_mj
            U, S, V = torch.svd(A_mi_mj)
            # Compute the squared singular values
            S_squared = S ** 2
            # Add the squared singular values to the loss
            loss_low_rank += torch.sum(S_squared)  # sum over all singular values
        # Average over pairs of days
        loss_low_rank /= len(A_list)  # average over pairs of days
                
        return loss_low_rank
   
   
   
    def forward(self, multidays_spike_data):
        """
        Forward pass of the model.
        multidays_spike_data : list of spike data tensors which has M elements.
        each has the shape of (n_trials, N, T).
        N : int
            Number of neurons (or latent dimensions)
        T : int
            Number of time bins (length of the time series)
        """


        
        M = len(multidays_spike_data)  # number of days
        
        loss_rec_all = 0
        loss_gp_all = 0
        loss_wd_all = 0
        for m in range(M):
            # Get parameters for day m
            params_m = getattr(self, f"params_m_{m}")
            params_Km = getattr(self, f"params_Km_{m}")
            # Get spike data for day m
            spike_data = multidays_spike_data[m] # (n_trials, N, T)
                
            # compute the reconstruction loss 
            loss_rec = self.reconstruct_loss(spike_data, params_m, m)  
            loss_rec_all = loss_rec_all + loss_rec
            
            # compute the smoothness loss
            loss_gp = self.smoothness_loss(params_Km, m)
            loss_gp_all = loss_gp_all + loss_gp
            
            # compute the weight decay loss
            loss_wd = self.weight_decay_loss(params_m)
            loss_wd_all = loss_wd_all + loss_wd
            
        
        # compute the latent alignment loss
        Xm_list = [getattr(self, f"Xm_{m}") for m in range(M)]
        loss_latent_alignment = self.latent_alignment_loss(Xm_list)
        
        # compute the low rank constraint loss
        A_list = [getattr(self, f"A_mi_mj_{mi}_{mi+1}") for mi in range(M-1)]
        loss_low_rank = self.low_rank_constraint_loss(A_list)
        
        
        return self.beta*loss_rec_all + self.lamda*loss_gp_all + self.alpha*loss_wd_all  + self.gamma*loss_latent_alignment + self.delta*loss_low_rank
            

    
