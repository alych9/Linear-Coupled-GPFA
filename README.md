# Linear-coupled-multi-GPFA
Alignment of multi-days or multi-populations neural dynamics based on modified GPFA algorithm.

# Motivation
With the rapid advancement of neural‐recording technologies, methods for simultaneously monitoring large neural populations across days and brain regions are becoming increasingly mature. Chronic implantation of Neuropixels probes is among the most promising approaches: such implants have already enabled year-long recordings from two to three brain regions in mammals. Nonetheless, even chronic probes cannot remain perfectly stationary in the brain; probe drift and the natural turnover of neurons mean that the populations recorded on different days inevitably differ. This variability complicates the analysis of unified neural-dynamics trajectories across days. To tackle the problem of aligning dynamics while explicitly accounting for differences in the recorded populations, we propose the Linear-Coupled-Multi-GPFA method. The same framework can also be used to align dynamical trajectories across distinct brain regions.

# GPFA
**Gaussian Process Factor Analysis (GPFA)** models high-dimensional neural recordings $\mathbf{y}(t)\in\mathbb{R}^{N}$ as noisy linear projections of a low-dimensional set of latent trajectories $ \mathbf{x}(t)\in\mathbb{R}^{K}$ that evolve smoothly in time:

$$
\boxed{\;\mathbf{y}(t)=\mathbf{C}\,\mathbf{x}(t)+\mathbf{d}+\boldsymbol{\varepsilon}(t), 
\quad \boldsymbol{\varepsilon}(t)\sim\mathcal{N}\!\bigl(\mathbf{0},\mathbf{R}\bigr)\;}
$$

Each latent dimension $x_k(t)$ is treated as a zero-mean Gaussian process (GP) with covariance kernel $k_k(t,t')$ (typically squared-exponential), ensuring temporal smoothness:

$$
x_k(t)\sim\mathcal{GP}\!\bigl(0,\;k_k(t,t')\bigr), \qquad k_k(t,t')=\sigma_k^{2}\exp\!\left[-\tfrac{(t-t')^{2}}{2\ell_k^{2}}\right].
$$

Inference jointly learns the loading matrix $\mathbf{C}$, offsets $\mathbf{d}$, noise covariances $\mathbf{R}$, and GP hyper-parameters $\{\sigma_k,\ell_k\}$, yielding smooth low-dimensional trajectories that capture shared neural dynamics while filtering out independent noise.
