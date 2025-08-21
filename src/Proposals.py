

### Here are two proposals

# ================= Proposals: full-cov MVN and 1D eigen-line =================
import jax
import jax.numpy as jnp
from jax import random
import jax.scipy as jsp
import numpy as np
from jax.scipy.special import gammaln


# ---------- helpers ----------
def _as_period_vector(period, dim):
    """Allow scalar or per-dim vector periods."""
    if jnp.ndim(period) == 0:
        return jnp.full((dim,), period)
    return jnp.asarray(period)

def _eig_from_cov(cov):
    # symmetric eigendecomp (more stable than SVD for SPD)
    w, V = np.linalg.eigh(cov)      # w ascending
    w = np.clip(w, 1e-12, None)     # guard against tiny negatives
    return V, w

def _empirical_cov(x, ddof=1):
    x = np.asarray(x)
    if x.shape[0] <= 1:
        d = x.shape[1]
        return np.eye(d, dtype=x.dtype) * 1e-3
    return np.cov(x, rowvar=False, ddof=ddof)



def fold_periodic(x, fold_idx=(), period=1.0):
    """
    Fold specified coordinates into [0, period) using modulo.
    x: (..., dim)
    fold_idx: list/tuple of int indices to fold
    period: scalar or length-dim vector of periods
    """
    if not fold_idx:
        return x
    x = jnp.asarray(x)
    dim = x.shape[-1]
    P = _as_period_vector(period, dim)
    idx = jnp.asarray(fold_idx, dtype=jnp.int32)
    # Gather periods for those indices
    p_sel = P[idx]
    # Slice, mod, and write back
    xi = x[..., idx]
    xi = jnp.mod(xi, p_sel)
    return x.at[..., idx].set(xi)

def safe_cholesky(SPD):
    # robust against tiny asymmetry
    SPD = 0.5 * (SPD + SPD.T)
    return jnp.linalg.cholesky(SPD)

def log_normal_1d(x, mean, var):
    return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (x - mean) ** 2 / var)


# ---------- (1) Full-covariance Gaussian proposal ----------
def make_fullcov_proposal(cov, fold_idx=(), period=1.0):
    """
    Proposal: x' ~ N(x, (cd*scale)^2 * cov), with cd = 2.38 / sqrt(dim).
    Returns (sample_fn, logq_fn) matching your runner's API.
    """
    cov = jnp.asarray(cov)
    dim = cov.shape[0]
    L = safe_cholesky(cov)  # (dim, dim)
    logdet_cov = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    cd_const = 2.38 / jnp.sqrt(dim)

    def sample_fn(key, x, n, scale=1.0):
        """
        key: PRNGKey
        x: (dim,)
        n: int -> returns (n, dim)
        """
        x = jnp.asarray(x)
        eps = random.normal(key, shape=(n, dim))
        step = (cd_const * scale) * (eps @ L.T)     # (n, dim)
        x_new = x[None, :] + step
        return fold_periodic(x_new, fold_idx, period)

    def logq_fn(theta_new_batch, theta_old_batch, scale=1.0):
        """
        theta_new_batch: (B, dim)
        theta_old_batch: (B, dim)   (mean = theta_old)
        returns: (B,)
        """
        theta_new_batch = jnp.atleast_2d(theta_new_batch)
        theta_old_batch = jnp.atleast_2d(theta_old_batch)
        cd = cd_const * scale

        diff = theta_new_batch - theta_old_batch          # (B, dim)
        # Solve L z = (diff / cd)^T (lower triangular). z: (dim, B)
        z = jsp.linalg.solve_triangular(L, (diff / cd).T, lower=True)
        quad = jnp.sum(z**2, axis=0)                      # (B,)
        logdet_scaled = dim * jnp.log(cd**2) + logdet_cov
        log_norm = -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet_scaled)
        return log_norm - 0.5 * quad

    return sample_fn, logq_fn

# ---------- (2) Eigen-line (1D) proposal along covariance eigenvectors ----------
def make_eigenline_proposal(U, S, fold_idx=(), period=1.0, axis_probs=None, tol=1e-12):
    """
    1D move: pick axis i, sample s ~ N(0, (cd*scale)^2 * S[i]), propose x' = x + s * U[:, i].
    - U: (dim, dim) eigenvectors (columns)
    - S: (dim,)     eigenvalues (nonnegative)
    - axis_probs: optional length-dim probabilities over axes (defaults to uniform).
    Returns (sample_fn, logq_fn).
    """
    U = jnp.asarray(U)   # (dim, dim)
    S = jnp.asarray(S)   # (dim,)
    dim = U.shape[0]
    assert U.shape == (dim, dim) and S.shape == (dim,)

    cd_const = 2.38 / jnp.sqrt(dim)
    if axis_probs is None:
        log_axis_prob = -jnp.log(dim + 0.0)
        axis_probs = None
    else:
        axis_probs = jnp.asarray(axis_probs)
        axis_probs = axis_probs / jnp.sum(axis_probs)
        log_axis_prob = jnp.log(axis_probs + 1e-32)  # will index later

    def sample_fn(key, x, n, scale=1.0):
        """
        key, x: (dim,), n -> (n, dim)
        """
        x = jnp.asarray(x)
        k_idx, k_eps, k_cat = random.split(key, 3)

        if axis_probs is None:
            idx = random.randint(k_idx, shape=(n,), minval=0, maxval=dim)   # uniform axes
        else:
            # categorical sampling per draw
            logits = jnp.log(axis_probs)  # (dim,)
            idx = random.categorical(k_cat, logits[None, :].repeat(n, 0))   # (n,)

        stds = cd_const * scale * jnp.sqrt(S[idx])        # (n,)
        s = random.normal(k_eps, shape=(n,)) * stds       # (n,)
        U_sel = U[:, idx]                                 # (dim, n)
        delta = (U_sel * s).T                             # (n, dim)
        x_new = x[None, :] + delta
        return fold_periodic(x_new, fold_idx, period)

    def logq_fn(theta_new_batch, theta_old_batch, scale=1.0, tol=1e-12):
        """
        log q(new | old) for the eigen-line mixture:
        q = sum_i P(i) * N( s_i ; 0, (cd*scale)^2 S[i] )   if diff lies exactly on eigenvector i
            0                                              otherwise
        theta_*_batch: (B, dim)
        """
        theta_new_batch = jnp.atleast_2d(theta_new_batch)
        theta_old_batch = jnp.atleast_2d(theta_old_batch)
        cd = cd_const * scale

        diff = theta_new_batch - theta_old_batch          # (B, dim)
        a    = diff @ U                                   # (B, dim)  components in eigenbasis

        def per_point_logq(a_row):
            # choose axis with largest magnitude component
            abs_row = jnp.abs(a_row)
            i = jnp.argmax(abs_row)
            # zero out that axis and ensure the remainder is (numerically) zero
            resid2 = jnp.sum((a_row.at[i].set(0.0))**2)
            ok = resid2 <= tol**2

            s   = a_row[i]
            var = (cd**2) * S[i]
            log_gauss = log_normal_1d(s, 0.0, var)

            if axis_probs is None:
                log_pick = -jnp.log(dim + 0.0)         # uniform over axes
            else:
                log_pick = jnp.log(axis_probs[i] + 1e-32)

            return jnp.where(ok, log_pick + log_gauss, -jnp.inf)

        return jax.vmap(per_point_logq)(a)

    return sample_fn, logq_fn

def make_independence_proposal(mu, cov, fold_idx=(), period=1.0):
    """
    Independence: sample x' ~ N(mu, cov) (ignores current x).
    """
    L = jnp.linalg.cholesky(0.5*(cov+cov.T))
    dim = cov.shape[0]
    logdet = 2.0*jnp.sum(jnp.log(jnp.diag(L)))

    def sample_fn(key, x, n, scale=1.0):
        # scale here multiplies the covariance's Cholesky Independent of current chain state
        eps = random.normal(key, shape=(n, dim))
        x_new = mu[None,:] + (scale * (eps @ L.T))
        return fold_periodic(x_new, fold_idx, period)

    def logq_fn(theta_new_batch, theta_old_batch, scale=1.0):
        y = jnp.atleast_2d(theta_new_batch) - mu[None,:]
        z = jax.scipy.linalg.solve_triangular(L, (y/scale).T, lower=True)
        quad = jnp.sum(z**2, axis=0)
        dim = cov.shape[0]
        logdet_scaled = dim*jnp.log(scale**2) + logdet
        const = -0.5*(dim*jnp.log(2*jnp.pi) + logdet_scaled)
        return const - 0.5*quad

    return sample_fn, logq_fn

def make_pcn_proposal(mu, cov, fold_idx=(), period=1.0, beta=0.3):
    """
    pCN: x' = sqrt(1-beta^2)*x + beta*xi, xi ~ N(mu, cov).
    Symmetric in the pCN sense; we provide a valid logq anyway.
    """
    L = jnp.linalg.cholesky(0.5*(cov+cov.T))
    alpha = jnp.sqrt(1.0 - beta**2)
    dim = cov.shape[0]

    def sample_fn(key, x, n, scale=1.0):
        # scale multiplies the innovation term (acts like beta_tuned = beta*scale)
        eps = random.normal(key, shape=(n, dim)) @ L.T
        x_new = alpha * x[None,:] + (beta*scale) * eps + (1.0 - alpha) * mu[None,:]
        return fold_periodic(x_new, fold_idx, period)

    def logq_fn(theta_new_batch, theta_old_batch, scale=1.0):
        # Exact logq for pCN isn’t needed if you treat it as symmetric; keeping a placeholder:
        # Using forward kernel density is possible but more work; you can safely return zeros
        # if you use the symmetric MH form for pCN (common practice).
        return jnp.zeros((theta_new_batch.shape[0],), dtype=theta_new_batch.dtype)

    return sample_fn, logq_fn

def make_student_t_proposal(cov, fold_idx=(), period=1.0, nu=5.0):
    """
     Multivariate Student-t RW: x' = x + cd*scale * L @ t_nu,  L=chol(cov)
    Returns (sample_fn, logq_fn) matching your runner's API.
    """
    cov = jnp.asarray(cov)
    dim = cov.shape[0]
    L = safe_cholesky(cov)  # (dim, dim)

    cd_const = 2.38 / jnp.sqrt(dim)

    def sample_fn(key, x, n, scale=1.0):
        """
        key: PRNGKey
        x: (dim,)
        n: int -> returns (n, dim)
        """
        x = jnp.asarray(x)
        key, k1, k2 = jax.random.split(key, 3)
        z = jax.random.normal(k1, shape=(n, dim))
        g = jax.random.gamma(k2, shape=(n,), a=nu/2.) * (2./nu)  # chi^2_nu / nu
        t = z / jnp.sqrt(g)[:, None]  # i.i.d. t_nu in R^dim (independent components)
        step = cd_const * scale * (t @ L.T)
        x_new = x[None, :] + step
        return fold_periodic(x_new, fold_idx, period)

         

    def logq_fn(theta_new_batch, theta_old_batch, scale=1.0):
        """
        theta_new_batch: (B, dim)
        theta_old_batch: (B, dim)   (mean = theta_old)
        returns: (B,)
        """
        theta_new_batch = jnp.atleast_2d(theta_new_batch)
        theta_old_batch = jnp.atleast_2d(theta_old_batch)
        cd = cd_const * scale

        L_scaled = cd*L  # (dim, dim)

        diff = theta_new_batch - theta_old_batch
        y = jax.scipy.linalg.solve_triangular(L_scaled, diff.T, lower=True).T
        log_L = jnp.sum(jnp.log(jnp.diag(L_scaled)))  # log|L|
        # Multivariate Student-t density (up to constants that cancel in MH if both sides same nu/cov_s)
        m2 = jnp.sum(y*y, axis=1)
        # Full log-density for correctness (kept even if symmetric so logq_rev-logq_fwd cancels):

        c0 = (gammaln((nu+dim)/2.) - gammaln(nu/2.)
                - 0.5*dim*jnp.log(nu*jnp.pi) - log_L)
        logpdf = c0 - 0.5*(nu+dim)*jnp.log1p(m2/nu)
        return logpdf

    return sample_fn, logq_fn




# full_sample, full_logq = make_fullcov_proposal(cov, fold_idx=fold_indx, period=1.0)
# eig_sample,  eig_logq  = make_eigenline_proposal(U, S, fold_idx=fold_indx, period=1.0)

# proposal_fns     = (full_sample, eig_sample)
# log_proposal_fns = (full_logq,  eig_logq)

def build_general_mixture_components_per_chain(
    covs, eig_U, eig_S, means_for_indep,  # lists/arrays per chain
    temperatures, scales_base,             # arrays length n_chains
    fold_idx=(), period=1.0,
    beta_base=0.3,                          # pCN beta at T=1
    beta_temp_scale=True,                   # if True, beta -> beta*sqrt(T)
    kappa_line=3.0,                        # eigen-line larger factor
    weights=None                           # (n_chains, 3)
):
    """
    Returns:
      proposal_components: tuple of length n_chains; each is tuple of 3 sample_fns
      logq_components    : tuple of length n_chains; each is tuple of 3 logq_fns
      weights            : jnp.ndarray (n_chains, 3)
    """
    n_chains = len(covs)
    samp, logq = [], []
    W = []
    for c in range(n_chains):
        # temperature-dependent scales
        T = float(temperatures[c])
        # s_small = float(scales_base[c] * jnp.sqrt(T))    # component 1
        # s_line  = float(kappa_line * s_small)                          # component 2
        beta_c  = float(beta_base * (jnp.sqrt(T) if beta_temp_scale else 1.0))  # component 3 (pCN)

        # comp 1: full-cov RW
        # f_s, f_q = make_fullcov_proposal(jnp.asarray(covs[c]), fold_idx=fold_idx, period=period)
        # comp1_s = (lambda key, x, n, sf=s_small, f=f_s: f(key, x, n, sf))
        # comp1_q = (lambda newB, oldB, sf=s_small, q=f_q: q(newB, oldB, sf))
        f_s, f_q = make_student_t_proposal(jnp.asarray(covs[c]), fold_idx=fold_idx, period=period, nu=5.0)
        comp1_s = (lambda key, x, n, sf=1.0, f=f_s: f(key, x, n, sf))
        comp1_q = (lambda newB, oldB, sf=1.0, q=f_q: q(newB, oldB, sf))

        # comp 2: eigen-line (curvature-aligned big steps)
        # e_s, e_q = make_eigenline_proposal(jnp.asarray(eig_U[c]), jnp.asarray(eig_S[c]),
        #                                    fold_idx=fold_idx, period=period,
        #                                    axis_probs=(jnp.sqrt(jnp.asarray(eig_S[c]))/
        #                                                jnp.sum(jnp.sqrt(jnp.asarray(eig_S[c])))))
        e_s, e_q = make_eigenline_proposal(jnp.asarray(eig_U[c]), jnp.asarray(eig_S[c]),
                                           fold_idx=fold_idx, period=period)
        # comp2_s = (lambda key, x, n, sf=s_line, f=e_s: f(key, x, n, sf))
        # comp2_q = (lambda newB, oldB, sf=s_line, q=e_q: q(newB, oldB, sf))
        comp2_s = (lambda key, x, n, sf=1.0, f=e_s: f(key, x, n, sf))
        comp2_q = (lambda newB, oldB, sf=1.0, q=e_q: q(newB, oldB, sf))

        # comp 3: independence or pCN using running mean/cov
        mu_c = jnp.asarray(means_for_indep[c])
        g_s, g_q = make_pcn_proposal(mu_c, jnp.asarray(covs[c]), fold_idx=fold_idx, period=period, beta=beta_c)

        #     g_s, g_q = make_independence_proposal(mu_c, jnp.asarray(covs[c]), fold_idx=fold_idx, period=period)
        comp3_s = (lambda key, x, n, sf=1.0, f=g_s: f(key, x, n, sf))
        comp3_q = (lambda newB, oldB, sf=1.0, q=g_q: q(newB, oldB, sf))

        samp.append((comp1_s, comp2_s, comp3_s))
        logq.append((comp1_q, comp2_q, comp3_q))
        # Default to equal mixture weights when none are provided.  The previous
        # values summed to more than one, effectively biasing the sampler.  Each
        # component should receive an equal probability mass of one third.
        W.append([1.0/3.0, 1.0/3.0, 1.0/3.0] if weights is None else list(weights[c]))

    return tuple(samp), tuple(logq), jnp.asarray(W)




# def run_adaptive_3pro_mix_ptmcmc(
#     key,
#     initial_thetas: jnp.ndarray,      # (n_chains, dim)
#     temperatures: jnp.ndarray,        # (n_chains,)
#     true_logprob_fn,
#     base_cov: np.ndarray,             # (dim, dim)
#     fold_idx=(),
#     period=1.0,
#     m_epochs: int = 6,
#     N_steps: int = 500,
#     target_accept: float = 0.234,
#     eta: float = 0.05,                # Robbins–Monro step for RW/eigen scales
#     scale_init: float = 1.0,
#     scale_min: float = 0.1,
#     scale_max: float = 10.0,
#     shrink: float = 0.1,
#     jitter: float = 1e-6,
#     kappa_line: float = 3.0,
#     beta_base: float = 0.3,
#     beta_temp_scale: bool = True,
#     init_weights: np.ndarray | None = None,  # (n_chains, 3)
# ):
#     """
#     Per-chain adaptation with a 3-component mixture per chain: full-cov, eigen-line, pCN.
#     Returns:
#       state, covs(list), eig_U(list), eig_S(list), scales(np.ndarray), weights(jnp.ndarray), means(list), history(dict)
#     """
#     # --- init per-chain structures ---
#     key = jax.random.PRNGKey(int(key[0]) if isinstance(key, jnp.ndarray) else key)
#     initial_thetas = jnp.asarray(initial_thetas); temperatures = jnp.asarray(temperatures)
#     n_chains, dim = initial_thetas.shape

#     base_cov = _shrink_spd(np.asarray(base_cov, dtype=np.float64), shrink=shrink, jitter=jitter)
#     covs = [base_cov.copy() for _ in range(n_chains)]
#     eig_U, eig_S = [], []
#     for c in range(n_chains):
#         Uc, Sc = _eig_from_cov(covs[c]); eig_U.append(Uc); eig_S.append(Sc)

#     # per-chain RW/eigen scale
#     scales = np.full((n_chains,), float(scale_init), dtype=np.float64)

#     # per-chain mixture weights
#     if init_weights is None:
#         init_weights = np.tile(np.array([0.6, 0.25, 0.15], dtype=np.float64), (n_chains, 1))
#     else:
#         init_weights = np.asarray(init_weights, dtype=np.float64)
#         assert init_weights.shape == (n_chains, 3)
#     init_weights = init_weights / init_weights.sum(axis=1, keepdims=True)
#     weights = jnp.asarray(init_weights)

#     # simple per-chain running mean (for pCN/independence); start at initial thetas
#     means = [np.array(initial_thetas[c]) for c in range(n_chains)]
#     mean_ema = 0.9  # EMA coefficient

#     # --- build components & run one step to init state ---
#     props, logqs, weights = build_general_mixture_components_per_chain(
#         covs, eig_U, eig_S, means, temperatures, scales,
#         fold_idx=fold_idx, period=period,
#         kappa_line=kappa_line,
#         beta_base=beta_base, beta_temp_scale=beta_temp_scale,
#         weights=weights
#     )
#     key, subkey = jax.random.split(key)
#     final_state, states, infos = PTwarmup_collect_mixture(
#         subkey, initial_thetas, temperatures, true_logprob_fn,
#         props, logqs, weights, n_steps=1
#     )
#     state = final_state

#     history = {"accept_rate": [], "scale": [], "swap_rate": [], "cov_diag": [], "weights": []}

#     for epoch in range(m_epochs):
#         # --- run N steps with current mixture ---
#         key, subkey = jax.random.split(key)
#         final_state, states, infos = PTwarmup_collect_mixture(
#             subkey, state.thetas, state.temperatures, true_logprob_fn,
#             props, logqs, weights, n_steps=N_steps
#         )
#         state = final_state

#         # --- per-chain acceptance & accepted points ---
#         acc_pts_list, acc_rates = _extract_accepted_points_per_chain(infos)
#         last_states = np.array(states.thetas[-1])  # (n_chains, dim)

#         # update per-chain running mean (for pCN)
#         for c in range(n_chains):
#             means[c] = mean_ema * means[c] + (1.0 - mean_ema) * last_states[c]

#         # update per-chain covariance & eigenbasis
#         for c in range(n_chains):
#             Xc = last_states[c : c+1, :] if acc_pts_list[c].size == 0 \
#                  else np.vstack([last_states[c : c+1, :], acc_pts_list[c]])
#             cov_hat_c = _empirical_cov(Xc, ddof=1)
#             covs[c]   = _shrink_spd(cov_hat_c, shrink=shrink, jitter=jitter)
#             eig_U[c], eig_S[c] = _eig_from_cov(covs[c])

#         # Robbins–Monro scale for RW/eigen (per chain)
#         scales = np.clip(scales * np.exp(eta * (acc_rates - target_accept)), scale_min, scale_max)

#         # (optional) adapt mixture weights by component success; keep fixed for now
#         # Example to lift a component’s weight if it produced many accepts:
#         # comp = np.array(infos["comp_idx"])   # (N_steps, n_chains)
#         # acc  = np.array(infos["accepted"])   # (N_steps, n_chains)
#         # ... compute per-chain accepted-per-component and do an EMA toward that ...

#         # rebuild components with updated covs/eigs/means/scales
#         props, logqs, weights = build_general_mixture_components_per_chain(
#             covs, eig_U, eig_S, means, temperatures, scales,
#             fold_idx=fold_idx, period=period,
#             kappa_line=kappa_line,
#             beta_base=beta_base, beta_temp_scale=beta_temp_scale,
#             weights=weights
#         )

#         # diagnostics
#         swap_dec = np.array(infos["swap_decisions"])
#         swap_rate = swap_dec.mean().item() if swap_dec.size else 0.0
#         history["accept_rate"].append(acc_rates.copy())
#         history["scale"].append(scales.copy())
#         history["swap_rate"].append(swap_rate)
#         history["cov_diag"].append(np.stack([np.diag(C) for C in covs], axis=0))
#         history["weights"].append(np.array(weights))

#     return state, covs, eig_U, eig_S, scales, weights, means, history


# def PTwarmup_collect_mixture(
#     key,
#     initial_thetas: jnp.ndarray,   # (n_chains, dim)
#     temperatures: jnp.ndarray,     # (n_chains,)
#     log_prob_fn: Callable,         # function(theta) -> logpi(theta)
#     proposal_fns_per_chain: tuple, # len n_chains; each is tuple of sample_fns
#     logq_fns_per_chain: tuple,     # len n_chains; each is tuple of logq_fns
#     weights: jnp.ndarray,          # (n_chains, n_components)
#     n_steps: int
# ):
#     """
#     Parallel tempering warmup loop with per-chain mixture proposals.

#     proposal_fns_per_chain[c][k](key, theta, n, scale) -> (n, dim) proposals
#     logq_fns_per_chain[c][k](theta_new_batch, theta_old_batch, scale) -> (n,) logq
#     weights[c][k] is prob. of picking component k for chain c.
#     """

#     n_chains, dim = initial_thetas.shape
#     n_comp = weights.shape[1]

#     class PTState(NamedTuple):
#         thetas: jnp.ndarray       # (n_chains, dim)
#         log_probs: jnp.ndarray    # (n_chains,)
#         temperatures: jnp.ndarray # (n_chains,)
#         n_accepted: jnp.ndarray   # (n_chains,)
#         n_swaps: int
#         n_swap_attempts: int

#     # --- init log_probs ---
#     initial_log_probs = jax.vmap(log_prob_fn)(initial_thetas)
#     state0 = PTState(
#         thetas=initial_thetas,
#         log_probs=initial_log_probs,
#         temperatures=temperatures,
#         n_accepted=jnp.zeros(n_chains, dtype=int),
#         n_swaps=0,
#         n_swap_attempts=0
#     )

#     def one_step(carry, key):
#         state = carry
#         key, subkey_comp, subkey_prop, subkey_acc, subkey_swap = jax.random.split(key, 5)

#         # Pick component index for each chain
#         comp_idx = jax.vmap(
#             lambda w, k: random.categorical(k, jnp.log(w))
#         )(weights, random.split(subkey_comp, n_chains))  # (n_chains,)

#         # Draw proposals
#         def propose_for_chain(c_idx, chain_idx, key, theta):
#             sample_fn = proposal_fns_per_chain[chain_idx][c_idx]
#             return sample_fn(key, theta, 1, 1.0)[0]  # take shape (dim,)

#         keys_prop = random.split(subkey_prop, n_chains)
#         proposals = jax.vmap(propose_for_chain)(comp_idx, jnp.arange(n_chains), keys_prop, state.thetas)

#         # Compute logq forward and backward
#         def logq_for_chain(c_idx, chain_idx, theta_new, theta_old):
#             logq_fn = logq_fns_per_chain[chain_idx][c_idx]
#             return logq_fn(theta_new[None, :], theta_old[None, :], 1.0)[0]

#         logq_forward = jax.vmap(logq_for_chain)(comp_idx, jnp.arange(n_chains), proposals, state.thetas)
#         logq_backward = jax.vmap(logq_for_chain)(comp_idx, jnp.arange(n_chains), state.thetas, proposals)

#         # Compute log_probs for proposals
#         proposal_log_probs = jax.vmap(log_prob_fn)(proposals)

#         # Tempered log_probs
#         tempered_current = state.log_probs / state.temperatures
#         tempered_proposal = proposal_log_probs / state.temperatures

#         # MH acceptance
#         log_alpha = tempered_proposal - tempered_current + logq_backward - logq_forward
#         alpha = jnp.minimum(1.0, jnp.exp(log_alpha))
#         accept = random.uniform(subkey_acc, shape=(n_chains,)) < alpha

#         # Update
#         new_thetas = jnp.where(accept[:, None], proposals, state.thetas)
#         new_log_probs = jnp.where(accept, proposal_log_probs, state.log_probs)
#         new_n_acc = state.n_accepted + accept.astype(int)

#         # Parallel tempering swap
#         key, new_thetas, new_log_probs, swap_acc, swap_dec = parallel_tempering_swap(
#             subkey_swap, state.temperatures, new_thetas, new_log_probs
#         )

#         new_state = PTState(
#             thetas=new_thetas,
#             log_probs=new_log_probs,
#             temperatures=state.temperatures,
#             n_accepted=new_n_acc,
#             n_swaps=state.n_swaps + swap_acc.astype(int),
#             n_swap_attempts=state.n_swap_attempts + 1
#         )

#         info = {
#             "accepted": accept,
#             "comp_idx": comp_idx,
#             "swap_decisions": swap_dec
#         }
#         return new_state, info

#     keys = random.split(key, n_steps)
#     final_state, infos = jax.lax.scan(one_step, state0, keys)
#     return final_state, infos


# class ResBlock(nn.Module):
#     width: int
#     dropout: float = 0.0
#     @nn.compact
#     def __call__(self, x, train: bool): 
#         h = nn.LayerNorm()(h)
#         h = nn.Dense(self.width)(x)  # Moved here
#         h = nn.gelu(h)
#         if self.dropout > 0:
#             h = nn.Dropout(self.dropout)(h, deterministic=not train)
#         h = nn.Dense(self.width)(h)
#         return x + h

# class LogProbNet(nn.Module):
#     widths: Sequence[int] = (512, 512, 512, 512)   # Base
#     blocks_per_layer: int = 2
#     dropout: float = 0.0
#     heteroscedastic: bool = False
#     @nn.compact
#     def __call__(self, x, train: bool = True):
#         x = nn.LayerNorm()(x)
#         h = nn.Dense(self.widths[0])(x)
#         for w in self.widths:
#             for _ in range(self.blocks_per_layer):
#                 h = ResBlock(w, dropout=self.dropout)(h, train=train)
#         if self.heteroscedastic:
#             out = nn.Dense(2)(h)
#             mu, log_var = out[...,0], jnp.clip(out[..., 1], -12.0, 5.0)
#             return mu, log_var
#         else:
#             return nn.Dense(1)(h).squeeze(-1)