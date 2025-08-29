
"""Proposal utilities for parallel tempering samplers."""
import jax
import jax.numpy as jnp
from jax import random, vmap
import jax.scipy as jsp
import numpy as np
from jax.scipy.special import gammaln
from jax.scipy.linalg import solve_triangular


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
        beta_c  = float(beta_base * (jnp.sqrt(T) if beta_temp_scale else 1.0))  # component 3 (pCN)

        # component 1: Student-t random walk
        f_s, f_q = make_student_t_proposal(jnp.asarray(covs[c]), fold_idx=fold_idx, period=period, nu=5.0)
        comp1_s = (lambda key, x, n, sf=1.0, f=f_s: f(key, x, n, sf))
        comp1_q = (lambda newB, oldB, sf=1.0, q=f_q: q(newB, oldB, sf))

        # component 2: eigen-line moves
        e_s, e_q = make_eigenline_proposal(jnp.asarray(eig_U[c]), jnp.asarray(eig_S[c]),
                                           fold_idx=fold_idx, period=period)
        comp2_s = (lambda key, x, n, sf=1.0, f=e_s: f(key, x, n, sf))
        comp2_q = (lambda newB, oldB, sf=1.0, q=e_q: q(newB, oldB, sf))

        # component 3: pCN using running mean/cov
        mu_c = jnp.asarray(means_for_indep[c])
        g_s, g_q = make_pcn_proposal(mu_c, jnp.asarray(covs[c]), fold_idx=fold_idx, period=period, beta=beta_c)
        comp3_s = (lambda key, x, n, sf=1.0, f=g_s: f(key, x, n, sf))
        comp3_q = (lambda newB, oldB, sf=1.0, q=g_q: q(newB, oldB, sf))

        samp.append((comp1_s, comp2_s, comp3_s))
        logq.append((comp1_q, comp2_q, comp3_q))
        # Default to equal mixture weights when none are provided.  The previous
        # values summed to more than one, effectively biasing the sampler.  Each
        # component should receive an equal probability mass of one third.
        W.append([1.0/3.0, 1.0/3.0, 1.0/3.0] if weights is None else list(weights[c]))

    return tuple(samp), tuple(logq), jnp.asarray(W)


# ---------------------------------------------------------------------------
# Proposal helpers originally defined in ``runner.py``


def _build_epoch_components(
    covs: jnp.ndarray,
    scale_small: jnp.ndarray,
    scale_line: jnp.ndarray,
    scale_big: jnp.ndarray,
    jitter: float = 1e-9,
):
    """Construct per-chain proposal components for an epoch.

    Parameters
    ----------
    covs: (C, D, D)
        Empirical covariance matrices for each chain.
    scale_small, scale_line, scale_big: (C,)
        Per-chain scale factors for the full-covariance, eigen-line and
        "big" proposal components respectively.
    jitter: float, optional
        Diagonal jitter added for numerical stability.

    Returns
    -------
    fullcov, eigenline : dict
        Dictionaries with pre-computed quantities used by the proposal
        factories in :func:`run_epoch_device_fast`.
    """

    C, D, _ = covs.shape
    I = jnp.eye(D)[None, :, :]
    sym = 0.5 * (covs + jnp.swapaxes(covs, -1, -2)) + jitter * I

    L_chol = jnp.linalg.cholesky(sym)  # (C, D, D)

    S, U = jnp.linalg.eigh(sym)
    S = jnp.clip(S, 1e-12, None)
    axis_logits = jnp.log(jnp.sqrt(S) + 1e-12)

    fullcov = {
        "L_chol": L_chol,
        "scale_small": scale_small,
        "scale_big": scale_big,
    }
    eigenline = {"U": U, "S": S, "axis_logits": axis_logits, "scale": scale_line}
    return fullcov, eigenline


def _as_ensemble(x):
    """Ensure x has shape (C, W, D)."""
    if x.ndim == 2:  # (C, D) -> (C, 1, D)
        x = x[:, None, :]
        squeeze = True
    elif x.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"x must be (C,D) or (C,W,D); got {x.shape}")
    C, W, D = x.shape
    return x, C, W, D, squeeze

def _propose_fullcov(key, x, L, scale):
    """Full-covariance Gaussian random walk proposal.
        Accepts x of shape (C,D) or (C,W,D). Returns same rank as x.
    """
    x, C, W, D, squeeze = _as_ensemble(x)
    z = random.normal(key, (C, W, D))
    # transform noise per chain via Cholesky
    eps = jnp.einsum("cij,cwj->cwi", L, z)  # (C,W,D)
    # scaling
    # scale_c = jnp.asarray(scale)            # () or (C,)
    # if scale_c.ndim == 0:
    #     scale_c = jnp.full((C,), scale_c)
    # 2.38/sqrt(D) factor (classic)
    cd_const = 2.38 / jnp.sqrt(D) * 0.5
    # step = cd_const * scale_c[:, None, None] * eps
    step = cd_const * eps
    out = x + step
    return out[:, 0, :] if squeeze else out


def _propose_eigenline(key_axis, key_noise, x, U, S, scale, axis_logits=None):
    """Propose along a single eigen-direction for each chain."""
    x, C, W, D, squeeze = _as_ensemble(x)

    # choose axis per (C,W)
    if axis_logits is None:
        axes = random.randint(key_axis, (C, W), 0, D)  # (C,W)
    else:
        lg = jnp.asarray(axis_logits)
        if lg.shape == (C, D):
            # same categorical over axes for all walkers at chain c
            ks = random.split(key_axis, C)
            axes_c = vmap(lambda lg_c, k: random.categorical(k, lg_c))(lg, ks)  # (C,)
            axes = jnp.repeat(axes_c[:, None], W, axis=1)                        # (C,W)
        elif lg.shape == (C, W, D):
            ks = random.split(key_axis, C * W).reshape(C, W, 2)  # 2 keys not needed; but shape ok
            # Use one key per (c,w)
            def cat_one(lg_cw, kpair):
                # kpair[0] is fine
                return random.categorical(kpair[0], lg_cw)
            axes = jax.vmap(jax.vmap(cat_one, in_axes=(0,0)), in_axes=(0,0))(lg, ks)  # (C,W)
        else:
            raise ValueError(f"axis_logits must be (C,D) or (C,W,D); got {lg.shape}")

    # scale_c = jnp.asarray(scale)
    # if scale_c.ndim == 0:
    #     scale_c = jnp.full((C,), scale_c)
    
    # gather eigenvectors/eigenvalues for chosen axis
    # U[:, :, axis] -> (C,W,D)
    U_ax = U[jnp.arange(C)[:, None], :, axes]            # (C,W,D)
    S_ax = S[jnp.arange(C)[:, None], axes]               # (C,W)

    r = random.normal(key_noise, (C, W))
    # step_mag = (scale_c[:, None] * r * jnp.sqrt(S_ax))[..., None]  # (C,W,1)
    step = ( r * jnp.sqrt(S_ax))[..., None]  # (C,W,1)
    out = x + step * U_ax
    return out[:, 0, :] if squeeze else out


def _propose_student_t(key_norm, key_gamma, x, L, scale, nu=5.0):
    """Student-t random walk proposal."""
    x, C, W, D, squeeze = _as_ensemble(x)

    z = random.normal(key_norm, (C, W, D))   # base normal
    nu_c = jnp.asarray(nu)
    if nu_c.ndim == 0:
        nu_c = jnp.full((C,), nu_c)

    # Gamma ~ χ²_ν as Gamma(ν/2, 1/2) -> here we use Gamma(ν/2) and rescale
    g = random.gamma(key_gamma, a=nu_c[:, None] / 2.0, shape=(C, W))  # (C,W)
    g = g * (2.0 / nu_c[:, None])                                     # (C,W)
    t = z / jnp.sqrt(g[..., None])                                    # (C,W,D)

    t_tr = jnp.einsum("cij,cwj->cwi", L, t)                           # (C,W,D)

    # scale_c = jnp.asarray(scale)
    # if scale_c.ndim == 0:
    #     scale_c = jnp.full((C,), scale_c)

    cd_const = 2.38 / jnp.sqrt(D) * 0.5
    # step = cd_const * scale_c[:, None, None] * t_tr
    step = cd_const  * t_tr
    out = x + step
    return out[:, 0, :] if squeeze else out


# -------------------- pCN proposal --------------------
def _propose_pcn(key, x, mu, L, scale, beta=0.3):
    """
    Preconditioned Crank–Nicolson proposal for ensemble.
    mu: (C,D); L: (C,D,D); scale: () or (C,); beta: () or (C,)
    """
    x, C, W, D, squeeze = _as_ensemble(x)

    beta_c  = jnp.asarray(beta)
    scale_c = jnp.asarray(scale)
    if beta_c.ndim == 0:
        beta_c = jnp.full((C,), beta_c)
    if scale_c.ndim == 0:
        scale_c = jnp.full((C,), scale_c)
    b = beta_c * scale_c                       # (C,)
    b = jnp.clip(b, 1e-8, 1.0 - 1e-8)
    a = jnp.sqrt(1.0 - b * b)                  # (C,)

    eps = random.normal(key, (C, W, D))
    eps = jnp.einsum("cij,cwj->cwi", L, eps)   # (C,W,D)

    mu_c = jnp.asarray(mu)                     # (C,D)
    out = a[:, None, None] * x + b[:, None, None] * eps + (1.0 - a)[:, None, None] * mu_c[:, None, :]
    return out[:, 0, :] if squeeze else out

# -------------------- pCN Δlog q(y|x) - Δlog q(x|y) --------------------
def _pcn_logq_delta(x, y, mu, L, scale, beta=0.3):
    """
    Difference in log proposal densities for pCN proposals:
      Δ = log q(y|x) - log q(x|y)
    Works for x,y with shape (C,D) or (C,W,D). Returns (C,) or (C,W) respectively.
    """
    # upgrade to ensemble
    x, C, W, D, squeeze = _as_ensemble(x)
    y, C2, W2, D2, _ = _as_ensemble(y)
    assert (C2, W2, D2) == (C, W, D)

    beta_c  = jnp.asarray(beta)
    scale_c = jnp.asarray(scale)
    if beta_c.ndim == 0:
        beta_c = jnp.full((C,), beta_c)
    if scale_c.ndim == 0:
        scale_c = jnp.full((C,), scale_c)
    b  = jnp.clip(beta_c * scale_c, 1e-8, 1.0 - 1e-8)   # (C,)
    b2 = b * b
    a  = jnp.sqrt(1.0 - b2)                             # (C,)

    mu_c = jnp.asarray(mu)  # (C,D)
    m_x = a[:, None, None] * x + (1.0 - a)[:, None, None] * mu_c[:, None, :]
    m_y = a[:, None, None] * y + (1.0 - a)[:, None, None] * mu_c[:, None, :]

    # Mahalanobis squared using L (C,D,D), for each chain over all walkers
    # We compute ||L^{-1} v||^2 where v has shape (W,D) per chain.
    def maha_sq_chain(Lc, Vw):  # Lc: (D,D), Vw: (W,D) -> (W,)
        # solve for each walker as RHS; solve_triangular supports batched RHS via trailing dims
        Wloc = solve_triangular(Lc, Vw.T, lower=True)  # (D,W)
        return jnp.sum(Wloc * Wloc, axis=0)            # (W,)

    maha_y_given_x = jax.vmap(maha_sq_chain, in_axes=(0, 0))(L, (y - m_x))  # (C,W)
    maha_x_given_y = jax.vmap(maha_sq_chain, in_axes=(0, 0))(L, (x - m_y))  # (C,W)

    delta = -0.5 * (maha_x_given_y - maha_y_given_x) / (b2[:, None] + 1e-32)  # (C,W)
    return delta[:, 0] if squeeze else delta



# ----- NEW: Goodman–Weare stretch move for an ensemble of walkers -----
def _propose_stretch_ensemble(
    key_partner,
    key_scale,
    X,            # (C, W, D)
    a: float = 2.0,
):
    C, W, D = X.shape
    arange_C = jnp.arange(C)[:, None]
    arange_W = jnp.arange(W)[None, :]

    # --- self-avoiding partners: partner = (w + off) % W, off ∈ {1..W-1}
    # Works also for W=1 (we’ll just fall back to identity below).
    off = jax.random.randint(key_partner, shape=(C, W), minval=1, maxval=jnp.maximum(W, 2))
    partner = (arange_W + off) % jnp.maximum(W, 1)

    Y = X[arange_C, partner, :]  # (C, W, D)

    # --- draw z with g(z) ∝ 1/sqrt(z) on [1/a, a]
    u = jax.random.uniform(key_scale, shape=(C, W))
    sa = jnp.sqrt(a)
    z = (u * (sa - 1.0 / sa) + 1.0 / sa) ** 2  # (C, W)

    # If W==1, partner==self; make the move a no-op (keeps code safe).
    same = (W == 1)
    X_prop = jnp.where(same, X, Y + z[..., None] * (X - Y))

    log_J = (D - 1) * jnp.log(jnp.where(same, jnp.ones_like(z), z))
    return X_prop, log_J, z


def _propose_stretch(
    key_partner,
    key_scale,
    X,                       # (C, W, D)
    a: float = 2.0,
    z: jnp.ndarray | None = None,   # (C, W) or None
):
    """
    Goodman–Weare stretch proposal for an ensemble (per temperature c, across walkers w).

    If `z` is provided (shape (C, W)), each (c, w) only chooses a partner among walkers
    at the same temperature `c` that share the same label z[c,w]. If no such partner
    exists, the move for that (c,w) is a no-op with log-Jacobian 0.

    If `z` is None, partners are chosen uniformly among all NON-SELF walkers in (c, ·).
    For W == 1 the move is a no-op with log-Jacobian 0.

    Returns:
      X_prop:        (C, W, D)
      logJ:          (C, W)
      partner_idx:   (C, W)   indices of chosen partners (self if no eligible partner)
      has_partner:   (C, W)   bool mask, True if a non-self eligible partner existed
      z_factor:      (C, W)   the sampled stretch factor on [1/a, a] (useful for debug)
    """
    C, W, D = X.shape
    arW = jnp.arange(W)

    # --- build eligibility mask (C, W, W) ---
    # not-self mask shared by both modes
    not_self = (arW[None, :, None] != arW[None, None, :])  # (1, W, W) -> broadcast
    if z is None:
        # unrestricted: any non-self partner eligible
        elig = jnp.broadcast_to(not_self, (C, W, W))        # (C, W, W)
    else:
        # restricted: same z AND not self
        same_z = (z[:, :, None] == z[:, None, :])           # (C, W, W)
        elig   = same_z & not_self                          # (C, W, W)

    has_partner = elig.any(axis=-1)                         # (C, W)

    # --- sample partner via masked Gumbel-max (stable inside jit) ---
    logits  = jnp.where(elig, 0.0, -1e9)                    # (C, W, W)
    g       = jax.random.gumbel(key_partner, logits.shape)
    partner = jnp.argmax(logits + g, axis=-1)               # (C, W)
    # fallback to self where no eligible partner
    partner = jnp.where(has_partner, partner, arW[None, :]) # (C, W)

    # gather partners' positions
    Y = X[jnp.arange(C)[:, None], partner, :]               # (C, W, D)

    # --- draw stretch factor zfac ~ g(z) ∝ 1/sqrt(z) on [1/a, a] ---
    u   = jax.random.uniform(key_scale, (C, W))
    sa  = jnp.sqrt(a)
    zfac = (u * (sa - 1.0 / sa) + 1.0 / sa) ** 2            # (C, W)

    # propose; no-op where no partner or W==1
    sameW = (W == 1)
    do_move = has_partner & (~sameW)                        # (C, W)
    X_prop  = jnp.where(do_move[..., None], Y + zfac[..., None] * (X - Y), X)
    logJ    = jnp.where(do_move, (D - 1) * jnp.log(zfac), 0.0)

    return X_prop, logJ, partner, has_partner, zfac



def _propose_de_two_point(
    key_partner,
    key_gamma,
    X,                             # (C, W, D)
    z: jnp.ndarray | None = None,  # (C, W) or None  (restrict partners to same label if provided)
    *,
    same_z_required: bool = True,  # True => only same-z partners (if z provided)
    gamma: float | None = None,    # fixed |γ|; if None, draw symmetric Normal(0, σ^2)
    gamma_scale: float = 2.38,     # σ = gamma_scale / sqrt(2D) when gamma is None (classic DE)
    crossover_rate: float = 0.8,   # per-dimension prob to update (DE "CR"); set 1.0 for full update
    jitter_scale: float = 0.0,     # ε ~ N(0, jitter_scale^2 I) (small)
):
    """
    Differential-Evolution (two-point) proposal for each (c,w):
        x' = x + γ * (y - z) + ε
    Partners y and z are sampled among walkers at the same temperature.
    If z is provided and same_z_required=True, restrict to same z[c,w].
    Safe when fewer than 2 partners exist: returns a no-op for that (c,w).

    Returns:
      X_prop     : (C, W, D)
      idx_y      : (C, W)  chosen partner indices for y   (self if no pair)
      idx_z      : (C, W)  chosen partner indices for z   (self if no pair)
      has_pair   : (C, W)  bool, True if ≥2 eligible partners existed
      mask_used  : (C, W, D) bool crossover mask actually applied
    """
    C, W, D = X.shape
    arW = jnp.arange(W)

    # ---------- eligibility (C,W,W): non-self, and optionally same-z ----------
    not_self = (arW[None, :, None] != arW[None, None, :])  # (1,W,W) -> broadcast
    if (z is not None) and same_z_required:
        same_lbl = (z[:, :, None] == z[:, None, :])        # (C,W,W)
        elig = same_lbl & not_self
    else:
        elig = jnp.broadcast_to(not_self, (C, W, W))

    n_elig   = elig.sum(axis=-1)                           # (C,W)
    has_pair = n_elig >= 2

    # ---------- sample two distinct partners via masked Gumbel-max ----------
    logits = jnp.where(elig, 0.0, -1e9)                    # (C,W,W)  eligible entries 0, else -inf
    g1     = random.gumbel(key_partner, logits.shape)
    idx_y  = jnp.argmax(logits + g1, axis=-1)              # (C,W)

    # mask out y to sample distinct z
    mask_y  = jax.nn.one_hot(idx_y, W, dtype=bool)         # (C,W,W)
    elig2   = elig & (~mask_y)
    logits2 = jnp.where(elig2, 0.0, -1e9)

    key_partner2 = random.fold_in(key_partner, 1)
    g2     = random.gumbel(key_partner2, logits2.shape)
    idx_z  = jnp.argmax(logits2 + g2, axis=-1)             # (C,W)

    # fallback to self if <2 partners (we'll no-op those below)
    idx_y = jnp.where(has_pair, idx_y, arW[None, :])
    idx_z = jnp.where(has_pair, idx_z, arW[None, :])

    # gather partners and form the DE difference
    Y = X[jnp.arange(C)[:, None], idx_y, :]                # (C,W,D)
    Z = X[jnp.arange(C)[:, None], idx_z, :]                # (C,W,D)
    diff = Y - Z                                           # (C,W,D)

    # ---------- draw γ (symmetric) ----------
    # If gamma=None: γ ~ Normal(0, σ^2) with σ = 2.38/sqrt(2D)  (classic DE)
    # Else: use fixed |γ| but randomize sign to keep symmetry.
    key_gam, key_aux = random.split(key_gamma)
    if gamma is None:
        sigma = gamma_scale / jnp.sqrt(2.0 * D)
        gam = sigma * random.normal(key_gam, (C, W))       # (C,W)
    else:
        signs = jnp.where(random.uniform(key_gam, (C, W)) < 0.5, -1.0, 1.0)
        gam = signs * float(gamma)

    # ---------- crossover mask (C,W,D), ensure at least one True per (c,w) ----------
    if crossover_rate >= 1.0:
        mask = jnp.ones((C, W, D), dtype=bool)
    else:
        mask = random.bernoulli(key_aux, p=jnp.full((C, W, D), float(crossover_rate))).astype(bool)
        # force at least one dim if all False
        key_force = random.fold_in(key_aux, 2)
        force_j   = random.randint(key_force, (C, W), 0, D)   # (C,W)
        none_sel  = ~mask.any(axis=-1)                        # (C,W)
        mask = mask.at[jnp.arange(C)[:, None], jnp.arange(W)[None, :], force_j].set(
            jnp.where(none_sel, True, mask[jnp.arange(C)[:, None], jnp.arange(W)[None, :], force_j])
        )

    step = gam[..., None] * diff                             # (C,W,D)
    step = jnp.where(mask, step, 0.0)

    # ---------- jitter ε (optional; isotropic) ----------
    if jitter_scale > 0.0:
        key_eps = random.fold_in(key_aux, 3)
        eps = jitter_scale * random.normal(key_eps, (C, W, D))
    else:
        eps = jnp.zeros_like(X)

    # no-op where insufficient partners
    X_prop = X + jnp.where(has_pair[..., None], step, 0.0) + eps

    # ---------- proposal-density correction ----------
    # Translation + symmetric γ (+ symmetric ε) => symmetric kernel => Δlog q = 0
    # so no need to correct log q

    return X_prop, idx_y, idx_z, has_pair, mask

# prop_de, idx_y, idx_z, has_pair, mask = _propose_de_two_point(
#     k_partner, k_gamma, Xth, z=z_at_prop,
#     same_z_required=True, gamma=None, gamma_scale=2.38,
#     crossover_rate=0.9, jitter_scale=1e-6
# )

# # ... compute prop_lp ...
# log_alpha = (prop_lp - lp) / temperatures[:, None] + de_logq  # de_logq == 0 here


# -------- stretch restricted to same z (self-avoiding; safe fallback when alone) --------
def _propose_stretch_same_z(key_partner, key_scale, X, a: float = 2.0):
    """
    Goodman–Weare stretch where each (c,w) picks a partner among walkers at the
    same chain 'c' that share the same model label 'z[c,w]'. Self-avoiding.
    If no partner exists (group size == 1), returns (no-op, logJ=0, has_partner=False).
    Args:
      X: (C,W,D), z: (C,W)
    Returns:
      X_prop: (C,W,D), logJ: (C,W), has_partner: (C,W) bool
    """
    C, W, D = X.shape
    arW = jnp.arange(W)

    # eligible partners mask per (c,w): same z and not self
    same_z   = (z[:, :, None] == z[:, None, :])                      # (C,W,W)
    not_self = (arW[None, :, None] != arW[None, None, :])            # (1,W,W)
    elig     = same_z & not_self                                     # (C,W,W)

    has_partner = elig.any(axis=-1)                                  # (C,W)

    # masked categorical via Gumbel-max: logits 0 for eligible, -inf for others
    logits  = jnp.where(elig, 0.0, -1e9)
    g       = jax.random.gumbel(key_partner, logits.shape)
    partner = jnp.argmax(logits + g, axis=-1)                        # (C,W)

    # fallback to self if none (we'll no-op those later)
    partner = jnp.where(has_partner, partner, arW[None, :])
    Y = X[jnp.arange(C)[:, None], partner, :]                        # (C,W,D)

    # draw zfac ~ g(z) ∝ 1/sqrt(z) on [1/a, a]
    u   = jax.random.uniform(key_scale, (C, W))
    sa  = jnp.sqrt(a)
    zcf = (u * (sa - 1.0 / sa) + 1.0 / sa) ** 2

    X_prop = jnp.where(has_partner[..., None], Y + zcf[..., None] * (X - Y), X)
    logJ   = jnp.where(has_partner, (D - 1) * jnp.log(zcf), 0.0)
    return X_prop, logJ, has_partner