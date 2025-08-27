
"""Proposal utilities for parallel tempering samplers."""
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


