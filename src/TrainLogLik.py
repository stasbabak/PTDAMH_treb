### Training NN to predict log-likelihood

import numpy as np
from typing import Sequence
import jax, jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state
from functools import partial
from tqdm.auto import tqdm
import pickle

def build_features(
    X, y, *,
    periodic_idx=(),
    period=1.4,
    add_invT=False, T=None,
):
    X = np.asarray(X); y = np.asarray(y)
    D = X.shape[1]
    mask = np.ones(D, dtype=bool)
    mask[list(periodic_idx)] = False

    X_lin = X[:, mask]
    if len(periodic_idx):
        X_per = X[:, list(periodic_idx)]
        Xs = np.sin(2*np.pi*X_per/period)
        Xc = np.cos(2*np.pi*X_per/period)
        parts = [X_lin, Xs, Xc]
    else:
        parts = [X_lin]

    if add_invT and T is not None:
        parts.append((1.0/np.maximum(T, 1e-12))[:, None])

    X_feat = np.concatenate(parts, axis=1).astype(np.float32)
    # print ('debug, ', X_feat.shape)
    # print ('debug, mean:', X_feat.mean(0))

    x_mean, x_std = X_feat.mean(0), X_feat.std(0) + 1e-8
    Xn = (X_feat - x_mean) / x_std

    y_mean, y_std = y.mean(), y.std() + 1e-8
    yn = ((y - y_mean) / y_std).astype(np.float32)

    meta = dict(
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        D_raw=D, periodic_idx=np.array(periodic_idx), period=period,
        add_invT=add_invT
    )
    return Xn, yn, meta



# class ResBlock(nn.Module):
#     width: int
#     dropout: float = 0.0
#     @nn.compact
#     def __call__(self, x, train: bool):
#         h = nn.Dense(self.width)(x)  # Moved here
#         h = nn.LayerNorm()(h)
#         h = nn.gelu(h)
#         if self.dropout > 0:
#             h = nn.Dropout(self.dropout)(h, deterministic=not train)
#         h = nn.Dense(self.width)(h)
#         return x + h

# class LogProbNet(nn.Module):
#     widths: Sequence[int] = (512, 512)
#     blocks_per_layer: int = 1
#     dropout: float = 0.1
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
#             mu, log_var = out[...,0], nn.softplus(out[...,1]) + 1e-6
#             return mu, log_var
#         else:
#             return nn.Dense(1)(h).squeeze(-1)

class ResBlock(nn.Module):
    width: int
    dropout: float = 0.0
    @nn.compact
    def __call__(self, x, train: bool): 
        h = nn.LayerNorm()(x)
        h = nn.gelu(h)
        h = nn.Dense(self.width)(h)  # Moved here
        
        if self.dropout > 0:
            h = nn.Dropout(self.dropout)(h, deterministic=not train)
        h = nn.gelu(h)
        h = nn.Dense(self.width)(h)
        if x.shape[-1] != self.width:
            x = nn.Dense(self.width, use_bias=False, name="skip_proj")(x)

        return x + h

class LogProbNet(nn.Module):
    widths: Sequence[int] = (512, 512, 512, 512)   # Base
    blocks_per_layer: int = 2
    dropout: float = 0.0
    heteroscedastic: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
    
        h = nn.Dense(self.widths[0])(x)
        for w in self.widths:
            for _ in range(self.blocks_per_layer):
                h = ResBlock(w, dropout=self.dropout)(h, train=train)
        h = nn.LayerNorm()(h)

        if self.heteroscedastic:
            out = nn.Dense(2)(h)
            mu, log_var = out[...,0], jnp.clip(out[..., 1], -12.0, 5.0)
            return mu, log_var
        else:
            return nn.Dense(1)(h).squeeze(-1)
        
# alternative is widths=(768, 768, 768, 768, 768), dropout=0.1




### Data loader

def split_train_val(X, y, val_frac=0.1, seed=0, block=False):
    N = len(X)
    if block:
        n_val = int(N*val_frac)
        return (X[:-n_val], y[:-n_val]), (X[-n_val:], y[-n_val:])
    rng = np.random.default_rng(seed)
    idx = np.arange(N); rng.shuffle(idx)
    n_val = int(N*val_frac)
    return (X[idx[n_val:]], y[idx[n_val:]]), (X[idx[:n_val]], y[idx[:n_val]])

def data_loader(X, y, batch_size, shuffle=True, seed=0):
    N = len(X)
    order = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed); rng.shuffle(order)
    for i in range(0, N, batch_size):
        idx = order[i:i+batch_size]
        yield X[idx], y[idx]



### Training




class TrainState(train_state.TrainState):
    pass


# @partial(jax.jit, static_argnames=("heteroscedastic",))
# def _grad_step(params, apply_fn, x, y, heteroscedastic):
#     def loss_fn(p):
#         out = apply_fn({'params': p}, x, train=True)
#         if heteroscedastic:
#             mu, logv = out
#             return jnp.mean(0.5 * ((y - mu)**2 * jnp.exp(-logv) + logv))
#         else:
#             pred = out
#             return jnp.mean((y - pred)**2)
#     return jax.grad(loss_fn)(params)


# @jax.jit
# def _apply_updates(params, opt_state, tx, grads):
#     updates, opt_state = tx.update(grads, opt_state, params)
#     return optax.apply_updates(params, updates), opt_state



# def make_optimizer(lr, weight_decay, steps, clip_norm=1.0):
#     sched = optax.warmup_cosine_decay_schedule(
#         init_value=0.0, peak_value=lr,
#         warmup_steps=int(0.05*steps),
#         decay_steps=steps, end_value=0.05*lr
#     )
#     tx = optax.chain(
#         optax.clip_by_global_norm(clip_norm),
#         optax.adamw(sched, weight_decay=weight_decay),
#     )
#     return tx

def make_optimizer(total_steps, *, peak_lr=3e-4, end_lr=1e-5,
                   warmup_frac=0.05, weight_decay=1e-4, clip_norm=1.0):
    warmup = int(total_steps * warmup_frac)
    decay  = max(1, total_steps - warmup)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup,
        decay_steps=decay,
        end_value=end_lr,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=weight_decay),
    )
    return tx, schedule

@partial(jax.jit, static_argnames=("heteroscedastic",))
def train_step(state, x, y, heteroscedastic: bool):
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, x, train=True)
        if heteroscedastic:
            mu, log_var = pred
            inv_var = jnp.exp(-log_var)
            loss = jnp.mean(0.5 * ((y - mu)**2 * inv_var + log_var))
        else:
            loss = jnp.mean((y - pred)**2)
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


def memmap_loader(Xmm, ymm, batch, wmm=None, shuffle=True, seed=0):
    N = len(Xmm)
    order = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed); rng.shuffle(order)
    for i in range(0, N, batch):
        idx = order[i:i+batch]
        Xb = np.asarray(Xmm[idx])
        yb = np.asarray(ymm[idx])
        if wmm is None:
            wb = np.ones((len(idx),), dtype=np.float32)
        else:
            wb = np.asarray(wmm[idx], dtype=np.float32)
        yield Xb, yb, wb

############## Main driver ##############

def train_with_accum(
    model, params, tx, Xtr, ytr, Xva, yva, *,
    Wtr=None, Wva=None, ## weights
    heteroscedastic=False, batch=8192, microbatch=1024,
    epochs=150, seed=0, patience=20,
    val_weighted=False
):
    opt_state = tx.init(params)
    best_loss, best_params, bad = np.inf, params, 0

    # ---------- loaders ----------
    def epoch_loader(X, Y, B, W=None, shuffle=True, sd=0):
        N = len(X)
        order = np.arange(N)
        if shuffle:
            rng = np.random.default_rng(sd); rng.shuffle(order)
        for i in range(0, N, B):
            idx = order[i:i+B]
            Xb = np.asarray(X[idx])
            Yb = np.asarray(Y[idx])
            if W is None:
                Wb = np.ones((len(idx),), dtype=np.float32)
            else:
                Wb = np.asarray(W[idx], dtype=np.float32)
            yield Xb, Yb, Wb

    # ---------- jitted inner steps (CLOSE OVER model.apply and tx) ----------
    @partial(jax.jit, static_argnames=("heteroscedastic",))
    def _grad_step(params, x, y, w, heteroscedastic, drng):
        alpha = 2.0  # hard-coded alpha value
        def loss_fn(p):
            out = model.apply({'params': p}, x, train=True, rngs={'dropout': drng})
            if heteroscedastic:
                mu, logv = out

                invv = jnp.exp(-logv)
                e = y - mu
                base = 0.5 * (e*e * invv + logv)    # (B,)
                # extra penalty ONLY on the error term when y>mu
                extra = (alpha - 1.0) * 0.5 * (e*e) * invv * (e > 0)
                per = base + extra
                return jnp.mean(w * per)
            else:
                pred = out

                e = y - pred
                # symmetric MSE + tilted extra on positive residuals
                base = e*e
                extra = (alpha - 1.0) * (e*e) * (e > 0)
                per = base + extra
                return jnp.mean(w * per)
        return jax.grad(loss_fn)(params)

    @jax.jit
    def _apply_updates(params, opt_state, grads):
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    @partial(jax.jit, static_argnames=("heteroscedastic", "weighted"))
    def _eval_step(params, x, y, w, heteroscedastic, weighted: bool):
        out = model.apply({'params': params}, x, train=False)
        if heteroscedastic:
            mu, logv = out
            # logv = jnp.clip(logv, -15.0, 6.0)
            per = 0.5 * ((y - mu)**2 * jnp.exp(-logv) + logv)
        else:
            per = (y - out)**2
        # weighted or unweighted aggregation
        return jnp.mean(w * per) if weighted else jnp.mean(per)
    
    summary = {
        "train_losses": [],
        "val_losses": [],
        "best_val_loss": np.inf,
    }

    base_key = jax.random.PRNGKey(seed)
    # ---------- training loop  ----------
    for ep in tqdm(range(epochs), desc="Training", leave=True):
        ep_key = jax.random.fold_in(base_key, ep)
        for big_batch_id, (big_x, big_y, big_w) in enumerate(epoch_loader(Xtr, ytr, batch, W=Wtr, shuffle=True, sd=seed+ep)):
            bb_key = jax.random.fold_in(ep_key, big_batch_id)
            grad_sum, count = None, 0
            # micro-batching (handles last partial chunk correctly)
            for i in range(0, len(big_x), microbatch):
                mb_key = jax.random.fold_in(bb_key, i)
                xb = jnp.asarray(big_x[i:i+microbatch])
                yb = jnp.asarray(big_y[i:i+microbatch])
                wb = jnp.asarray(big_w[i:i+microbatch])
                g = _grad_step(params, xb, yb, wb, heteroscedastic, mb_key)
                # g = _grad_step(params, xb, yb, heteroscedastic)
                grad_sum = g if grad_sum is None else jax.tree_util.tree_map(lambda a, b: a + b, grad_sum, g)
                count += 1
            grads = jax.tree_util.tree_map(lambda g: g / count, grad_sum)
            params, opt_state = _apply_updates(params, opt_state, grads)
            
        # ----- Metrics -----
        Xtr_eval, ytr_eval = jnp.asarray(Xtr), jnp.asarray(ytr)
        Xva_eval, yva_eval = jnp.asarray(Xva), jnp.asarray(yva)
        wtr_eval = jnp.ones((len(Xtr),), dtype=jnp.float32) if Wtr is None else jnp.asarray(Wtr)
        wva_eval = jnp.ones((len(Xva),), dtype=jnp.float32) if Wva is None else jnp.asarray(Wva)

        train_metric = float(_eval_step(params, Xtr_eval, ytr_eval, wtr_eval, heteroscedastic, False))
        val_metric   = float(_eval_step(params, Xva_eval, yva_eval, wva_eval, heteroscedastic, val_weighted))

    
        summary["train_losses"].append(train_metric)
        summary["val_losses"].append(val_metric)  

        
        # val_metric = float(_eval_step(params, jnp.asarray(Xva), jnp.asarray(yva), heteroscedastic))
        # train_metric = float(_eval_step(params, jnp.asarray(Xtr), jnp.asarray(ytr), heteroscedastic))
        # train_losses.append(train_metric)
        # val_losses.append(val_metric)   

        improved = val_metric + 1e-4 < best_loss
        if improved:
            best_loss, best_params, bad = val_metric, params, 0
        else:
            bad += 1

        tqdm.write(f"Epoch {ep+1}/{epochs} | Val loss: {val_metric:.4f} | Best: {best_loss:.4f} | Bad epochs: {bad}")
        if bad >= patience:
            tqdm.write("Early stopping.")
            break

    return best_params, summary


@partial(jax.jit, static_argnames=("heteroscedastic",))
def eval_step(params, apply_fn, x, y, heteroscedastic: bool):
    pred = apply_fn({'params': params}, x, train=False)
    if heteroscedastic:
        mu, log_var = pred
        rmse = jnp.sqrt(jnp.mean((y - mu)**2))
        nll  = jnp.mean(0.5 * ((y - mu)**2 * jnp.exp(-log_var) + log_var))
        return float(rmse), float(nll)
    else:
        rmse = jnp.sqrt(jnp.mean((y - pred)**2))
        return float(rmse), None

def train_logprob_net(
    Xn, yn, *,
    heteroscedastic=False,
    widths=(512,512,512,512), blocks=2, dropout=0.0,
    batch=4096, epochs=200, lr=3e-4, weight_decay=1e-4,
    val_frac=0.1, val_block=True, seed=0
):
    (Xtr, ytr), (Xva, yva) = split_train_val(Xn, yn, val_frac=val_frac, seed=seed, block=val_block)

    model = LogProbNet(widths=widths, blocks_per_layer=blocks,
                       dropout=dropout, heteroscedastic=heteroscedastic)
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, jnp.zeros((1, Xn.shape[1])), train=False)['params']

    steps_per_epoch = int(np.ceil(len(Xtr)/batch))
    total_steps = steps_per_epoch * epochs
    tx = make_optimizer(lr, weight_decay, total_steps)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    best_rmse, best_params, bad, patience = np.inf, params, 0, 20

    for ep in range(epochs):
        for xb, yb in data_loader(Xtr, ytr, batch, shuffle=True, seed=seed+ep):
            state = train_step(state, jnp.asarray(xb), jnp.asarray(yb), heteroscedastic)

        rmse, _ = eval_step(state.params, state.apply_fn, jnp.asarray(Xva), jnp.asarray(yva), heteroscedastic)
        if rmse + 1e-4 < best_rmse:
            best_rmse, best_params, bad = rmse, state.params, 0
        else:
            bad += 1
        if bad >= patience:
            break

    state = state.replace(params=best_params)
    return model, state, {"val_rmse_norm": best_rmse}



def _featurize_batch(X_raw, *, meta, T=None):
    """Same feature pipeline used at training: linear + sin/cos(periodic) + optional 1/T, then standardize."""
    X_raw = np.asarray(X_raw)
    D = meta["D_raw"]
    periodic_idx = np.asarray(meta["periodic_idx"])
    period = float(meta["period"])
    add_invT = bool(meta.get("add_invT", False))

    mask = np.ones(D, dtype=bool)
    if periodic_idx.size:
        mask[periodic_idx] = False

    X_lin = X_raw[:, mask]
    parts = [X_lin]

    if periodic_idx.size:
        Xp = X_raw[:, periodic_idx]
        parts += [np.sin(2*np.pi*Xp/period), np.cos(2*np.pi*Xp/period)]

    if add_invT:
        if T is None:
            raise ValueError("meta.add_invT=True but no T provided")
        parts.append((1.0/np.maximum(np.asarray(T), 1e-12))[:, None])

    X_feat = np.concatenate(parts, axis=1).astype(np.float32)
    Xn = (X_feat - np.asarray(meta["x_mean"])) / np.asarray(meta["x_std"])
    return Xn.astype(np.float32)

def load_surrogate(pkl_path):
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    params = saved["params"]
    meta   = saved["meta"]
    cfg    = meta["model_cfg"]

    # Recreate the model with the same architecture
    model = LogProbNet(**cfg)

    # JITed batched apply
    @jax.jit
    def _apply_batched(x_batched):
        return model.apply({"params": params}, x_batched, train=False)

    def predict(theta, T=None, return_var=False):
        """
        theta: (D_raw,) or (N, D_raw) raw parameters (pre-feature)
        T    : scalar or (N,) if meta['add_invT'] is True; else ignored
        Returns unnormalized log-prob (de-standardized).
        If heteroscedastic=True, set return_var=True to also get variance.
        """
        single = (np.asarray(theta).ndim == 1)
        if single:
            Xn = _featurize_batch(np.asarray(theta)[None, :], meta=meta, T=T)
        else:
            if meta.get("add_invT", False):
                if T is None:
                    raise ValueError("meta.add_invT=True but T is None for batch.")
                if np.isscalar(T):
                    T = np.full((np.asarray(theta).shape[0],), T, dtype=np.float32)
                elif len(T) != np.asarray(theta).shape[0]:
                    raise ValueError("T must be scalar or shape (N,).")
            Xn = _featurize_batch(theta, meta=meta, T=T)

        out = _apply_batched(jnp.asarray(Xn))
        y_mean = float(meta["y_mean"]); y_std = float(meta["y_std"])
        y_max = float(meta.get("y_max", 0.0)) # Get the shift, default to 0 if not saved

        if cfg.get("heteroscedastic", False):
            mu, logv = out
            # De-standardize and then add back the shift
            mu = np.array(mu) * y_std + y_mean + y_max
            var = np.exp(np.array(logv)) * (y_std**2)
            if single:
                mu, var = mu[0], var[0]
            return (mu, var) if return_var else mu
        else:
            # De-standardize and then add back the shift
            mu = np.array(out) * y_std + y_mean + y_max
            if single: mu = mu[0]
            return mu

    return predict, meta, params




### after training

# def make_surrogate_logprob_fn(model, params, meta, Xn_train=None):
#     mu_f, inv_cov = None, None
#     if Xn_train is not None and len(Xn_train) > 1000:
#         mu_f = Xn_train.mean(0)
#         cov  = np.cov(Xn_train, rowvar=False) + 1e-3*np.eye(Xn_train.shape[1])
#         inv_cov = np.linalg.inv(cov)

#     def _featurize(theta, T=None):
#         theta = np.asarray(theta)[None, :]
#         D = meta["D_raw"]; mask = np.ones(D, bool); mask[meta["periodic_idx"]] = False
#         X_lin = theta[:, mask]
#         parts = [X_lin]
#         if meta["periodic_idx"].size:
#             thp = theta[:, meta["periodic_idx"]]
#             parts += [np.sin(2*np.pi*thp/meta["period"]), np.cos(2*np.pi*thp/meta["period"])]
#         if meta.get("add_invT", False) and T is not None:
#             parts.append(np.array([[1.0/max(T, 1e-12)]]))
#         x = np.concatenate(parts, axis=1)
#         x = (x - meta["x_mean"]) / meta["x_std"]
#         return x.astype(np.float32)

#     @jax.jit
#     def _apply(x_batched):
#         return model.apply({'params': params}, x_batched, train=False)

#     def logprob(theta, T=None, return_ood=False):
#         x = _featurize(theta, T)
#         pred = _apply(jnp.asarray(x))
#         if isinstance(pred, tuple):  # heteroscedastic
#             pred = pred[0]
#         y_norm = np.array(pred)[0]
#         y = y_norm * meta["y_std"] + meta["y_mean"]
#         if return_ood and mu_f is not None:
#             d = x - mu_f
#             ood = float(d @ inv_cov @ d.T)  # Mahalanobis in feature space
#             return y, ood
#         return y

#     return logprob

# # 1) Get dataset from your sampler
# ds = run_info["dataset"]
# X_raw = ds["X"]                # (N, 24)
# y     = ds["y_logprob"]        # (N,)
# T     = ds["temperature"]      # (N,) if you ever need it

# # 2) Build features
# Xn, yn, meta = build_features(
#     X_raw, y, periodic_idx=fold_idx, period=1.0, add_invT=False, T=None
# )

# # 3) Train (Base)
# model, state, summ = train_logprob_net(
#     Xn, yn,
#     heteroscedastic=False,
#     widths=(512,512,512,512), blocks=2, dropout=0.0,
#     batch=4096, epochs=250, lr=3e-4, weight_decay=1e-4,
#     val_frac=0.1, val_block=True, seed=0
# )
# print("best normed val RMSE:", summ["val_rmse_norm"])

# # 4) Surrogate fn
# surrogate = make_surrogate_logprob_fn(model, state.params, meta, Xn_train=Xn)

# # 5) Use inside PT-(D)AMH stage-1 with OOD fallback if you like
# lp_pred, ood = surrogate(theta, return_ood=True)


