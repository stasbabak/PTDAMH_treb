# PTDAMH_treb

Parallel Tempering Delayed Acceptance MCMC (PT-DAMH) with neural network
surrogate models. The project demonstrates how parallel tempering
coupled with a learned log-likelihood approximation can accelerate Bayesian
inference for expensive simulators.

## Installation prerequisites

The codebase is written in Python and relies on [JAX](https://github.com/google/jax)
for efficient numerical computation. A typical environment requires:

- Python 3.9 or later
- `jax` and `jaxlib` (CPU or GPU build)
- `numpy`
- `flax` and `optax` for neural networks
- `tqdm` for progress bars

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourname/PTDAMH_treb.git
   cd PTDAMH_treb
   ```

2. Create and activate a virtual environment (optional but recommended).

3. Install dependencies:

   ```bash
   pip install jax jaxlib numpy flax optax tqdm
   ```

## Example usage

Train a neural network surrogate for the log-likelihood:

```bash
python src/TrainLogLik.py
```

Then run parallel tempering with the trained surrogate (see notebooks for
complete examples):

```python
from src.PTDAMH import temperature_ladder, run_adaptive_pt_device_fast

# Define your log-probability function and initial state here
T = temperature_ladder(n_temps=16, T_max=100.0)
state = run_adaptive_pt_device_fast(init_state, log_prob_fn, T)
```

Interactive demonstrations and experiments are available in the
`notebooks/` directory.

## Repository structure

- `src/` – core PT-DAMH algorithm, proposal distributions, and NN training
- `models/` – pre-trained neural network surrogates and example datasets
- `notebooks/` – Jupyter notebooks illustrating training and sampling workflows

## License

This project is released under the [MIT License](LICENSE).

## References

- C. J. Geyer. *Markov chain Monte Carlo maximum likelihood*, 1991.
- D. J. Earl and M. W. Deem. *Parallel tempering: Theory, applications, and new perspectives*, 2005.
- J. A. Christen and C. Fox. *Markov chain Monte Carlo using an approximation*, 2005.
