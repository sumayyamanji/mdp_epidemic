"""
inference.py
============
Variational inference engine — minimises Free Energy F w.r.t. Q(s).

The update loop:

  1.  PREDICT  : propagate Q(s_{t-1}) through the transition prior
                 → get P(s_t | s_{t-1}, a_{t-1})

  2.  UPDATE   : given new observation o_t, minimise
                     F(Q) = KL[Q ‖ P] - E_Q[log P(o|s,a)]
                 w.r.t. Q's parameters (μ, log σ²) via gradient descent.

This is the perception / state-estimation half of Active Inference.
The action-selection half lives in agent.py.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from functools import partial
from typing import Tuple

from environment import EpiParams, DEFAULT_PARAMS
from generative_model import (
    BeliefState,
    transition_prior,
    free_energy,
)


# ---------------------------------------------------------------------------
# Gradient-descent update on (μ, log σ²)
# ---------------------------------------------------------------------------

def _F_params(mu: jnp.ndarray,
              log_var: jnp.ndarray,
              prior: BeliefState,
              obs: jnp.ndarray,
              action: int,
              params: EpiParams) -> float:
    """Free energy as a function of raw variational parameters."""
    posterior = BeliefState(mu=mu, log_var=log_var)
    return free_energy(posterior, prior, obs, action, params)


_grad_F = jit(value_and_grad(_F_params, argnums=(0, 1)))


def update_belief(prior: BeliefState,
                  obs: jnp.ndarray,
                  action: int,
                  params: EpiParams = DEFAULT_PARAMS,
                  n_steps: int = 60,
                  lr_mu: float = 1e-5,
                  lr_logvar: float = 5e-6) -> Tuple[BeliefState, jnp.ndarray]:
    """
    Minimise F(Q) via gradient descent to obtain the posterior Q*(s_t).

    Initialise Q = P (posterior starts at the prior) then iterate:

        μ       ←  μ       - lr_μ      · ∂F/∂μ
        log σ²  ←  log σ²  - lr_{logσ²}· ∂F/∂(log σ²)

    Parameters
    ----------
    prior    : BeliefState  — prediction from transition_prior()
    obs      : jnp.ndarray  — current observation o_t
    action   : int
    params   : EpiParams
    n_steps  : int          — number of gradient steps
    lr_mu    : float        — learning rate for mean
    lr_logvar: float        — learning rate for log-variance

    Returns
    -------
    posterior : BeliefState — Q*(s_t) after minimising F
    F_history : jnp.ndarray — free energy at each step (for diagnostics)
    """
    mu      = prior.mu.copy()
    log_var = prior.log_var.copy()
    F_history = []

    for _ in range(n_steps):
        F_val, (g_mu, g_logvar) = _grad_F(mu, log_var, prior,
                                           obs, action, params)
        mu      = mu      - lr_mu      * g_mu
        log_var = log_var - lr_logvar  * g_logvar

        # Constrain log_var to avoid degenerate posteriors
        log_var = jnp.clip(log_var, -10.0, 0.0)
        # Constrain mu to valid probability range
        mu = jnp.clip(mu, 1e-6, 1.0 - 1e-6)

        F_history.append(float(F_val))

    posterior = BeliefState(mu=mu, log_var=log_var)
    return posterior, jnp.array(F_history)


# ---------------------------------------------------------------------------
# Full perception step: predict → update
# ---------------------------------------------------------------------------

def perception_step(belief: BeliefState,
                    obs: jnp.ndarray,
                    action: int,
                    params: EpiParams = DEFAULT_PARAMS,
                    n_steps: int = 40) -> Tuple[BeliefState, jnp.ndarray]:
    """
    Combined prediction + update: the perception cycle of Active Inference.

    Parameters
    ----------
    belief   : BeliefState  — Q(s_{t-1})
    obs      : jnp.ndarray  — observation at time t
    action   : int          — action taken at t-1 → t
    params   : EpiParams

    Returns
    -------
    posterior : BeliefState — Q(s_t)
    F_history : jnp.ndarray — convergence trace of F
    """
    # Step 1: Predict
    prior_t = transition_prior(belief, action, params)

    # Step 2: Update (minimise F)
    posterior, F_history = update_belief(prior_t, obs, action, params, n_steps)

    return posterior, F_history
