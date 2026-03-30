"""
generative_model.py
===================
The agent's internal generative model — its beliefs about how the world works.

This is distinct from the true environment (environment.py). In practice they
share the same functional form but the agent fits its own parameters.

The generative model defines:
  P(o_t | s_t, a_t)   — likelihood     (observation model)
  P(s_t | s_{t-1}, a) — transition     (dynamics prior)
  P(s_0)               — initial prior

The agent represents its posterior belief as a Gaussian:

  Q(s_t) = N(μ_t, Σ_t)

where s_t = (S_t, I_t, R_t).  Since S+I+R=1 we work in the 2D subspace
(S, I), recovering R = 1 - S - I.

All functions are pure (no side effects) and JAX-traceable.
"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
from jax import random
from typing import NamedTuple

from environment import EpiParams, DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

class BeliefState(NamedTuple):
    """
    Gaussian belief over (S, I) — the variational posterior Q(s).

    mu    : jnp.ndarray shape (2,)    — mean  [μ_S, μ_I]
    log_var: jnp.ndarray shape (2,)   — log-variance (ensures positivity)
    """
    mu: jnp.ndarray
    log_var: jnp.ndarray

    @property
    def var(self):
        return jnp.exp(self.log_var)

    @property
    def std(self):
        return jnp.exp(0.5 * self.log_var)

    def to_full_state(self) -> jnp.ndarray:
        """Recover (S, I, R) from the 2D belief."""
        S, I = self.mu
        R = jnp.clip(1.0 - S - I, 0.0, 1.0)
        return jnp.array([S, I, R])


def init_belief(I_init: float = 0.01,
                uncertainty: float = 0.05) -> BeliefState:
    """
    Initialise prior belief: most people susceptible, small infected fraction.
    High uncertainty (we don't know the true state at t=0).
    """
    S_init = 1.0 - I_init
    mu = jnp.array([S_init, I_init])
    log_var = jnp.log(jnp.array([uncertainty ** 2, uncertainty ** 2]))
    return BeliefState(mu=mu, log_var=log_var)


# ---------------------------------------------------------------------------
# Transition prior  P(s_t | s_{t-1}, a)
# ---------------------------------------------------------------------------

def transition_prior(belief: BeliefState,
                     action: int,
                     params: EpiParams = DEFAULT_PARAMS
                     ) -> BeliefState:
    """
    Propagate the belief through the SIR dynamics (deterministic part only).

    This is the Prediction Step in the variational filtering loop:

        μ_{t|t-1}  =  f(μ_{t-1}, a)          (SIR drift)
        Σ_{t|t-1}  =  Σ_{t-1} + Q             (process noise Q = σ² I)

    We linearise the SIR dynamics around the current mean (Extended Kalman
    style) but keep it simple: propagate the mean through the nonlinear map
    and add process noise to the variance.

    Parameters
    ----------
    belief : BeliefState  — Q(s_{t-1})
    action : int
    params : EpiParams

    Returns
    -------
    predicted_belief : BeliefState  — P(s_t | s_{t-1}, a)  (prior at t)
    """
    S, I = belief.mu

    beta = jnp.where(action == 2,
                     params.beta_0 * params.lockdown_factor,
                     params.beta_0)

    # Deterministic SIR drift
    dS = -beta * S * I * params.dt
    dI = (beta * S * I - params.gamma * I) * params.dt

    mu_pred = jnp.array([S + dS, I + dI])
    mu_pred = jnp.clip(mu_pred, 1e-6, 1.0 - 1e-6)

    # Inflate variance with process noise
    process_noise = params.sigma_stoch ** 2 * params.dt
    log_var_pred = jnp.log(belief.var + process_noise)

    return BeliefState(mu=mu_pred, log_var=log_var_pred)


# ---------------------------------------------------------------------------
# Observation likelihood  log P(o | s, a)
# ---------------------------------------------------------------------------

def log_obs_likelihood(obs: jnp.ndarray,
                       belief: BeliefState,
                       action: int,
                       params: EpiParams = DEFAULT_PARAMS) -> float:
    """
    E_{Q(s)}[ log P(o | s, a) ]

    We evaluate the expected log-likelihood under the current belief,
    approximated by plugging in the mean (delta approximation).

    Observations are normalized rates: o ~ Normal(ρ(a) * I, σ_obs)
    where σ_obs is chosen to match the Poisson variance.

    For a Gaussian likelihood:
        log P(o | I, a) = -½ log(2π σ²) - (o - μ)²/(2σ²)
        where μ = ρ_eff * I, σ² ≈ ρ_eff * I / N (Poisson variance normalized)
    """
    I = belief.mu[1]
    rho_eff = jnp.where(action == 1,
                        jnp.minimum(params.rho / params.noise_reduction, 1.0),
                        params.rho)

    # Mean of observation
    mu_obs = rho_eff * I

    # Variance approximation: Poisson variance is λ = ρ*I*N, normalized by N² gives ρ*I/N
    # Use a reasonable observation noise
    sigma_obs = jnp.sqrt(mu_obs / params.N + 1e-6)  # Add small constant for stability

    o = obs[0]
    return jss.norm.logpdf(o, loc=mu_obs, scale=sigma_obs)


# ---------------------------------------------------------------------------
# KL divergence  KL[ Q(s) ‖ P(s) ]
# ---------------------------------------------------------------------------

def kl_divergence(posterior: BeliefState,
                  prior: BeliefState) -> float:
    """
    Analytic KL divergence between two diagonal Gaussians:

        KL[ N(μ_Q, Σ_Q) ‖ N(μ_P, Σ_P) ]
        = ½ [ tr(Σ_P⁻¹ Σ_Q)
              + (μ_P - μ_Q)ᵀ Σ_P⁻¹ (μ_P - μ_Q)
              - d
              + log(|Σ_P| / |Σ_Q|) ]

    For diagonal covariances this simplifies to a sum over dimensions.
    """
    d = posterior.mu.shape[0]

    var_Q = posterior.var
    var_P = prior.var
    mu_diff = prior.mu - posterior.mu

    kl = 0.5 * jnp.sum(
        var_Q / var_P
        + mu_diff ** 2 / var_P
        - 1.0
        + jnp.log(var_P / var_Q)
    )
    return kl


# ---------------------------------------------------------------------------
# Variational Free Energy  F
# ---------------------------------------------------------------------------

def free_energy(posterior: BeliefState,
                prior: BeliefState,
                obs: jnp.ndarray,
                action: int,
                params: EpiParams = DEFAULT_PARAMS) -> float:
    """
    Variational Free Energy:

        F = KL[ Q(s) ‖ P(s) ] - E_{Q(s)}[ log P(o | s, a) ]

    Minimising F w.r.t. Q(s):
      → KL term pulls the posterior towards the prior (regularisation)
      → likelihood term pulls the posterior towards the data

    The negative ELBO is:  -ELBO = F  (we minimise F = maximise ELBO)

    Returns
    -------
    F : float  (lower is better)
    """
    kl = kl_divergence(posterior, prior)
    ell = log_obs_likelihood(obs, posterior, action, params)
    return kl - ell
