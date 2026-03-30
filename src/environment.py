"""
environment.py
==============
SIR epidemic model as a Partially Observable Markov Decision Process (POMDP).

State space  : s = (S, I, R) ∈ [0,1]^3  (fractions of population)
Action space : a ∈ {0, 1, 2}
                 0 → do nothing       (β unchanged)
                 1 → test / surveil   (reduces observation noise σ)
                 2 → lockdown         (reduces β by lockdown_factor)

Observations : o ~ Normal(ρ · I, σ_obs)   noisy hospitalisation rates (normalized)
               ρ  = ascertainment rate (what fraction we detect)
               N  = population size

The agent never sees (S, I, R) directly — only the noisy observation o.
"""

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

class EpiParams(NamedTuple):
    beta_0: float = 0.35       # baseline transmission rate
    gamma: float = 0.10        # recovery rate  (1/γ ≈ 10 days)
    rho: float = 0.15          # ascertainment rate
    N: int = 100_000           # population size
    lockdown_factor: float = 0.50   # β multiplier under lockdown
    noise_reduction: float = 0.50   # σ multiplier under surveillance
    dt: float = 1.0            # time step (days)
    sigma_stoch: float = 0.02  # SDE diffusion coefficient


DEFAULT_PARAMS = EpiParams()


# ---------------------------------------------------------------------------
# Stochastic SIR dynamics  (Euler–Maruyama discretisation)
# ---------------------------------------------------------------------------

def sir_step(state: jnp.ndarray,
             action: int,
             params: EpiParams,
             key: jax.Array) -> jnp.ndarray:
    """
    One Euler–Maruyama step of the stochastic SIR model.

    ds/dt = -β(a) S I  +  σ dW_1
    dI/dt =  β(a) S I  - γ I  +  σ dW_2
    dR/dt =  γ I       +  σ dW_3

    Parameters
    ----------
    state  : jnp.ndarray shape (3,)  — (S, I, R) fractions
    action : int  — 0 (nothing), 1 (test), 2 (lockdown)
    params : EpiParams
    key    : JAX PRNGKey

    Returns
    -------
    new_state : jnp.ndarray shape (3,)
    """
    S, I, R = state

    # Effective transmission rate depends on action
    beta = jnp.where(action == 2,
                     params.beta_0 * params.lockdown_factor,
                     params.beta_0)

    # Deterministic drift
    dS = -beta * S * I * params.dt
    dI = (beta * S * I - params.gamma * I) * params.dt
    dR = params.gamma * I * params.dt

    # Stochastic diffusion  (Wiener increments ~ N(0, dt))
    k1, k2, k3 = random.split(key, 3)
    noise_scale = params.sigma_stoch * jnp.sqrt(params.dt)
    dW = jnp.array([
        random.normal(k1) * noise_scale,
        random.normal(k2) * noise_scale,
        random.normal(k3) * noise_scale,
    ])

    new_state = state + jnp.array([dS, dI, dR]) + dW

    # Project back to the simplex  [0,1]^3,  S+I+R ≈ 1
    new_state = jnp.clip(new_state, 0.0, 1.0)
    new_state = new_state / new_state.sum()

    return new_state


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

def observe(state: jnp.ndarray,
            action: int,
            params: EpiParams,
            key: jax.Array) -> jnp.ndarray:
    """
    Generate a noisy hospitalisation-rate observation.

    Under action=1 (surveillance), ascertainment rate doubles.

    Raw count: c ~ Poisson(ρ_eff · I · N)
    Normalized observation: o = c / N ~ approximately Normal(ρ_eff · I, σ_obs)

    Returns
    -------
    obs : jnp.ndarray shape (1,)  — observed infection rate (float for JAX compat)
    """
    _, I, _ = state

    rho_eff = jnp.where(action == 1,
                        jnp.minimum(params.rho / params.noise_reduction, 1.0),
                        params.rho)

    # Generate raw count
    lam = rho_eff * I * params.N
    count = random.poisson(key, lam=lam, shape=(1,)).astype(jnp.float32)

    # Normalize to rate
    obs_rate = count[0] / params.N
    return jnp.array([obs_rate])


# ---------------------------------------------------------------------------
# Log-likelihood of observation given belief
# ---------------------------------------------------------------------------

def log_likelihood(obs: jnp.ndarray,
                   I_belief: float,
                   action: int,
                   params: EpiParams) -> float:
    """
    log p(o | I, a) under the Gaussian observation model on normalized rates.

    Used in the variational update to weight the posterior.
    """
    rho_eff = jnp.where(action == 1,
                        jnp.minimum(params.rho / params.noise_reduction, 1.0),
                        params.rho)
    mu_obs = rho_eff * I_belief
    sigma_obs = jnp.sqrt(mu_obs / params.N + 1e-8)

    import jax.scipy.stats as jss
    return jss.norm.logpdf(obs[0], loc=mu_obs, scale=sigma_obs)


# ---------------------------------------------------------------------------
# Reward / preference model
# ---------------------------------------------------------------------------

def preference_log_prob(state: jnp.ndarray,
                        action: int,
                        params: EpiParams) -> float:
    """
    log P̃(o, s)  — the agent's prior preference over outcomes.

    We encode two goals:
      1. Keep infections low       → penalise I heavily
      2. Avoid costly lockdowns    → penalise action=2 lightly

    This is the 'C' matrix in standard Active Inference notation.
    """
    _, I, _ = state
    infection_cost = -10.0 * I           # prefer I → 0
    action_cost = jnp.where(action == 2, -0.5, 0.0)  # lockdown is expensive
    return infection_cost + action_cost
