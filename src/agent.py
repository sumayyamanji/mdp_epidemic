"""
agent.py
========
Active Inference agent — action selection via Expected Free Energy (G).

The key insight of Active Inference is that perception and action share the
same objective: minimise (Expected) Free Energy.

    Perception  : minimise F  = KL[Q‖P] - E_Q[log P(o|s)]
    Action      : minimise G  = E_{Q(s,o'|a)}[ log Q(s') - log P̃(o',s') ]

where P̃(o', s') encodes the agent's *preferences* (desired outcomes).

Crucially, G uses a tractable approximation that captures:

    G(a) = Epistemic value  +  Pragmatic value
         = -H[P(o'|a, Q)]  +  KL[ Q(s'|a) ‖ P̃(s') ]

    Epistemic  (exploration) : actions that reduce uncertainty about the
                               hidden state → drives testing/surveillance
    Pragmatic  (exploitation): actions that achieve preferred outcomes →
                               drives lockdown when I is high

This approximation uses entropy instead of full information gain, and state
preferences rather than outcome preferences. The decomposition falls naturally
from the mathematics — the agent doesn't need a hand-crafted exploration bonus.
Uncertainty reduction is intrinsically motivated by the Free Energy principle.

References
----------
Friston et al. (2017). Active inference and epistemic value. Cognitive Neuroscience.
Parr & Friston (2019). Generalised free energy and active inference. Biological Cybernetics.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import List, Tuple

from environment import EpiParams, DEFAULT_PARAMS, preference_log_prob
from generative_model import (
    BeliefState,
    transition_prior,
    log_obs_likelihood,
    kl_divergence,
    init_belief,
)


# Number of possible actions
N_ACTIONS = 3   # 0: nothing, 1: test/surveil, 2: lockdown

ACTION_NAMES = {
    0: "Do nothing",
    1: "Surveillance",
    2: "Lockdown",
}


# ---------------------------------------------------------------------------
# Expected Free Energy  G(a)
# ---------------------------------------------------------------------------

def expected_free_energy(belief: BeliefState,
                         action: int,
                         params: EpiParams = DEFAULT_PARAMS,
                         n_samples: int = 200,
                         key: jax.Array = None) -> Tuple[float, float, float]:
    """
    Compute G(a) = Epistemic value + Pragmatic value for a candidate action.

    We use a tractable approximation of the Expected Free Energy that captures
    epistemic (uncertainty-reducing) and pragmatic (goal-directed) components.

    Epistemic value (information gain approximation):
        -H[P(o'|a, Q)] ≈ -E_{s'}[ H[Poisson(ρ I' N)] ]
                       = -E_{s'}[ ½ log(2πe ρ I' N) ]     (Gaussian approx)

    Pragmatic value (goal achievement):
        E_{s'}[ log P̃(s') ]

    Parameters
    ----------
    belief    : BeliefState  — current Q(s_t)
    action    : int
    params    : EpiParams
    n_samples : int          — MC samples for approximation
    key       : JAX PRNGKey

    Returns
    -------
    G            : float  — total Expected Free Energy (lower is preferred)
    epistemic    : float  — epistemic component (uncertainty reduction)
    pragmatic    : float  — pragmatic component (preference satisfaction)
    """
    if key is None:
        key = random.PRNGKey(0)

    # Predicted belief under action a
    prior_next = transition_prior(belief, action, params)

    # Sample future states from the predicted belief
    key, subkey = random.split(key)
    eps = random.normal(subkey, shape=(n_samples, 2))
    s_samples = prior_next.mu + prior_next.std * eps   # (n_samples, 2)
    s_samples = jnp.clip(s_samples, 1e-6, 1.0 - 1e-6)

    I_samples = s_samples[:, 1]   # infected fraction

    # --- Epistemic value ---
    # Entropy of the Gaussian observation model H[Normal(μ, σ)] = ½ log(2π e σ²)
    rho_eff = jnp.where(action == 1,
                        jnp.minimum(params.rho / params.noise_reduction, 1.0),
                        params.rho)
    mu_obs_samples = rho_eff * I_samples
    sigma_obs_samples = jnp.sqrt(mu_obs_samples / params.N + 1e-8)
    # Negative entropy of predicted observations (we want to reduce it)
    H_obs = 0.5 * jnp.log(2 * jnp.pi * jnp.e * sigma_obs_samples ** 2)
    epistemic = -jnp.mean(H_obs)   # negative entropy → lower is less uncertain

    # --- Pragmatic value ---
    # KL between predicted state and preferred state P̃
    # We use the preference function as a proxy for log P̃
    pragmatic_samples = jnp.array([
        preference_log_prob(
            jnp.array([s[0], s[1], jnp.clip(1.0 - s[0] - s[1], 0, 1)]),
            action,
            params
        )
        for s in s_samples
    ])
    pragmatic = -jnp.mean(pragmatic_samples)   # negative: we want high pref

    G = epistemic + pragmatic
    return float(G), float(epistemic), float(pragmatic)


# ---------------------------------------------------------------------------
# Softmax action selection (stochastic policy)
# ---------------------------------------------------------------------------

def action_posterior(G_values: jnp.ndarray,
                     temperature: float = 1.0) -> jnp.ndarray:
    """
    Convert G values into a probability distribution over actions.

    P(a) ∝ exp(-G(a) / τ)

    Lower G → higher probability. Temperature τ controls determinism:
      τ → 0   : greedy (always pick argmin G)
      τ → ∞   : uniform random

    Parameters
    ----------
    G_values    : jnp.ndarray shape (n_actions,)
    temperature : float

    Returns
    -------
    probs : jnp.ndarray shape (n_actions,)
    """
    logits = -G_values / temperature
    logits = logits - logits.max()   # numerical stability
    probs = jnp.exp(logits)
    return probs / probs.sum()


def select_action(belief: BeliefState,
                  params: EpiParams = DEFAULT_PARAMS,
                  temperature: float = 1.0,
                  key: jax.Array = None) -> Tuple[int, jnp.ndarray, List]:
    """
    Evaluate G for all actions and sample from the resulting policy.

    Parameters
    ----------
    belief      : BeliefState
    params      : EpiParams
    temperature : float      — policy temperature
    key         : JAX PRNGKey

    Returns
    -------
    action      : int        — selected action
    probs       : jnp.ndarray— action probabilities
    G_details   : list       — [(G, epistemic, pragmatic)] per action
    """
    if key is None:
        key = random.PRNGKey(42)

    G_details = []
    G_values = []

    for a in range(N_ACTIONS):
        key, subkey = random.split(key)
        G, ep, pr = expected_free_energy(belief, a, params, key=subkey)
        G_values.append(G)
        G_details.append((G, ep, pr))

    G_array = jnp.array(G_values)
    probs = action_posterior(G_array, temperature)

    key, subkey = random.split(key)
    action = int(random.choice(subkey, N_ACTIONS, p=probs))

    return action, probs, G_details
