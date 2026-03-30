"""
simulate.py
===========
Run a full Active Inference episode and return a results dictionary
suitable for plotting and analysis.

Episode loop (per time step t):
  1. Agent selects action a_t  via select_action()  [agent.py]
  2. Environment steps: s_t → s_{t+1}, o_t          [environment.py]
  3. Agent updates belief via perception_step()      [inference.py]
  4. Log everything

The agent never sees the true state — only observations.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, Any

from environment import (
    EpiParams, DEFAULT_PARAMS,
    sir_step, observe,
)
from generative_model import init_belief, BeliefState
from inference import perception_step
from agent import select_action, ACTION_NAMES


def run_episode(
    n_steps: int = 120,
    params: EpiParams = DEFAULT_PARAMS,
    I_init: float = 0.01,
    temperature: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single Active Inference episode.

    Parameters
    ----------
    n_steps     : int    — episode length in days
    params      : EpiParams
    I_init      : float  — initial infected fraction
    temperature : float  — action-selection temperature (0 = greedy)
    seed        : int    — RNG seed
    verbose     : bool   — print step-by-step summary

    Returns
    -------
    results : dict with keys:
        true_states   : (n_steps, 3)  — true (S, I, R) at each step
        observations  : (n_steps,)    — observed counts
        belief_means  : (n_steps, 2)  — posterior mean (μ_S, μ_I)
        belief_stds   : (n_steps, 2)  — posterior std  (σ_S, σ_I)
        actions       : (n_steps,)    — action taken at each step
        action_probs  : (n_steps, 3)  — P(a) at each step
        G_epistemic   : (n_steps, 3)  — epistemic component per action
        G_pragmatic   : (n_steps, 3)  — pragmatic component per action
        free_energies : (n_steps,)    — F after update at each step
        params        : EpiParams
    """
    key = random.PRNGKey(seed)

    # Initialise true state and belief
    true_state = jnp.array([1.0 - I_init, I_init, 0.0])
    belief = init_belief(I_init=I_init, uncertainty=0.05)

    # Storage
    true_states    = np.zeros((n_steps, 3))
    observations   = np.zeros(n_steps)
    belief_means   = np.zeros((n_steps, 2))
    belief_stds    = np.zeros((n_steps, 2))
    actions        = np.zeros(n_steps, dtype=int)
    action_probs   = np.zeros((n_steps, 3))
    G_epistemic    = np.zeros((n_steps, 3))
    G_pragmatic    = np.zeros((n_steps, 3))
    free_energies  = np.zeros(n_steps)

    for t in range(n_steps):
        # ---- 1. Action selection ----
        key, subkey = random.split(key)
        action, probs, G_details = select_action(
            belief, params, temperature, key=subkey
        )

        # ---- 2. Environment step ----
        key, k_step, k_obs = random.split(key, 3)
        new_state = sir_step(true_state, action, params, k_step)
        obs = observe(new_state, action, params, k_obs)

        # ---- 3. Belief update (perception) ----
        new_belief, F_history = perception_step(
            belief, obs, action, params, n_steps=40
        )

        # ---- 4. Log ----
        true_states[t]   = np.array(new_state)
        observations[t]  = float(obs[0])
        belief_means[t]  = np.array(new_belief.mu)
        belief_stds[t]   = np.array(new_belief.std)
        actions[t]       = action
        action_probs[t]  = np.array(probs)
        free_energies[t] = float(F_history[-1])

        for a_idx, (G, ep, pr) in enumerate(G_details):
            G_epistemic[t, a_idx] = ep
            G_pragmatic[t, a_idx] = pr

        # ---- 5. Advance ----
        true_state = new_state
        belief = new_belief

        if verbose:
            I_true = float(new_state[1])
            I_est  = float(new_belief.mu[1])
            print(
                f"t={t:3d} | I_true={I_true:.4f}  I_est={I_est:.4f} "
                f"| obs={float(obs[0]):.4f} | action={ACTION_NAMES[action]:<14s}"
                f"| F={float(F_history[-1]):6.3f}"
            )

    return {
        "true_states":   true_states,
        "observations":  observations,
        "belief_means":  belief_means,
        "belief_stds":   belief_stds,
        "actions":       actions,
        "action_probs":  action_probs,
        "G_epistemic":   G_epistemic,
        "G_pragmatic":   G_pragmatic,
        "free_energies": free_energies,
        "params":        params,
    }


def run_ablation(n_steps: int = 120,
                 params: EpiParams = DEFAULT_PARAMS,
                 seed: int = 0) -> Dict[str, Any]:
    """
    Run three episodes for ablation analysis:
      - Full agent (epistemic + pragmatic G)
      - Greedy agent (temperature → 0, always exploits)
      - Random agent (uniform action selection)

    Returns dict of results keyed by condition.
    """
    results = {}

    results["active_inference"] = run_episode(
        n_steps=n_steps, params=params, temperature=1.0, seed=seed
    )
    results["greedy"] = run_episode(
        n_steps=n_steps, params=params, temperature=0.01, seed=seed
    )

    # Random baseline: override action selection
    key = random.PRNGKey(seed + 99)
    true_state = jnp.array([1.0 - 0.01, 0.01, 0.0])
    belief = init_belief(I_init=0.01, uncertainty=0.05)

    ts = np.zeros((n_steps, 3))
    obs_arr = np.zeros(n_steps)

    for t in range(n_steps):
        key, k_a, k_step, k_obs = random.split(key, 4)
        action = int(random.randint(k_a, shape=(), minval=0, maxval=3))
        new_state = sir_step(true_state, action, params, k_step)
        obs = observe(new_state, action, params, k_obs)
        new_belief, _ = perception_step(belief, obs, action, params)
        ts[t] = np.array(new_state)
        obs_arr[t] = float(obs[0])
        true_state = new_state
        belief = new_belief

    results["random"] = {"true_states": ts, "observations": obs_arr,
                         "actions": np.random.randint(0, 3, n_steps)}

    return results


if __name__ == "__main__":
    print("Running Active Inference episode (120 days)...\n")
    results = run_episode(n_steps=120, verbose=True)
    I_peak = results["true_states"][:, 1].max()
    print(f"\nPeak infection: {I_peak:.2%} of population")
    print(f"Actions taken: {dict(zip(*np.unique(results['actions'], return_counts=True)))}")
