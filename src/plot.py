"""
plot.py
=======
Visualisation functions for Active Inference epidemic results.

All functions accept the dict returned by simulate.run_episode() or
simulate.run_ablation() and produce matplotlib figures.

Figures produced:
  1. fig_belief_tracking   — true vs. estimated I(t) with uncertainty bands
  2. fig_action_decomp     — epistemic vs. pragmatic G per action over time
  3. fig_phase_portrait    — (S, I) phase plane with belief trajectory
  4. fig_ablation          — infection curves under AI vs. greedy vs. random
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ACTION_COLORS = {0: "#4A90D9", 1: "#F5A623", 2: "#D0021B"}
ACTION_LABELS = {0: "Do nothing", 1: "Surveillance", 2: "Lockdown"}


def _action_background(ax, actions, alpha=0.12):
    """Shade background by action taken."""
    n = len(actions)
    for t in range(n):
        ax.axvspan(t, t + 1, color=ACTION_COLORS[actions[t]], alpha=alpha, linewidth=0)


# ---------------------------------------------------------------------------
# Figure 1: Belief tracking
# ---------------------------------------------------------------------------

def fig_belief_tracking(results: dict, save_path: str = None):
    """
    Plot true I(t) vs. agent's posterior belief μ_I(t) ± 2σ_I.
    Shade background by action. Show raw observations.
    """
    true_I   = results["true_states"][:, 1]
    mu_I     = results["belief_means"][:, 1]
    std_I    = results["belief_stds"][:, 1]
    obs      = results["observations"]  # Already normalized rates
    actions  = results["actions"]
    T        = len(true_I)
    t        = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0F1117")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0F1117")
        for spine in ax.spines.values():
            spine.set_color("#333")

    # Action background
    _action_background(ax1, actions, alpha=0.15)

    # Observations (noisy rates)
    ax1.scatter(t, obs, s=8, color="#888", alpha=0.5, label="Observations (noisy rates)", zorder=2)

    # Posterior uncertainty band
    ax1.fill_between(t, mu_I - 2 * std_I, mu_I + 2 * std_I,
                     color="#00D4FF", alpha=0.15, label="Posterior ±2σ")

    # Posterior mean
    ax1.plot(t, mu_I, color="#00D4FF", lw=1.8, label="Belief μ_I", zorder=4)

    # True state
    ax1.plot(t, true_I, color="#FF6B6B", lw=2, ls="--", label="True I(t)", zorder=5)

    ax1.set_ylabel("Infected fraction I(t)", color="white", fontsize=12)
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#1A1D27", edgecolor="#333", labelcolor="white", fontsize=9)
    ax1.set_title("Active Inference Epidemic Agent — Belief Tracking",
                  color="white", fontsize=14, pad=12)

    # Action plot
    for a in range(3):
        mask = actions == a
        ax2.scatter(t[mask], np.ones(mask.sum()) * a,
                    color=ACTION_COLORS[a], s=20, label=ACTION_LABELS[a])
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Nothing", "Surveil", "Lockdown"], color="white", fontsize=9)
    ax2.set_xlabel("Time (days)", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    ax2.set_title("Action selected", color="#AAA", fontsize=10)

    # Legend patches
    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_LABELS[a]) for a in range(3)]
    ax2.legend(handles=patches, facecolor="#1A1D27", edgecolor="#333",
               labelcolor="white", fontsize=9, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Epistemic vs. Pragmatic decomposition
# ---------------------------------------------------------------------------

def fig_action_decomp(results: dict, save_path: str = None):
    """
    For each time step, show the epistemic and pragmatic components of G
    for each action. Highlight the selected action.
    """
    T           = len(results["actions"])
    t           = np.arange(T)
    G_ep        = results["G_epistemic"]   # (T, 3)
    G_pr        = results["G_pragmatic"]   # (T, 3)
    actions     = results["actions"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle("Expected Free Energy Decomposition: G = Epistemic + Pragmatic",
                 color="white", fontsize=13)

    titles = ["Epistemic Value (uncertainty reduction)", "Pragmatic Value (goal achievement)"]
    G_arrays = [G_ep, G_pr]

    for ax, G_arr, title in zip(axes, G_arrays, titles):
        ax.set_facecolor("#0F1117")
        for spine in ax.spines.values():
            spine.set_color("#333")

        for a in range(3):
            ax.plot(t, G_arr[:, a], color=ACTION_COLORS[a],
                    alpha=0.7, lw=1.2, label=ACTION_LABELS[a])

        # Highlight selected action
        for tt in range(T):
            a = actions[tt]
            ax.scatter(tt, G_arr[tt, a], color=ACTION_COLORS[a],
                       s=18, zorder=5, alpha=0.9)

        ax.set_title(title, color="#CCC", fontsize=11)
        ax.set_xlabel("Time (days)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1A1D27", edgecolor="#333", labelcolor="white", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Phase portrait
# ---------------------------------------------------------------------------

def fig_phase_portrait(results: dict, save_path: str = None):
    """
    (S, I) phase plane showing true trajectory and belief trajectory.
    """
    true_S = results["true_states"][:, 0]
    true_I = results["true_states"][:, 1]
    mu_S   = results["belief_means"][:, 0]
    mu_I   = results["belief_means"][:, 1]
    actions = results["actions"]
    T = len(true_S)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    for spine in ax.spines.values():
        spine.set_color("#333")

    # True trajectory coloured by action
    for t in range(T - 1):
        ax.plot(true_S[t:t+2], true_I[t:t+2],
                color=ACTION_COLORS[actions[t]], lw=1.5, alpha=0.8)

    # Belief trajectory
    ax.plot(mu_S, mu_I, color="#00D4FF", lw=1, ls=":", alpha=0.6, label="Belief trajectory")

    # Start / end markers
    ax.scatter(true_S[0], true_I[0], color="white", s=80, zorder=10, label="Start")
    ax.scatter(true_S[-1], true_I[-1], color="#FF6B6B", s=80, marker="*", zorder=10, label="End")

    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_LABELS[a]) for a in range(3)]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color="#00D4FF", ls=":", label="Belief trajectory"),
        plt.Line2D([0], [0], marker="o", color="white", label="Start"),
        plt.Line2D([0], [0], marker="*", color="#FF6B6B", label="End"),
    ], facecolor="#1A1D27", edgecolor="#333", labelcolor="white", fontsize=9)

    ax.set_xlabel("Susceptible S(t)", color="white", fontsize=12)
    ax.set_ylabel("Infected I(t)", color="white", fontsize=12)
    ax.set_title("Phase Portrait — True Trajectory vs. Belief", color="white", fontsize=13)
    ax.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Ablation comparison
# ---------------------------------------------------------------------------

def fig_ablation(ablation_results: dict, save_path: str = None):
    """
    Compare infection curves under AI agent vs. greedy vs. random policy.
    """
    colors = {"active_inference": "#00D4FF", "greedy": "#F5A623", "random": "#FF6B6B"}
    labels = {"active_inference": "Active Inference (epistemic + pragmatic)",
              "greedy": "Greedy (pragmatic only, τ→0)",
              "random": "Random policy (baseline)"}

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    for spine in ax.spines.values():
        spine.set_color("#333")

    for key, res in ablation_results.items():
        I = res["true_states"][:, 1]
        T = np.arange(len(I))
        ax.plot(T, I, color=colors[key], lw=2, label=labels[key])

    ax.set_xlabel("Time (days)", color="white", fontsize=12)
    ax.set_ylabel("Infected fraction I(t)", color="white", fontsize=12)
    ax.set_title("Ablation: Effect of Epistemic Value on Epidemic Control",
                 color="white", fontsize=13)
    ax.legend(facecolor="#1A1D27", edgecolor="#333", labelcolor="white", fontsize=10)
    ax.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
