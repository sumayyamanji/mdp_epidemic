# Active Inference for Epidemic Control
### A Markov Decision Process Agent that Minimises Variational Free Energy

> *"The brain is a hypothesis testing machine... and so is the government during a pandemic."*

This repository implements an **Active Inference agent** that controls a stochastic SIR epidemic under **partial observability**. The agent never sees the true infection state — only noisy hospitalisation counts. It must simultaneously *infer* the hidden state (perception) and *act* to control the outbreak (action), using the same mathematical objective for both.

The core claim: **Perception minimises variational free energy, while action minimises its expected future counterpart (Expected Free Energy).**

---

## Table of Contents

1. [Problem Setup](#1-problem-setup)
2. [The Generative Model](#2-the-generative-model)
3. [Variational Inference and the ELBO](#3-variational-inference-and-the-elbo)
4. [Active Inference: Action via Expected Free Energy](#4-active-inference-action-via-expected-free-energy)
5. [The Exploration–Exploitation Decomposition](#5-the-explorationexploitation-decomposition)
6. [Implementation](#6-implementation)
7. [Results](#7-results)
8. [Installation](#8-installation)
9. [References](#9-references)

---

## 1. Problem Setup

### The Epidemic as a POMDP

We model the epidemic as a **Partially Observable Markov Decision Process**:

- **Hidden state**: $s_t = (S_t, I_t, R_t) \in [0,1]^3$, fractions of Susceptible / Infected / Recovered  
- **Action**: $a_t \in \{0, 1, 2\}$ — do nothing, deploy surveillance, impose lockdown  
- **Observation**: $o_t \sim \text{Poisson}(\rho \cdot I_t \cdot N)$ — noisy hospitalisation counts (normalized to rates for inference)  
- **Ascertainment rate** $\rho \ll 1$: most cases go undetected

The agent sees $o_t$, not $s_t$. This is the fundamental challenge of epidemic management.

### True Dynamics (Stochastic SIR)

The hidden state evolves via an Itô SDE (Euler–Maruyama discretisation):

$$dS = -\beta(a) S I \, dt + \sigma \, dW_1$$
$$dI = \left(\beta(a) S I - \gamma I\right) dt + \sigma \, dW_2$$
$$dR = \gamma I \, dt + \sigma \, dW_3$$

where:
- $\beta(a) = \beta_0 \cdot \mathbf{1}[a \neq 2] + \beta_0 \cdot \lambda_{\text{ld}} \cdot \mathbf{1}[a = 2]$ — lockdown reduces $\beta$
- $\gamma = 0.1$ — recovery rate ($1/\gamma \approx 10$ days)
- $\sigma = 0.02$ — stochastic diffusion (demographic noise)

---

## 2. The Generative Model

The agent maintains an **internal generative model** — its beliefs about how the world works:

$$p(o_{1:T}, s_{1:T} \mid a_{1:T}) = p(s_0) \prod_{t=1}^{T} p(s_t \mid s_{t-1}, a_t) \cdot p(o_t \mid s_t, a_t)$$

### Likelihood: $p(o_t \mid s_t, a_t)$

$$o_t \mid s_t, a_t \sim \text{Normal}\!\left(\rho(a_t) \cdot I_t,\; \sigma_{\text{obs}}^2\right)$$

where $\sigma_{\text{obs}}^2 \approx \rho(a_t) \cdot I_t / N$ (normalized Poisson variance).

Surveillance ($a=1$) increases ascertainment: $\rho(1) = \min(\rho / \lambda_\sigma, 1)$.

**Observations are normalised by population size so that inference operates on infection rates rather than raw counts, improving numerical stability and alignment with the continuous state representation.**

### Transition prior: $p(s_t \mid s_{t-1}, a_t)$

Linearised SIR dynamics around the current mean, with additive Gaussian process noise:

$$p(s_t \mid s_{t-1}, a) = \mathcal{N}\!\left(f(s_{t-1}, a),\; Q\right), \quad Q = \sigma^2 \mathbf{I}$$

### Preference model: $\tilde{p}(o, s)$

Active Inference encodes goals as a prior *preference* over outcomes — not a reward function:

$$\log \tilde{p}(s) = -10 \cdot I - 0.5 \cdot \mathbf{1}[a = 2]$$

This says: the agent strongly prefers low infections and weakly avoids the economic cost of lockdowns.

---

## 3. Variational Inference and the ELBO

The agent cannot compute the true posterior $p(s_t \mid o_{1:t}, a_{1:t})$ exactly (it's intractable for nonlinear models). Instead, it approximates it with a **variational distribution** $Q_\phi(s_t)$, chosen from a tractable family (diagonal Gaussians):

$$Q_\phi(s_t) = \mathcal{N}(\mu_t, \text{diag}(\sigma_t^2))$$

### Deriving the ELBO

We seek the posterior $p(s_t \mid o_t)$. By Bayes' rule:

$$\log p(o_t) = \log \int p(o_t \mid s_t) \, p(s_t) \, ds_t$$

This marginal log-likelihood is intractable. We instead maximise a lower bound. For any distribution $Q$:

$$\log p(o_t) = \mathbb{E}_{Q}\!\left[\log \frac{p(o_t, s_t)}{Q(s_t)}\right] + \underbrace{D_{\mathrm{KL}}\!\left[Q(s_t) \;\|\; p(s_t \mid o_t)\right]}_{\geq\, 0}$$

Since the KL term is non-negative:

$$\log p(o_t) \geq \underbrace{\mathbb{E}_{Q}\!\left[\log p(o_t \mid s_t)\right] - D_{\mathrm{KL}}\!\left[Q(s_t) \;\|\; p(s_t)\right]}_{\text{ELBO} \;=\; \mathcal{L}(\phi)}$$

Maximising the **Evidence Lower Bound (ELBO)** $\mathcal{L}(\phi)$ is equivalent to minimising:

$$\boxed{F(\phi) = -\mathcal{L}(\phi) = D_{\mathrm{KL}}\!\left[Q_\phi(s_t) \;\|\; p(s_t)\right] - \mathbb{E}_{Q_\phi}\!\left[\log p(o_t \mid s_t)\right]}$$

This is the **Variational Free Energy** $F$ — the central quantity in Active Inference.

### Analytic KL for Diagonal Gaussians

For $Q = \mathcal{N}(\mu_Q, \Sigma_Q)$ and $P = \mathcal{N}(\mu_P, \Sigma_P)$, both diagonal:

$$D_{\mathrm{KL}}[Q \| P] = \frac{1}{2} \sum_{i=1}^{d} \left[ \frac{\sigma_{Q,i}^2}{\sigma_{P,i}^2} + \frac{(\mu_{P,i} - \mu_{Q,i})^2}{\sigma_{P,i}^2} - 1 + \log \frac{\sigma_{P,i}^2}{\sigma_{Q,i}^2} \right]$$

This is computed analytically in `generative_model.py`, giving exact gradients.

### The Perception Loop

We minimise $F$ with respect to $\phi = (\mu, \log \sigma^2)$ via gradient descent:

$$\mu \leftarrow \mu - \eta_\mu \nabla_\mu F, \qquad \log \sigma^2 \leftarrow \log \sigma^2 - \eta_\sigma \nabla_{\log \sigma^2} F$$

JAX's `value_and_grad` computes both $F$ and its gradients in one pass. See `inference.py`.

---

## 4. Active Inference: Action via Expected Free Energy

Perception (minimising $F$) gives us beliefs. But how does the agent choose *actions*?

In standard reinforcement learning, the agent maximises expected reward. In Active Inference, the agent minimises **Expected Free Energy** $G(a)$ — the free energy it *expects to experience* if it takes action $a$:

$$G(a) = \mathbb{E}_{Q(o', s' \mid a)}\!\left[\log Q(s' \mid a) - \log \tilde{p}(o', s')\right]$$

where $\tilde{p}$ is the preference model and $Q(s' \mid a)$ is the predicted belief after taking action $a$.

Expanding:

$$G(a) = \underbrace{-\mathbb{E}_{Q(s'\mid a)}\!\left[H\!\left[p(o' \mid s', a)\right]\right]}_{\text{Epistemic value}} + \underbrace{D_{\mathrm{KL}}\!\left[Q(s' \mid a) \;\|\; \tilde{p}(s')\right]}_{\text{Pragmatic value}}$$

The agent selects actions according to the **softmax policy**:

$$\pi(a) \propto \exp\!\left(-G(a) / \tau\right)$$

where $\tau$ is a temperature parameter.

---

## 5. The Exploration–Exploitation Decomposition

The money shot. $G(a)$ uses a tractable approximation that captures two competing drives:

### Epistemic Value (Exploration)

$$G_{\text{epistemic}}(a) = -\mathbb{E}_{Q(s'\mid a)}\!\left[H\!\left[p(o' \mid s', a)\right]\right]$$

This is the **negative expected entropy of future observations**. Actions that are expected to *reduce uncertainty* about the hidden state have low (negative) epistemic value, making them more likely to be selected.

- **Surveillance** ($a=1$) doubles the ascertainment rate → sharper observations → lower entropy → lower $G_{\text{epistemic}}$ → agent is intrinsically motivated to test.

### Pragmatic Value (Exploitation)

$$G_{\text{pragmatic}}(a) = D_{\mathrm{KL}}\!\left[Q(s' \mid a) \;\|\; \tilde{p}(s')\right]$$

This measures the KL divergence between the predicted future state and the preferred state. Actions that are expected to drive $I \to 0$ minimise this term.

- **Lockdown** ($a=2$) reduces $\beta$ → drives $I$ toward 0 → preferred → lower $G_{\text{pragmatic}}$.

**The critical insight:** We use entropy instead of full information gain, and state preferences rather than outcome preferences. This approximation naturally decomposes exploration (test to learn the state) and exploitation (act to change the state) without hand-crafted bonuses. The tension falls out of the mathematics automatically.

| Phase | Dominant drive | Agent's behaviour |
|-------|----------------|-------------------|
| Early outbreak (high uncertainty) | Epistemic | Deploy surveillance |
| Peak (high I, certain) | Pragmatic | Impose lockdown |
| Recovery (low I, low uncertainty) | Neither | Do nothing |

---

## 6. Implementation

### Architecture

```
active-inference-epidemic/
├── src/
│   ├── environment.py       # True SIR POMDP + observation model
│   ├── generative_model.py  # Agent's internal model, BeliefState, F
│   ├── inference.py         # Perception: gradient descent on F
│   ├── agent.py             # Action: G computation + softmax policy
│   └── simulate.py          # Episode runner + ablation utilities
├── notebooks/
│   └── exploration.ipynb    # Experiments and figure generation
└── results/                 # Saved figures
```

### Stack

- **JAX** — autodifferentiation for $\nabla_\phi F$, JIT compilation, functional purity
- **NumPy / Matplotlib** — storage and visualisation
- No deep learning frameworks — this is *mathematical* inference, not neural approximation

### Key design choices

1. **Diagonal Gaussian posterior** — tractable, analytic KL, differentiable parameters
2. **Euler–Maruyama SDE** — captures demographic stochasticity without full agent-based model
3. **Monte Carlo EFE** — $G(a)$ approximated by sampling from $Q(s' \mid a)$; clean and general
4. **JAX `value_and_grad`** — single-pass gradient computation, no manual backprop

---

## 7. Results

### Belief Tracking

The agent accurately tracks the hidden infection curve from noisy counts, with calibrated uncertainty that shrinks as evidence accumulates.

![Belief Tracking](results/belief_tracking.png)

### Exploration–Exploitation

The epistemic and pragmatic components of $G$ drive qualitatively different behaviours. Early in the epidemic, surveillance dominates (epistemic drive). At peak, lockdown dominates (pragmatic drive).

![G Decomposition](results/action_decomp.png)

### Phase Portrait

The agent's belief trajectory (dashed) closely follows the true $(S, I)$ trajectory in phase space, despite never observing $S$ or $I$ directly.

![Phase Portrait](results/phase_portrait.png)

### Ablation: Value of Epistemic Component

Removing epistemic drive (greedy agent, $\tau \to 0$) leads to later detection and a higher infection peak. A random policy performs worst.

![Ablation](results/ablation.png)

---

## 8. Installation

```bash
git clone https://github.com/YOUR_USERNAME/active-inference-epidemic
cd active-inference-epidemic
pip install jax jaxlib numpy matplotlib jupyter
```

### Run a single episode

```bash
cd src
python simulate.py
```

### Generate all figures

Run the notebook: 

`notebooks/exploration.ipynb`

---

## 9. References

- Friston, K. et al. (2017). **Active inference and epistemic value**. *Cognitive Neuroscience*, 8(4), 187–214.
- Parr, T. & Friston, K. (2019). **Generalised free energy and active inference**. *Biological Cybernetics*, 113(5–6), 495–513.
- Da Costa, L. et al. (2020). **Active inference on discrete state-spaces: A synthesis**. *Journal of Mathematical Psychology*, 99, 102474.
- Wainwright, M. & Jordan, M. (2008). **Graphical models, exponential families, and variational inference**. *Foundations and Trends in Machine Learning*, 1(1–2), 1–305.
- Kermack, W. O. & McKendrick, A. G. (1927). **A contribution to the mathematical theory of epidemics**. *Proceedings of the Royal Society A*, 115(772), 700–721.
- Bottemanne, H. & Friston, K.J. (2021). **An active inference account of protective behaviours during the COVID-19 pandemic**. *Cognitive, Affective, & Behavioral Neuroscience*, 21(5), 1117–1129. https://doi.org/10.3758/s13415-021-00947-0
- Wood, F., Warrington, A., Naderiparizi, S., Weilbach, C., Masrani, V., Harvey, W., Ścibior, A., Beronov, B., Grefenstette, J., Campbell, D. & Nasseri, S.A. (2022). **Planning as Inference in Epidemiological Dynamics Models**. *Frontiers in Artificial Intelligence*, 4, 550603. doi: 10.3389/frai.2021.550603

---

*Built as a demonstration that the mathematics of Bayesian brain theory (Karl Friston's Free Energy Principle) can be applied directly to public health decision-making — where the "brain" is the government, the "body" is the population, and the "environment" is the epidemic.*
