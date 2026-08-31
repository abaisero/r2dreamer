# r2dreamer

A Dreamer-style agent whose policy is trained on rollouts simulated by a learned world model. The research question is whether conditioning the policy-gradient baseline on the simulator's exogenous noise reduces gradient variance.

## Language

**Imagination rollout**:
A trajectory generated entirely inside the world model, starting from a latent state taken from replay. No environment interaction occurs.
_Avoid_: dream, simulated episode, rollout (unqualified)

**Exogenous noise**:
The randomness of a world-model transition, drawn independently of any action. Held as a uniform variate $u$ and its Gumbel transform $g = -\log(-\log u)$; given $g$, a transition is a deterministic function of the state and the action.
_Avoid_: Gumbel noise, transition noise, epsilon

**Exogenous window**:
The span of future exogenous noise the causal critic conditions on at a given step: every remaining draw from that step to the end of the imagination rollout.
_Avoid_: horizon, noise horizon, lookahead

**Standard critic**:
The value network, conditioned on state alone. The sole source of value estimates: bootstrapping, the $\lambda$-return, and the replay value loss all use it, whichever critic the advantage uses.
_Avoid_: critic (unqualified)

**Causal critic**:
A second value network, conditioned on the state *and* the exogenous window. Used only as the baseline subtracted from the return in the policy gradient; it never supplies value estimates. Its correctness condition is independence from the action at the same step, not accuracy.
_Avoid_: causal value, critic (unqualified)

**Gradient variance**:
The variance of the policy-gradient estimator, $\mathrm{Var}[A \nabla \log \pi]$. The quantity the causal critic is meant to reduce and the measure any result is judged by.
_Avoid_: advantage variance, return variance

Naming is a conversational convention, not a code convention: the code says `causal_baseline` and stays that way. "Baseline" and "critic" are interchangeable for these two networks; what matters is that neither is ever left unqualified — always "standard" or "causal". "Horizon" alone always means the discount setting; "imagination horizon" means the rollout length. Neither refers to the exogenous window.
