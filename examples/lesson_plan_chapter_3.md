# Connect Four RL Agent — Chapter 3: The Bootstrapping Spectrum and What Your Agent Actually Learned

Chapters 1 and 2 taught you two families of RL algorithms — value-based (DQN) and policy-based (REINFORCE, PPO). You built working agents, trained them, and compared their performance. But you made choices along the way that you may not fully understand yet: Why does DQN update from one step ahead while REINFORCE waits for the whole game to end? Why does PPO's GAE use λ=0.95 instead of 0 or 1? And the deeper question: do your agents actually *understand* Connect Four, or did they just memorize patterns against specific opponents?

This chapter steps back from building new agents and focuses on **understanding**. Part I reveals the theoretical spine that connects every algorithm you've built — the bias-variance trade-off in temporal difference learning. Part II turns that understanding inward, probing what your trained networks actually learned and where they break.

No new agents in this chapter. Instead, you'll modify existing ones, run controlled experiments, and develop the kind of intuition that lets you diagnose problems and make better design decisions in any RL project.

### A note on infrastructure

Since this chapter was originally outlined, the repo has consolidated:

- **One training entry point.** All algorithms now run through [training/train.py](training/train.py) with `--algorithm {dqn,reinforce,ppo}` and `--opponent {heuristic,optimal,random,self-play}`. There is no longer a separate `train_reinforce.py` / `train_ppo.py`. Wherever this chapter says "create a script that trains …", read it as "invoke `python -m training.train --algorithm X` with appropriate flags, possibly extending the CLI."
- **MLflow is wired in.** Chapter 4 already landed: [training/mlflow_utils.py](training/mlflow_utils.py) logs hyperparameters, metrics, and artifacts on every run, and the local `mlflow.db` keeps history. **Every experiment in this chapter should be an MLflow run.** When the text says "plot the learning curves," log the metrics to MLflow as you train, then pull them back out for plotting. This makes the comparisons reproducible and turns "what hyperparameters did my best run use?" from a directory-grep into a query. (Use `mlflow ui` to browse.)
- **`uv` is the package manager.** Always run scripts as `uv run python -m training.train …`, not `python -m …`.
- **The strong opponent is `OptimalAgent`.** Where the chapter says "minimax-style opponent," it now means [agents/optimal_agent.py](agents/optimal_agent.py), a rule-based agent with threat detection, fork creation, trap avoidance, and positional scoring. Its `random_move_prob` is a soft difficulty knob (1.0 = random, 0.0 = full strength), and [training/curriculum.py](training/curriculum.py) lowers it adaptively as the agent improves.

The conceptual material — bias-variance, TD(λ), the deadly triad, probing — is unaffected. Only the build instructions and file paths have moved.

---

## Part I: The Bootstrapping Spectrum

---

## Module 15 — Prediction Before Control

**Goal:** Separate the problem of *evaluating* states from the problem of *choosing* actions. This distinction clarifies everything that follows.

### Key Concepts

Every RL algorithm you've built is doing two things at once: **prediction** (how good is this state?) and **control** (what action should I take?). These are tangled together in DQN (Q-values serve both purposes) and in actor-critic methods (the value network predicts, the policy network controls). This module isolates prediction so you can study it clearly.

- **The prediction problem**: Given a policy π (any policy — it doesn't have to be good), estimate V^π(s) — the expected return from state s if the agent follows π from that point on. This is the value function. It doesn't tell you what to *do*; it tells you how good things *are*. Think of V(s) as a real-estate appraiser: it doesn't decide what to buy, it just estimates the price tag. Control is the buyer's decision; prediction is the appraisal that informs the decision.

- **Why prediction matters for control**: Every algorithm you've built relies on accurate value estimates. DQN picks the action with the highest Q-value — if Q-values are wrong, the policy is wrong. PPO's advantage `A(s,a) = G - V(s)` compares actual returns to predicted value — if V(s) is inaccurate, the advantage signal is noisy. Better prediction → better control. The contrapositive is also true and underrated: if your agent isn't learning, *prediction* may be broken even when *control* (the policy update rule) is fine.

- **Two extremes of estimation**:
  - **Monte Carlo (MC)**: Wait until the episode ends, compute the actual return Gₜ, and use that as the estimate. No bias — you saw the real outcome. But high variance — one episode's return is a noisy sample.
  - **Temporal Difference (TD)**: Don't wait. After one step, estimate the return as `r + γ·V(s')` — the immediate reward plus the discounted estimate of the next state. Low variance (you're averaging over your value estimate), but biased (your estimate of V(s') might be wrong).

  Intuition pump: imagine you're predicting whether a chess game ends in a win. **MC** is the friend who refuses to guess until the king topples — always correct in hindsight, but unhelpful at move 12. **TD** is the friend who says "this position looks +0.8 to me" mid-game — fast and consistent, but only as good as their positional judgment. Early in training your TD friend is a beginner; their estimates are confidently wrong. As they improve, TD's bias shrinks. MC, meanwhile, is *always* right at the end and *always* useless until then.

- **Bias vs. variance, concretely**:
  - **Bias** = systematic error. If V(s') tends to underestimate the true value by 0.2, every TD target inherits that 0.2 offset. Averaging over a thousand updates does not fix bias.
  - **Variance** = random spread around the truth. Two MC samples from the same state can differ wildly because the rest of the episode is random. Averaging over a thousand samples *does* fix variance.

  The deep point: **bias survives averaging; variance dies under it.** This is why people occasionally prefer MC's high-variance honesty to TD's low-variance lies — a thousand high-variance samples converge to truth, a thousand biased samples converge to bias.

- **Where your algorithms fall**:
  - DQN uses **TD(0)** — one-step bootstrapping. The target is `r + γ · max Q(s', a')`. Bootstrapping is the technical term for "estimating an estimate from another estimate" — DQN literally pulls itself up by its own laces, which is why it needs stabilizers (target network, replay buffer) to keep from falling over.
  - REINFORCE uses **Monte Carlo** — the full discounted return from the episode. No bootstrapping anywhere. That's why REINFORCE doesn't need a target network or a replay buffer: there's no estimate-of-an-estimate to destabilize.
  - PPO's GAE with λ=0.95 is **almost Monte Carlo**, but not quite — it blends TD and MC. λ=0.95 is closer to MC than to TD(0); you can think of it as "MC, but with a small TD shock absorber to dampen variance."

### No Code This Module

This is conceptual. Make sure you can answer:

1. Your DQN agent updates Q-values using `target = r + γ · Q(s')`. What assumption does this make about V(s')? When is this assumption most wrong?
<!-- The assumption is that Q(s') is accurate. Early in training when the network is randomly initialized, Q(s') is essentially noise — so the target is noise + reward. This is the bootstrapping problem: you're updating estimates toward other estimates. The target network helps (it's more stable), but the fundamental issue remains. This is why DQN needs many iterations — each update only makes Q slightly more accurate, and accuracy cascades slowly from terminal states backward. -->

2. REINFORCE waits for the full episode. Why does this eliminate bias but increase variance? Think concretely about two 20-move Connect Four games that both end in a win — are the returns for each move identical?
<!-- Both games end in +1, so the final return is the same. But intermediate returns differ because discounting depends on game length and move position. More importantly, the *path* to the win differs — one game might have been a close game where the agent almost lost, the other a blowout. Monte Carlo can't tell the difference; it just sees the final outcome. This is the variance: two games with the same outcome produce different return sequences, and the agent has to average over many such samples to get a stable signal. The "no bias" part: you literally observed the real return — no estimates involved. -->

3. Imagine you have a perfect value function V*(s) for Connect Four. If you plug it into a TD update `r + γ·V*(s')`, is the target still biased? What about if V(s) is trained but inaccurate?
<!-- With a perfect V*, the TD target is unbiased — it equals the true expected return. The bias in TD comes entirely from using an *imperfect* estimate. This is the key insight: TD's bias is not inherent to the method, it's a consequence of approximation error. As V gets more accurate, TD's bias shrinks. Monte Carlo avoids this by never using V at all — but pays with variance. -->

### Exercises

- [ ] **Map your algorithms.** Create a table with columns: Algorithm, Update target, Uses V(s')?, Waits for episode end?, Bias, Variance. Fill in rows for DQN, REINFORCE, and PPO-GAE. This should crystallize the relationships.
- [ ] **Thought experiment — extreme cases.** Imagine a Connect Four game that lasts exactly 2 moves (you play, opponent plays, game ends). Is there any difference between MC and TD in this case? What about a hypothetical game that lasts 1,000 moves?
- [ ] Re-read your PPO rollout buffer's `finalize()` method in [rollout_buffer.py](training/rollout_buffer.py). Find the line `delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]`. This is a TD error. Find where lambda blends these TD errors. You already implemented the theory — now name what you built.

---

## Module 16 — N-Step Returns: The Simplest Spectrum

**Goal:** Implement n-step return targets and observe the bias-variance trade-off directly.

### Key Concepts

N-step returns are the simplest way to move between TD(0) and Monte Carlo. Instead of bootstrapping after 1 step or waiting for the full episode, bootstrap after **n** steps:

```
1-step (TD):  G₁ = r₁ + γ · V(s₂)
2-step:       G₂ = r₁ + γ·r₂ + γ² · V(s₃)
3-step:       G₃ = r₁ + γ·r₂ + γ²·r₃ + γ³ · V(s₄)
...
n-step:       Gₙ = r₁ + γ·r₂ + ... + γⁿ⁻¹·rₙ + γⁿ · V(sₙ₊₁)
Full MC:      G  = r₁ + γ·r₂ + ... + γᵀ⁻¹·rₜ
```

As n increases:
- **More real rewards, less bootstrapping** → less bias (more of the target is observed, not estimated)
- **More terms in the sum** → more variance (more random steps contribute noise)
- At n = episode length, you recover Monte Carlo exactly

**A useful intuition pump.** Think of each n-step target as a sandwich. The bread is observed rewards (real, no bias). The filling is the bootstrap term γⁿ·V(s_{n+1}) (estimate, biased). At n=1, the sandwich is mostly filling — almost all of the target's value comes from V(s'). At n=∞ (MC), there's no filling at all — pure bread. Picking n is choosing the bread-to-filling ratio. More bread → more honest but more variable (the rewards depend on what random actions/opponents did). More filling → more stable but only as accurate as your value network's current opinion.

**Why bias and variance scale this way, in one paragraph.** The bias of an n-step target equals γⁿ·(V_estimate − V_true) at s_{n+1}. As n grows, γⁿ shrinks (γ=0.99 → γ¹⁰ ≈ 0.90, γ⁵⁰ ≈ 0.61) and the value-estimate error gets discounted away — that's the bias term dying. The variance, meanwhile, is the variance of a sum of n random reward terms (plus the small variance from the final bootstrap). Roughly, more random terms in the sum → bigger spread. So bias drops geometrically with n while variance grows roughly linearly. There's a sweet spot, and for Connect Four it sits somewhere in the n=3 to n=7 range — short enough that the rewards-sum variance stays manageable, long enough that V(s_{n+1})'s bias is meaningfully discounted.

**Why this matters specifically for Connect Four.** The reward is *terminal-only*: ±1 at the end, zero everywhere in between. With n=1, the reward signal only ever flows directly into the second-to-last state — every earlier state has to learn its value from the (very-slowly-updating) bootstrap of the state after it. The reward thus has to crawl backward through training, one state per cycle, like heat conducting along a bar. With n=5, the reward signal reaches the last 5 states in one update. That isn't just faster — it's structurally different. Sparse-reward environments are exactly where n>1 pays off most.

Your DQN agent already has n-step support via [training/n_step_buffer.py](training/n_step_buffer.py), and `--n-step` is a CLI flag on the unified trainer (default 3). So you don't have to write new training code — just run the trainer with different `--n-step` values and compare runs in MLflow.

### What to Build

Rather than a one-off notebook, run a small sweep through the unified trainer and use MLflow to compare. Suggested approach:

1. **Sweep script** (`notebooks/04_nstep_comparison.py`): a thin Python wrapper that calls `training.train.main()` (or shells out to `uv run python -m training.train`) for each value of n. For each run:
   - `--algorithm dqn --opponent heuristic --episodes 10000`
   - `--n-step {1,3,5,10}`
   - Fixed `--seed`, `--lr`, and `--arch` across runs.
   - Tag the MLflow run with `experiment="nstep_comparison"` and `n_step={n}` so they're easy to filter.

2. **Plot script** (same file, second half): query MLflow for the four runs, pull the `eval/win_rate` and `train/loss` metric histories, and overlay them on one figure. Save as `notebooks/nstep_comparison.png`.

**Suggested prompt to Claude Code:** *"Create notebooks/04_nstep_comparison.py. The script should (a) launch four DQN training runs against the heuristic opponent for 10,000 episodes each with n_step ∈ {1, 3, 5, 10}, sharing seed/lr/architecture, by invoking training.train.main programmatically with the right argparse Namespace, (b) tag each MLflow run with experiment='nstep_comparison' and n_step as a parameter, (c) after all runs finish, use the MLflow client to pull win-rate and loss metric histories for each run and produce an overlay plot. Save as notebooks/nstep_comparison.png."*

### After Running the Experiment

1. **Look at the learning curves.** Which value of n learns fastest in the first 2,000 episodes? Which reaches the highest final win rate? These may not be the same n.

2. **Look at the loss curves.** Higher n should produce noisier loss values (higher variance in targets). Can you see this in the plots?

3. **Think about why n matters for Connect Four specifically.** Connect Four has sparse reward — only the last move gets +1 or -1. With n=1, the reward signal only reaches the second-to-last state directly; all other states rely on bootstrapped values cascading backward through training. With n=5, the reward reaches the last 5 states directly. This is a concrete manifestation of the credit assignment problem.

### Key Insight

There's a subtle interaction between n-step returns and the replay buffer. Your replay buffer stores transitions and samples them uniformly. But an n-step transition `(s, a, Gₙ, sₙ₊₁)` was computed using a *specific sequence of actions* taken by a *specific policy* (the one active when the data was collected). If the policy has changed since then, the n-step return is computed under the wrong policy — it's **stale**.

This staleness gets worse as n increases. For n=1, the return is `r + γ·V(s')` — only V(s') can be stale. For n=10, the return includes 10 rewards and 10 implicit action choices from an old policy. This is the **off-policy problem**, and it's why large n + replay buffer can be unstable.

**The intuition for why staleness hurts.** Imagine you stored a 10-step return from 50,000 episodes ago, when your DQN basically played random moves. The "sum of rewards along that trajectory" reflects what *random-DQN* would have collected. But your *current* DQN, having learned, would have made different action choices at every step — probably better ones, leading to a higher return. So the stored target is an *underestimate* of what your current policy would actually achieve. Train on it long enough and you're systematically pulling your Q-values toward old, worse behavior. With n=1, this contamination is bounded — only the one bootstrap term V(s') is wrong, and at least the immediate reward is honest. With n=10, ten consecutive action choices are all "wrong policy," and the staleness compounds.

This is why principled off-policy n-step methods (Retrace(λ), V-trace) include **importance-sampling corrections** that down-weight transitions whose action choices the current policy wouldn't have made. We don't implement those here — but understanding why they're needed is half the point of the experiment.

### Exercises

- [ ] Run the comparison script. Identify the best n for Connect Four. Is it the same when training against random vs. heuristic?
- [ ] **Compute by hand.** In a 15-move game ending in a win (+1), with γ=0.99: What is the 1-step return for move 10? The 3-step return? The full Monte Carlo return? Use a calculator.
- [ ] **Experiment — n = episode length.** Set n to something very large (e.g., 100, larger than any game). This should behave like Monte Carlo. Does it? Compare the loss curve to n=1. You should see smoother learning but higher loss variance.
- [ ] **Predict before running:** If you increase the replay buffer from 100K to 1M, will large n become more or less stable? Think about how much older the stored n-step transitions become. Run it and check.
- [ ] **Thought experiment:** If you used n-step returns *without* a replay buffer — training on each transition as it's collected — would the staleness problem go away? What new problem would appear? (Hint: think about why DQN needs a replay buffer in the first place.)

---

## Module 17 — TD(λ) and Eligibility Traces

**Goal:** Understand the elegant solution to "which n should I pick?" — blend *all* values of n simultaneously using λ as a mixing parameter.

### Key Concepts

Picking a single n is a crude instrument. Each n-step return is a valid estimate with its own bias-variance profile. TD(λ) says: why not use a **weighted average** of all n-step returns?

- **λ-return** (forward view):
  ```
  Gₜᵏ = (1-λ) · [λ⁰·G₁ + λ¹·G₂ + λ²·G₃ + ... ] + λᵀ⁻ᵗ · Gₜ (MC)
  ```
  Each n-step return Gₙ gets weight `(1-λ)·λⁿ⁻¹`. The weights sum to 1 (it's a valid average). The geometric decay by λ means shorter-term returns get more weight.

  - **λ = 0**: All weight on G₁ (1-step return) → TD(0). Maximum bootstrapping, minimum variance, maximum bias.
  - **λ = 1**: All weight on the full Monte Carlo return → TD(1) = MC. No bootstrapping, maximum variance, zero bias.
  - **λ = 0.95**: Heavy weight on longer returns, but still some stabilizing influence from short-term bootstrapping. This is what your PPO GAE uses.

  **Why this average is useful, intuitively.** Every individual n-step return is a noisy estimator of the true value. Some are biased (small n, lots of bootstrap), some are high-variance (big n, lots of random rewards). If you average several noisy estimators with different error profiles, the average's error is *smaller than any individual estimator's*, provided the errors aren't perfectly correlated. The λ-return is exactly this trick applied to bootstrapping: a portfolio of n-step returns whose individual flaws partially cancel. λ then becomes the "asset allocation" knob — how aggressively do you reach for the longer-horizon estimators?

- **Why geometric weighting?** It's the simplest weighting scheme that (a) is a valid average, (b) decays smoothly, and (c) has a single tunable parameter. Other weighting schemes are possible but don't have the elegant backward-view equivalence. The geometric decay also has a nice property: the **effective horizon** of the λ-return is roughly 1/(1−λ) steps. For λ=0.9, that's 10 steps; for λ=0.95, 20 steps; for λ=0.99, 100 steps. So picking λ is implicitly picking how far into the future the agent's credit-assignment reaches.

- **Eligibility traces** (backward view): The forward view (weighted sum of n-step returns) requires looking into the future, which seems impractical for online learning. The **backward view** achieves the same update using **eligibility traces** — a per-state (or per-parameter) running tally that records how recently and frequently each state was visited.

  ```
  eₜ = γ·λ·eₜ₋₁ + ∇V(sₜ)    (accumulating trace)
  δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)  (TD error)
  Δθ = α · δₜ · eₜ             (parameter update)
  ```

  The trace `eₜ` is a vector the same size as the parameters. After visiting a state, its trace spikes; then it decays by γλ each step. When a TD error occurs (e.g., the agent wins), the update is applied to all recently-visited states in proportion to their trace — states visited more recently and more often get a bigger update. This is **credit assignment through decay**, and it's more principled than REINFORCE's approach of just discounting the final reward backward.

  **The mental model.** Imagine each parameter has a small glowing afterimage that lights up when the parameter is used (∇V(sₜ) flashes brightly) and dims geometrically each step (×γλ). When a surprise arrives — a positive or negative TD error — every parameter receives an update proportional to how brightly it's currently glowing. Recently-used parameters get a big update; long-forgotten ones get almost nothing. The trace is **memory of recent involvement**, and λ controls how long that memory lasts.

  This is a much more biologically plausible learning rule than backprop-through-time. It's also strictly local in time: you only ever need the *current* trace vector and the *current* TD error to make an update. No replay buffers, no lookahead, no remembering trajectories. That elegance is why TD(λ) was a foundational result.

- **Forward = backward**: The remarkable result (Sutton & Barto, Chapter 12) is that the forward view (λ-return) and backward view (eligibility traces) produce **identical updates** over a complete episode. The forward view is easier to understand; the backward view is easier to implement for online learning. The proof is a satisfying telescoping-sum argument — if you read one technical derivation from Sutton & Barto this chapter, make it this one.

### What to Build

This module is the one place in the chapter where you write meaningful new training code — TD(λ) with eligibility traces isn't in the repo and doesn't slot into the unified trainer cleanly (it's online, gradient-trace-based, no replay buffer). That's fine; treat `notebooks/05_td_lambda_prediction.py` as a self-contained value-prediction experiment that *uses* [models/value_network.py](models/value_network.py) and the env, but has its own training loop.

1. **TD(λ) value prediction:** *"Create notebooks/05_td_lambda_prediction.py. It should: (a) instantiate a ValueNetwork (reuse models/value_network.py), (b) play self-play games with HeuristicAgent on both sides — this fixes the policy so we're doing pure prediction, not control, (c) implement online TD(λ) with eligibility traces: after each step compute δ = r + γ·V(s') − V(s), update a per-parameter trace e ← γ·λ·e + ∇V(sₜ), and apply θ ← θ + α·δ·e, (d) train one model per λ ∈ {0, 0.5, 0.9, 1.0} for ~5,000 episodes each, (e) evaluate by holding out 500 episodes and measuring MSE between V(s) and the actual Monte Carlo return for every state visited, (f) log everything to MLflow (one run per λ, tagged experiment='td_lambda'), and (g) plot MSE-over-training for all four λ values overlaid."*

   Implementation note: PyTorch doesn't expose eligibility traces natively. The simplest implementation builds the trace as a list of tensors with the same shapes as `model.parameters()`, updates it manually after each `V(s).backward()`, then applies `param.data += lr * delta * trace[i]` per parameter (with `torch.no_grad()`). Don't use `optimizer.step()` — you're not doing standard gradient descent, you're applying a trace-weighted update.

2. **Visualizing the traces:** *"Add a visualization to the script: for a single episode, plot the eligibility trace L2-norm over time (x-axis = move number, y-axis = ‖e‖). Show this for λ=0.5 and λ=0.95 on the same plot. The norm should spike each step (from the ∇V(sₜ) injection) and decay between steps by factor γλ. λ=0.95 should show a much slower decay envelope."*

### After Running the Experiment

1. **Compare prediction accuracy.** Which λ produces the most accurate V(s) estimates fastest? The answer will depend on how long the games are and how accurate V(s) is at different training stages.

2. **Watch the trace visualization.** For λ=0.5, the trace should decay quickly — the first move is nearly forgotten by move 10. For λ=0.95, the trace lingers — the first move still has ~60% of its peak magnitude after 10 steps (`0.99 × 0.95)^10 ≈ 0.52`). This is how λ controls the effective "memory horizon" of credit assignment.

3. **Connect to GAE.** Open your PPO rollout buffer's `finalize()` method again. The GAE computation `gae = delta + gamma * gae_lambda * gae` is the backward-view eligibility trace for advantages. You built TD(λ) without knowing it. The only difference: GAE applies the trace to advantage estimation, not value prediction directly.

### Exercises

- [ ] **Compute the weights by hand.** For λ=0.9 in a 5-step episode: what weight does the 1-step return get? The 2-step? The 3-step? The full MC return? Verify they sum to 1.
<!-- Weights: (1-0.9)·0.9^0 = 0.1, (1-0.9)·0.9^1 = 0.09, (1-0.9)·0.9^2 = 0.081, (1-0.9)·0.9^3 = 0.0729, and the remaining 0.6561 goes to the MC return. Sum: 0.1 + 0.09 + 0.081 + 0.0729 + 0.6561 = 1.0 -->
- [ ] Run the prediction experiment. Identify the best λ for Connect Four game lengths (typically 15-25 moves).
- [ ] **Experiment — combine with n-step DQN.** Your DQN's n-step already uses a fixed n. What if you used the λ-return instead? Conceptually, how would you modify the `NStepBuffer` to compute λ-returns? You don't need to implement this — just sketch the change and think about why it's harder than fixed n with a replay buffer.
- [ ] **Connect the concepts.** Fill in this table:

  | Setting | λ value | n equivalent | Bias | Variance | Your algorithm |
  |---------|---------|-------------|------|----------|----------------|
  | TD(0) | 0 | n=1 | High | Low | DQN |
  | TD(λ) | 0.95 | ~weighted blend | Low-medium | Medium | PPO (GAE) |
  | MC | 1 | n=full episode | None | High | REINFORCE |

- [ ] **Thought experiment:** Eligibility traces decay by γλ per step. With γ=0.99 and λ=0.95, how many steps until the trace falls to 10% of its peak? To 1%? Does this "effective horizon" feel appropriate for a 20-move Connect Four game?
<!-- Decay factor = 0.99 × 0.95 = 0.9405 per step. To 10%: 0.9405^n = 0.1 → n ≈ 37.5. To 1%: 0.9405^n = 0.01 → n ≈ 75. So in a 20-move game, even the earliest moves retain significant trace magnitude at game end — appropriate since early moves in Connect Four really do matter. -->

---

## Module 18 — The Unified View

**Goal:** See all your algorithms as points in a single design space. Develop the vocabulary to reason about any RL algorithm's trade-offs.

### Key Concepts

You now have the framework to understand **any** value-based or actor-critic RL algorithm along a few key axes. Every algorithm makes choices along these spectra:

**Axis 1: Bootstrapping depth (bias-variance)**
```
TD(0) ←————————— TD(λ) ——————————→ Monte Carlo
n=1                                  n=full episode
λ=0              λ=0.95              λ=1
High bias        Balanced             Zero bias
Low variance     Medium variance      High variance
DQN              PPO (GAE)            REINFORCE
```

**Axis 2: On-policy vs. off-policy**
```
On-policy ←——————————————————————→ Off-policy
(learn from current policy)         (learn from stored data)
REINFORCE, PPO                      DQN (replay buffer)
Fresh data, no staleness            Reuses data, sample efficient
Cannot use replay buffer            Can use replay buffer
```

**Axis 3: Value-based vs. policy-based**
```
Value-based ←—————————————————→ Policy-based
(learn Q, derive policy)          (learn π directly)
DQN                               REINFORCE
Deterministic derived policy      Stochastic policy
                    ↕
              Actor-Critic
              (learn both V and π)
              PPO
```

- **The deadly triad**: Sutton & Barto identify three features that, when combined, can cause instability: (1) function approximation (neural networks), (2) bootstrapping (TD, not MC), and (3) off-policy learning (replay buffer). DQN has all three — which is why it needs a target network, experience replay, and careful hyperparameter tuning to remain stable. REINFORCE avoids the triad by using MC (no bootstrapping) and on-policy learning — but pays with high variance. PPO stays on-policy and uses moderate bootstrapping (GAE), avoiding the worst instabilities.

  **Why these three ingredients combust together.** Each feature alone is fine. Function approximation alone is supervised learning — it works. Bootstrapping alone (tabular TD) is provably convergent. Off-policy learning alone (with MC returns, like off-policy REINFORCE) is also stable. The trouble is the **feedback loop** that forms when all three combine: function approximation means an update at state s changes V everywhere (generalization is also "leakage"); bootstrapping means V(s) is the target for updates at other states; off-policy means you're updating from data the current policy didn't generate. So a bad value at one state propagates to other states through the function approximator, those propagated values become the bootstrap targets for the *next* update, and the data driving the whole loop reflects a distribution that doesn't match where the policy actually wants to be. Errors compound instead of damping. The target network and replay buffer are the engineering hacks that interrupt this feedback — the target network freezes the bootstrap source so updates can't immediately re-target themselves, and the replay buffer averages over a long history of policies so no single bad policy dominates the data.

- **Why this matters**: When you encounter a new RL problem, these axes tell you where to start. Short episodes with dense reward? TD(0) is fine — low bias because episodes are short, and you get reward signal quickly. Long episodes with sparse reward? You need higher n or higher λ to propagate the reward signal backward. Expensive to collect data? Off-policy methods let you reuse experience. Continuous action space? Policy-based methods handle this naturally.

- **A unified story for your three agents.** Stand back and look at what you actually built:
  - DQN solves prediction (Q-values) and gets control for free via argmax. It uses TD(0) for stability under function approximation, replay for sample efficiency, target networks to defuse the triad. Pays for it with brittleness — tune the hyperparameters wrong and the whole thing oscillates.
  - REINFORCE solves control directly (policy gradient) and never bothers with a value function. Uses MC returns, so no bias, but the variance is brutal. Pays for it with sample inefficiency — needs many episodes per real improvement.
  - PPO is the synthesis. It does control via policy gradient (like REINFORCE) but uses a value network as a baseline (like DQN). The GAE λ-return gives it MC-like accuracy with TD-like stability. The clip ratio is a separate trick that prevents the policy from drifting too far from the rollout-collection policy in one update — a kind of "trust region without the math."
  
  Three algorithms, one design space.

### No Code This Module

This is the conceptual capstone of Part I. Revisit your code with fresh eyes.

### Exercises

- [ ] **Revisit DQN with fresh vocabulary.** Open [dqn_agent.py](agents/dqn_agent.py) and find the target computation. Write a comment (for yourself, not the code) that says: "This is a TD(0) update with off-policy correction via target network, using function approximation — all three elements of the deadly triad." Do you understand why each stabilization mechanism (target network, replay buffer, soft updates) is necessary?

- [ ] **Revisit PPO with fresh vocabulary.** Open the PPO rollout buffer and agent. Identify: (1) where GAE computes the λ-return (TD(λ) backward view), (2) why PPO doesn't need a replay buffer (on-policy), (3) how the clip ratio prevents the "big step" problem. Write a one-paragraph explanation of PPO using the terminology from this module.

- [ ] **Design exercise.** A friend is building an RL agent for a game with these properties: (a) episodes last 500+ steps, (b) reward is given every step (dense), (c) the action space is continuous (e.g., angles and forces). Where on each axis should they be? Which algorithm family would you recommend and why?
<!-- Long episodes + dense reward → TD(0) or low λ is fine, reward signal propagates naturally. Continuous action space → policy-based or actor-critic, not pure value-based (can't argmax over continuous actions). Dense reward reduces the need for high n or λ. Recommendation: PPO with low λ (e.g., 0.8), or SAC (off-policy actor-critic designed for continuous control). -->

- [ ] **Design exercise.** Now the game has: (a) episodes last 10 steps, (b) reward only at the end (sparse), (c) 5 discrete actions. Different answer?
<!-- Short episodes + sparse reward → higher λ or MC is fine. The variance penalty of MC is small because episodes are short. Discrete actions → value-based is viable. DQN with n-step (n=5 or higher) or REINFORCE both work. With only 10 steps, even MC has manageable variance. -->

- [ ] **The big question.** If you could restart this project knowing what you know now, which algorithm and configuration would you choose for Connect Four? Write a paragraph justifying your choice using the axes from this module.

---

## Part II: What Did Your Agent Actually Learn?

---

## Module 19 — Probing Your Networks

**Goal:** Move from evaluating agents by win rate (a single number) to understanding *what knowledge* the networks have internalized.

### Key Concepts

Win rate tells you how well the agent performs against specific opponents. It tells you nothing about *why* it wins or *what it understands*. A high win rate against a random agent could mean the agent learned deep strategy — or it could mean the agent learned to not make obviously illegal moves. Probing techniques let you ask more specific questions.

**The framing that makes probing make sense.** A neural network is a high-dimensional function that produces outputs from inputs. Treat it as a black box and you can only ask "does it perform well?" Open the box and you can ask "what does it know?" Probing is what radiologists do — instead of asking "is the patient healthy?" they image the patient. Each probe in this module is a different imaging technique applied to your network: diagnostic positions are stress tests, value trajectories are time-series scans, saliency maps are X-rays of attention. None of these is conclusive on its own, but combined they give you a picture of what the network has actually internalized.

- **Diagnostic board positions**: Handcraft board states that test specific knowledge. Does the agent see a winning move? Can it spot a fork (two ways to win)? Does it block opponent threats? These are the Connect Four equivalent of unit tests for the network's understanding.

- **Value landscape**: For a given board, how does V(s) change across successive moves? A well-trained value network should show V(s) increasing when the agent makes good moves and decreasing when the opponent gains advantage. Plotting V(s) over a full game reveals whether the network tracks the game state accurately.

- **Policy entropy**: A policy network's entropy measures how "certain" it is. Low entropy (one action dominates) means the network is confident. High entropy (uniform distribution) means it's uncertain. Plotting entropy across board positions reveals where the agent knows what to do vs. where it's guessing.

- **Saliency maps**: Which cells on the board matter most for the network's decision? Saliency maps compute the gradient of the output with respect to the input — cells with large gradients strongly influence the decision. For Connect Four, you'd expect high saliency near clusters of pieces (where threats exist) and low saliency in empty regions far from the action.

  **Why the input gradient is "attention."** The gradient ∂Q/∂x[i,j] tells you: if I perturb cell (i,j) of the board by an infinitesimal amount, how much does the network's output change? Cells where this gradient is large are cells the network's decision is *sensitive* to. Cells where it's near zero are cells the network has effectively ignored. That's not exactly the same as "the network is paying attention to that cell" in a cognitive sense — neural networks don't have attention in the human sense — but it's the best operational proxy we have. Caveat: saliency maps are notoriously noisy, and the literature has documented many ways they mislead (e.g., gradients can be near-zero through ReLU saturation, hiding real importance). Treat them as one signal, not ground truth.

### What to Build

1. **Diagnostic positions:** *"Create a script at notebooks/06_network_probing.py that defines a set of diagnostic Connect Four board positions: (1) an obvious winning move available, (2) an obvious block needed, (3) a fork opportunity (two ways to win), (4) an empty opening board, (5) a mid-game position with no immediate threats. For each position, load the trained DQN, REINFORCE, and PPO agents and print: the Q-values (DQN), action probabilities (REINFORCE/PPO), and value estimate (PPO's critic). Create a formatted table comparing all three agents' assessments of each position."*

2. **Value trajectory:** *"Add a function that plays a full game between two agents, recording the PPO value network's V(s) estimate at every step. Plot V(s) over the course of the game for both players' perspectives. The winning player's V(s) should trend toward +1 and the losing player's toward -1."*

3. **Saliency maps:** *"Add a function that computes a simple saliency map for a given board position and agent. For the DQN agent: compute the gradient of the selected action's Q-value with respect to the input tensor, take absolute values, and sum across the 3 input channels to get a 6x7 saliency grid. Display it as a heatmap overlaid on the board position. Do this for 3 interesting board positions."*

### After Running the Probes

1. **Diagnostic positions.** Focus on the fork position — this tests deep understanding. A fork means two ways to win, and the opponent can only block one. Does the DQN assign the highest Q-value to the column that creates the fork? Does the policy network assign it high probability? If not, the agent may not understand forks — it might be relying on simpler patterns.

2. **Value trajectory.** The shape of the V(s) curve is revealing. A well-trained network should show gradual changes in V(s) — not random noise. Look for "turning points" where V(s) shifts dramatically — these correspond to critical moves that changed the game's trajectory.

3. **Saliency maps.** Do the highlighted cells make sense? If the agent is deciding between column 3 and column 5, the saliency should be high around those columns and the pieces that form threats there. If the saliency is scattered randomly, the network may be relying on spurious correlations rather than genuine board understanding.

### Exercises

- [ ] Run the diagnostic probes. Create a written summary: for each diagnostic position, do all three agents agree on the best move? Where do they disagree? Who's right?
- [ ] **Find a blind spot.** Experiment with different board positions until you find one where an agent makes an obviously wrong choice. This is more valuable than confirming correct behavior — it reveals the boundary of what the agent learned.
- [ ] **Value trajectory experiment.** Play 10 games and plot all 10 V(s) curves overlaid. Games that the agent wins should trend up; games it loses should trend down. Is V(s) a reliable predictor of game outcome by move 10? By move 15?
- [ ] **Saliency interpretation.** For a board position where the agent must block an opponent's three-in-a-row, does the saliency map highlight the opponent's pieces? Or does it highlight the agent's own pieces? The answer reveals whether the network learned to "look at threats" or "look at opportunities."
- [ ] **Compare architectures.** If your DQN and PPO use different network sizes (they do — DQN has 4 residual blocks, PPO has 2 conv layers), do the saliency maps look different? Does the larger network focus on different features?

---

## Module 20 — Breaking Your Agent

**Goal:** Systematically find the boundaries of your agent's competence. This is the most important module in Part II — it turns optimistic win rates into honest assessments.

### Key Concepts

- **Distribution shift**: Your agents trained against specific opponents (random, heuristic, self-play). Every training run creates an implicit "distribution" of board states the agent has seen. States outside that distribution are **out-of-distribution** — the agent has never practiced there, and its behavior is unpredictable. This is the central problem of generalization in RL.

  **Why this is uniquely bad in RL.** In supervised learning, the data distribution is at least *fixed* — you train on a dataset, you test on held-out samples from the same distribution. In RL, the data distribution is *generated by the agent's own policy*. A DQN trained against HeuristicAgent will visit board states that arise from HeuristicAgent's particular moves. States that arise from an aggressive opponent's moves are systematically absent from training. So the agent isn't just blind to "unusual" inputs in a static sense — entire *classes* of valid board states never appeared during training because no opponent ever produced them.

- **Overfitting to opponents**: In supervised learning, overfitting means memorizing training data. In RL, overfitting means learning opponent-specific exploits rather than general strategy. An agent that beats the heuristic agent 90% of the time might have learned to exploit the heuristic's specific weaknesses (e.g., it always plays center when no threats exist) rather than learning good Connect Four play.

  A concrete picture: HeuristicAgent has a deterministic move selection given a board state (modulo tie-breaking). Train against it long enough and your agent doesn't need to learn "what is a winning move?" — it can learn "what move makes the heuristic play into my trap?" These are very different policies even though they have the same win rate. The first generalizes; the second collapses against any opponent that doesn't behave like HeuristicAgent.

- **Adversarial evaluation**: The strongest test is an opponent specifically designed to exploit the agent's weaknesses. You don't need a sophisticated adversary — a human looking at the agent's saliency map blind spots can craft positions where the agent fails.

- **The opponent we actually have.** This module's original draft called for a custom minimax agent with tunable search depth. The current repo's strongest non-learned opponent is [agents/optimal_agent.py](agents/optimal_agent.py) — a rule-based agent with threat detection, fork creation/blocking, trap avoidance, and positional scoring. It does *not* search the game tree, so there's no "depth" knob. What it does have is a `random_move_prob` parameter, which acts as a soft "difficulty" dial: probability 1.0 → behaves as RandomAgent, probability 0.0 → full strength. We'll use `random_move_prob` as the analog of search depth — lower randomness ≈ deeper search in terms of "how hard is this opponent to beat?"

### What to Build

1. **Cross-opponent evaluation:** *"Create notebooks/07_generalization_test.py that loads all trained checkpoints (DQN, REINFORCE, PPO — pick your best run of each from MLflow) and evaluates each against every opponent: RandomAgent, HeuristicAgent, OptimalAgent at random_move_prob ∈ {0.6, 0.3, 0.0}. Run 200 games per matchup. Produce a results matrix (rows = trained agent, columns = opponent) and render it as a heatmap. Save the matrix as an MLflow artifact tagged experiment='generalization_test'. Identify the largest single drop in win rate between the agent's training opponent and its worst evaluation opponent."*

2. **Disagreement / blind-spot search:** *"Add a function that plays 1,000 games between a trained agent and OptimalAgent at random_move_prob=0.0 (full strength). At each step, record the board position, the trained agent's chosen action, and OptimalAgent's recommended action. Filter to positions where they disagree. For each disagreement, also record OptimalAgent's evaluation of the trained agent's move (e.g., 'this move walks into a fork', 'this move misses a winning capture'). Display the top 10 disagreements ranked by severity — those are the agent's worst blind spots."*

3. **Optional stretch — build the minimax agent.** If you want the search-vs-learned-policy contrast, build `agents/minimax_agent.py` with alpha-beta pruning to configurable depth, and re-do the cross-opponent evaluation with depth-scaling. This is a worthwhile detour but not required for the rest of the chapter.

### After Running the Tests

1. **Cross-opponent matrix.** The diagonal (trained-opponent performance) should be good. The off-diagonal entries reveal generalization. Key questions:
   - Does the DQN trained against the heuristic still perform well against OptimalAgent at random_move_prob=0.0? If not, it likely learned heuristic-specific patterns rather than general Connect Four play.
   - Does an agent trained with curriculum learning (random_move_prob annealed from 0.8 → 0.0 via [training/curriculum.py](training/curriculum.py)) generalize better than one trained against the heuristic at a fixed difficulty?
   - How quickly does performance degrade as random_move_prob decreases from 0.6 to 0.0? A gradual decline suggests the agent learned general strategy. A cliff suggests it relies on the opponent making mistakes.

2. **Disagreement analysis.** The positions where your agent disagrees most with OptimalAgent are goldmines for understanding. Categorize them: does the agent miss blocks? Does it miss forks? Does it overvalue certain columns? Each pattern reveals a gap in training.

### Exercises

- [ ] Verify the opponent hierarchy: OptimalAgent at random_move_prob=0.0 should beat HeuristicAgent decisively. Confirm by running a 200-game match between the two.
- [ ] Run the cross-opponent evaluation. Find the single matchup where performance drops the most. Hypothesize why.
- [ ] **Find the exploit.** Using the disagreement positions as hints, play against your trained agent yourself (via [play.py](play.py)). Target its weak spots. How many games does it take you to develop a strategy that consistently beats it?
- [ ] **Experiment — difficulty scaling.** Evaluate your best agent against OptimalAgent at random_move_prob ∈ {0.9, 0.7, 0.5, 0.3, 0.1, 0.0}. Plot win rate vs. (1 − random_move_prob). At what difficulty does the agent drop below 50%? This is a proxy for "how much opponent skill can your agent handle."
- [ ] **Predict before running:** Which of your three trained agents (DQN, REINFORCE, PPO) do you think will generalize best to OptimalAgent? Write your prediction and reasoning before running the tests. Were you right?
- [ ] **Thought experiment.** If you trained against *all* possible opponents (random, heuristic, OptimalAgent at every difficulty, self-play, hand-crafted aggressive opponents), would your agent be robust? What opponent could still beat it? (Hint: Connect Four is solved — perfect play from player 1 always wins.)
- [ ] **Stretch:** Build the minimax agent and add it to the evaluation. Compare its results to OptimalAgent. Does a true game-tree search reveal blind spots that the rule-based OptimalAgent misses?

---

## Module 21 — Training for Robustness

**Goal:** Use the weaknesses you discovered in Module 20 to train a more robust agent. This module touches the training infrastructure more than any other — you'll add PPO self-play and a pool-of-opponents training mode to [training/train.py](training/train.py).

### Key Concepts

- **Opponent pool (population-based training)**: Instead of training against one opponent, maintain a pool of diverse opponents and sample from them each episode. Diversity forces the agent to learn general strategy rather than opponent-specific exploits.

  **Why diversity prevents exploits, intuitively.** A policy that beats Opponent A by exploiting its specific weaknesses must, by definition, produce moves that are *better against A* than the moves a more general policy would produce. If the agent is only ever trained against A, gradient descent finds these exploit-moves and reinforces them. If the agent is trained against A and B and C, the exploit-moves against A *hurt* its performance against B and C — they no longer correspond to good Connect Four play, just to A's quirks. Gradient descent now has to find moves that are *acceptable everywhere*, and that pressure pushes the policy toward general competence.

- **Opponent sampling strategy**: Not all opponents in the pool are equally useful. Training against very weak opponents (random) wastes episodes — the agent already beats them. Training against much stronger opponents may be too hard to learn from. **Prioritized sampling** — spending more time against opponents at the agent's current skill boundary — accelerates learning.

  This is the same idea as *curriculum learning* from cognitive science: you learn fastest from problems that are slightly above your current skill (Vygotsky's "zone of proximal development"). Problems too easy provide no gradient signal; problems too hard provide a gradient that's mostly noise. The sweet spot is opponents you beat ~50% of the time.

- **The curriculum trade-off**: Population training is slower per-episode than single-opponent training (the agent must generalize, which is harder). But the resulting agent is more robust. This is a training-time vs. deployment-time trade-off — spend more compute during training to get a better final agent.

- **Measuring robustness, not just performance**: Win rate against a single opponent is a fragile metric. Define robustness as the **minimum win rate across a diverse set of opponents**. An agent with 95% vs. random and 30% vs. strong opponents is less robust than an agent with 80% vs. random and 60% vs. strong, even though the first has a higher average.

  This is a deep methodological point. Optimizing for *average* performance pushes you to specialize where it's easy to win. Optimizing for *minimum* performance pushes you to plug your weakest holes. The two objectives produce different agents. For deployment against unknown opponents, the minimum-win-rate floor is almost always the metric that matters — it's the worst-case guarantee.

### What to Build

This module has three build steps. Steps 1 and 2 modify [training/train.py](training/train.py); step 3 is a comparison notebook.

#### Step 1 — Add PPO self-play

The unified trainer currently rejects `--opponent self-play` for PPO ([training/train.py:124](training/train.py#L124)). DQN already has it. To make Module 21's comparison meaningful, PPO needs self-play too.

**Suggested prompt:** *"In training/train.py, lift the restriction that --opponent self-play only works with --algorithm dqn. Implement PPO self-play: maintain a frozen copy of the current policy network as the opponent. Update the frozen copy every N episodes (CLI flag --frozen-update-interval, default 2000). The rollout collection should still attribute trajectories to the current agent only — the frozen opponent's transitions are not used for gradient updates. Make sure the rollout buffer's reward/value bookkeeping handles the opponent's moves correctly (the opponent's move is part of the environment from the agent's perspective). Add an MLflow tag opponent='self-play'."*

#### Step 2 — Extend the trainer with a `pool` opponent type

**Suggested prompt:** *"In training/opponents.py, add a new opponent type 'pool'. A PoolOpponent wraps a list of (agent, weight) tuples and exposes select_action(state, valid_actions) by sampling one agent per episode and delegating. Sampling can be uniform (initial implementation) or prioritized. Add a --pool-config CLI flag that takes a comma-separated list like 'random:1,heuristic:1,optimal_06:1,optimal_03:1,selfplay:1' where the suffix after the colon is the initial sampling weight. For 'selfplay' entries, instantiate a frozen copy of the current network and update it on the same --frozen-update-interval as Step 1. In training/train.py, when --opponent pool, log per-opponent win rates separately to MLflow (e.g., 'eval/win_rate/heuristic', 'eval/win_rate/optimal_06')."*

#### Step 3 — Prioritized sampling

**Suggested prompt:** *"Add a --pool-sampling {uniform,prioritized} flag. For prioritized: every K episodes (CLI flag --priority-update-interval, default 200), recompute sampling weights as inversely proportional to recent win rate against each opponent. Specifically: weight_i = max(0.05, 1 - win_rate_i). Floor at 0.05 to prevent the agent from forgetting opponents it has fully solved. Normalize. Log the current weights to MLflow each time they update."*

#### Step 4 — Robustness comparison notebook

**Suggested prompt:** *"Create notebooks/08_robustness_comparison.py. Load four trained PPO models from MLflow: (a) PPO vs. HeuristicAgent (single opponent), (b) PPO vs. OptimalAgent with curriculum (single opponent, annealed), (c) PPO self-play (frozen copy), (d) PPO pool-trained (Step 2 above). Evaluate all four against the same opponent suite from Module 20: Random, Heuristic, Optimal at random_move_prob ∈ {0.6, 0.3, 0.0}. Report a results matrix and compute two metrics per agent: average win rate and minimum win rate. Render a radar chart with one polygon per agent, each axis being one opponent. Round shape = robust; spiky shape = specialized."*

### After Running the Experiment

1. **Uniform vs. prioritized sampling.** Prioritized sampling should show faster improvement against hard opponents but potentially slower improvement against easy ones (fewer episodes against them). Is the overall robustness better? Watch for the cliff: if the priority weights become too aggressive, easy opponents are sampled almost never, and the agent forgets basic tactics (catastrophic forgetting). The 0.05 floor is meant to prevent this.

2. **Pool training vs. self-play.** Self-play is a special case of population training (population = 1 evolving opponent). Does the diverse pool produce a more robust agent than pure self-play? The answer isn't guaranteed — self-play has its own strengths (automatic curriculum, no need to design opponents). On the other hand, self-play famously has *cycles*: rock-paper-scissors strategies where each iteration beats the last but a much earlier policy beats the current one. A diverse pool keeps the older strategies in scope.

3. **The robustness metric.** Compare agents by minimum win rate. Which training approach produces the highest floor? This is the metric that matters if the agent will face unknown opponents.

### Exercises

- [ ] Run pool training with uniform sampling. Does the agent learn to beat all opponents, or does it "forget" how to beat earlier opponents when facing harder ones? (This is **catastrophic forgetting** — a deep problem in multi-task learning.)
- [ ] Run pool training with prioritized sampling. Compare the learning curves to uniform sampling. Which converges faster overall? Use the per-opponent MLflow metrics to see which opponents the prioritizer spends time on at each stage of training.
- [ ] **Build the radar chart.** Visually compare the "shape" of each agent's competence. A robust agent should have a round shape (consistent across opponents). A specialized agent should have spikes (great against some, weak against others).
- [ ] **Experiment — remove the weakest opponent.** Train the pool without RandomAgent. Does this hurt or help? Random play may be too easy to provide useful gradient signal — or it might prevent catastrophic forgetting of basic tactics like "block the obvious three-in-a-row."
- [ ] **Experiment — add the strongest opponent.** Train with `optimal_00` (random_move_prob=0.0, full strength) in the pool. Is this helpful, or is it too strong for the agent to learn from in the early stages? Prioritized sampling should answer this dynamically: if the agent loses every game against `optimal_00`, the prioritizer keeps weight high and the agent burns episodes there with little progress; uniform sampling at least spends some time on opponents the agent can actually beat.
- [ ] **Final reflection.** Write a paragraph: what is the relationship between the bootstrapping depth (λ, n) you studied in Part I and the generalization behavior you observed in Part II? Does the choice of λ affect robustness to new opponents? Think about whether biased value estimates (low λ) or high-variance estimates (high λ) are worse when facing out-of-distribution states. (Hypothesis to test: high-bias value estimates may be more confidently *wrong* on OOD states than high-variance estimates, because the bias has been carved into the network by training. High variance is at least honest noise.)

---

## Module 22 — Synthesis: Connecting Theory to Practice

**Goal:** Consolidate everything from this chapter into a framework you can apply to any RL problem.

### No Code This Module

This is a reflection and synthesis module. The exercises are all writing and thinking.

### The Decision Framework

When approaching a new RL problem, you now know to ask:

1. **What is the episode structure?** Short or long? Sparse or dense reward? This determines your bootstrapping depth (n, λ).

2. **How expensive is data collection?** If expensive, use off-policy methods (DQN + replay buffer) to reuse experience. If cheap, on-policy methods (PPO) avoid staleness issues.

3. **What is the action space?** Discrete → value-based is viable. Continuous → policy-based or actor-critic.

4. **Who is the opponent?** Fixed and known → single-opponent training is efficient. Unknown or diverse → population-based training for robustness.

5. **How will you evaluate?** Win rate against one opponent is not enough. Define a robustness metric. Probe the network. Find the blind spots before deployment.

### Exercises

- [ ] **Write the retrospective.** Looking back at all three chapters, what was the single most surprising thing you learned? What concept seemed obvious in hindsight but wasn't obvious when you first encountered it? Skim [bugfixes/](bugfixes/) and [enhancements/](enhancements/) before writing — the PPO action-masking bug, the PPO entropy NaN bug, and the entropy-recovery patch are all real incidents from this project where the right diagnosis required exactly the kind of theoretical vocabulary this chapter builds. Reference at least one specifically.

- [ ] **Design document.** A teammate wants to build an RL agent for a new game: 8x8 Othello. Episodes last ~60 moves, reward is +1/-1 at the end (sparse), the action space is discrete (place a piece on a valid square, ~10 options per turn), and they want it to be robust against human players. Write a one-page design document recommending: (1) algorithm choice and why, (2) bootstrapping depth (λ or n) and why, (3) network architecture and why, (4) training curriculum and why, (5) evaluation strategy. Use the vocabulary and framework from this chapter.

- [ ] **Predict the failure mode.** For each of these hypothetical changes, predict what would go wrong and why:
  - Switch DQN from TD(0) to Monte Carlo returns while keeping the replay buffer (hint: the staleness story from Module 16 — MC returns are *entirely* on-policy, so storing them in a replay buffer breaks the whole premise)
  - Train PPO with λ=0 (pure TD advantage estimation) on Connect Four (hint: sparse terminal reward + 1-step bootstrap → reward signal has to crawl backward through training, very slowly)
  - Use population-based training but only sample from opponents the agent already beats 90%+ (hint: zone of proximal development — no gradient signal where you already win)
  - Remove the entropy bonus from self-play PPO training (hint: see [enhancements/entropy_recovery.md](enhancements/entropy_recovery.md) — there is real precedent in this repo)

- [ ] **The meta-question.** You used Claude Code to generate most of the implementation in this project. What did you learn that you *couldn't* have learned just by reading the generated code? What required running experiments, breaking things, and observing results? This distinction — between reading knowledge and operational knowledge — is at the heart of why this curriculum asks you to experiment, not just implement.

---

## Appendix — Recommended Resources

| Resource | What It Covers |
|----------|---------------|
| [Sutton & Barto, Chapter 7](http://incompleteideas.net/book/RLbook2020.pdf) | N-step bootstrapping — the definitive theoretical treatment |
| [Sutton & Barto, Chapter 12](http://incompleteideas.net/book/RLbook2020.pdf) | Eligibility traces — forward and backward views, the mathematical equivalence |
| [High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438) (Schulman et al., 2015) | The GAE paper — directly connects to PPO's advantage estimation |
| [The Deadly Triad](https://www.andrew.cmu.edu/course/10-703/textbook/BarsutR2018Draft.pdf) (Sutton & Barto, Ch. 11.3) | Why function approximation + bootstrapping + off-policy = instability |
| [Visualizing and Understanding Atari Agents](https://arxiv.org/abs/1711.00138) (Greydanus et al., 2017) | Saliency maps for RL agents — the technique from Module 19 |
| [Robust Adversarial RL](https://arxiv.org/abs/1703.02702) (Pinto et al., 2017) | Training robust agents via adversarial opponents |
| [David Silver's RL Lectures — Model-Free Prediction](https://www.davidsilver.uk/teaching/) (Lecture 4) | MC vs. TD prediction — the core theory of Part I |

---

## Full Project Structure (End of Chapter 3)

This reflects the actual repo layout, not the originally-imagined one.

```
c4/
├── pyproject.toml
├── game.md
├── lesson_plan.md                    # Chapter 1
├── lesson_plan_chapter_2.md          # Chapter 2
├── lesson_plan_chapter_3.md          # This file
├── lesson_plan_chapter_4.md          # Chapter 4 (MLflow + Databricks — already shipped)
├── game_runner.py
├── play.py
├── env/
│   ├── __init__.py
│   ├── connect_four.py
│   └── encoding.py
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   ├── human_agent.py
│   ├── optimal_agent.py              # Strong rule-based opponent (used in Module 20/21)
│   ├── dqn_agent.py                  # Chapter 1
│   ├── reinforce_agent.py            # Chapter 2
│   ├── ppo_agent.py                  # Chapter 2
│   └── minimax_agent.py              # Optional stretch in Module 20 — not required
├── models/
│   ├── __init__.py
│   ├── registry.py
│   ├── simple_cnn.py
│   ├── q_network.py                  # Chapter 1
│   ├── policy_network.py             # Chapter 2
│   └── value_network.py              # Chapter 2
├── training/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── device.py
│   ├── mlflow_utils.py               # Chapter 4
│   ├── plotting.py
│   ├── eval.py
│   ├── evaluate.py
│   ├── opponents.py                  # Opponent factory; gets a 'pool' type in Module 21
│   ├── curriculum.py                 # Difficulty annealing for OptimalAgent
│   ├── reward_shaping.py
│   ├── n_step_buffer.py              # Used in Module 16
│   ├── replay_buffer.py              # Chapter 1
│   ├── prioritized_replay_buffer.py
│   ├── rollout_buffer.py             # Chapter 2 (GAE = λ-return; see Module 17)
│   └── train.py                      # Unified trainer for DQN/REINFORCE/PPO
├── notebooks/
│   ├── 01_pytorch_basics.py          # Chapter 1
│   ├── 02_training.ipynb             # Chapter 2
│   ├── 03_reinforce_training.ipynb   # Chapter 2
│   ├── reinforce_baseline_comparison.py  # Chapter 2 baseline study
│   ├── 04_nstep_comparison.py        # Module 16 — n-step sweep + MLflow plots
│   ├── 05_td_lambda_prediction.py    # Module 17 — TD(λ) + traces
│   ├── 06_network_probing.py         # Module 19 — diagnostic positions / saliency
│   ├── 07_generalization_test.py     # Module 20 — cross-opponent eval
│   └── 08_robustness_comparison.py   # Module 21 — radar charts
├── bugfixes/
│   ├── ppo_action_masking.md
│   └── ppo_entropy_nan.md
├── enhancements/
│   └── entropy_recovery.md
├── future/
│   └── multi_opponent_training.md    # Pool-training design notes (Module 21)
├── checkpoints/
├── mlruns/                           # MLflow tracking store
├── mlflow.db
└── tests/
```
