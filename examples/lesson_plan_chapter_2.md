# Connect Four RL Agent — Chapter 2: Policy Gradient Methods

A progressive curriculum for building a **policy gradient** agent that learns to play Connect Four. Chapter 1 taught you value-based RL (DQN) — the agent learned Q-values and derived a policy from them. This chapter takes the other road: learning the policy *directly*. You'll build a REINFORCE agent, understand why baselines matter, then graduate to PPO — the workhorse of modern RL. Along the way, you'll develop intuition for when and why policy gradients outperform value methods.

You already have a working environment, game runner, evaluation harness, and baseline agents. This chapter reuses all of them. The new code lives alongside the DQN agent, not on top of it — by the end, you'll be able to pit your DQN against your PPO agent and compare what each one learned.

---

## Module 7 — Why Policy Gradients?

**Goal:** Understand the fundamental difference between value-based and policy-based RL, and why learning a policy directly is sometimes the better approach.

### Key Concepts

Read and understand these *before* you move on to implementation. They frame everything in this chapter.

- **Value-based vs. policy-based**: DQN learns a value function Q(s, a) and derives a policy from it (pick the action with the highest Q). Policy gradient methods skip the middleman — they directly learn a function π(a|s) that outputs the probability of taking each action in a given state.
- **Stochastic policy**: Instead of always picking the single best action, a stochastic policy outputs a *probability distribution* over actions. The agent *samples* from this distribution. This may feel wasteful compared to a deterministic greedy policy, but it has a deep advantage: exploration is built into the policy itself rather than bolted on with epsilon-greedy.
- **Policy parameterization**: The policy is a neural network with parameters θ. The output layer uses a **softmax** to produce valid probabilities over the 7 columns. The goal of training is to adjust θ so the network assigns higher probability to good moves and lower probability to bad ones.
- **Why not just use DQN?**: DQN works well for Connect Four, but it has limitations. Q-learning can overestimate values (you saw this with Double DQN). It requires a replay buffer and a target network to stabilize training. And it can only represent deterministic policies — the agent always picks the max-Q action at test time. Policy gradient methods can represent stochastic policies naturally, often have simpler implementations (REINFORCE is ~50 lines), and extend more naturally to continuous action spaces (not relevant for Connect Four, but important for robotics and game AI).

### No Code This Module

This module is conceptual. Make sure you can answer these questions before moving on:

1. In DQN, the network outputs 7 Q-values. In a policy network, the output is also 7 numbers — but what do they represent, and what constraint must they satisfy?
They represent the probability of picking each action. They must form a valid probability distribution: every value must be in [0, 1] and all 7 must sum to exactly 1. This is what softmax enforces — it takes raw logits and turns them into non-negative values that sum to 1.
<!-- KEY CONCEPT: Softmax as the bridge between raw network output and a valid probability distribution. The network outputs unconstrained logits; softmax is the function that imposes the probability constraints. -->
2. DQN uses epsilon-greedy for exploration. How does a stochastic policy explore *without* any explicit exploration mechanism?
Actions are sampled based on their probability distribution. Even low-probability actions get selected sometimes, so the agent naturally tries different moves without needing an epsilon parameter. As the model improves and concentrates probability on better actions, exploration is reduced organically.
<!-- KEY CONCEPT: Exploration is intrinsic to stochastic policies. In DQN, exploration (epsilon-greedy) is bolted on externally and requires tuning a schedule. In policy gradient methods, exploration emerges from the probability distribution itself — no extra mechanism needed. -->
3. If a policy network assigns probability 0.9 to column 3 and 0.1 spread across the others, is it still "exploring"? When would it stop exploring entirely?
Yes, the model is still exploring — it plays a non-column-3 action 10% of the time (each of the 6 other columns gets ~1.67% probability). It would stop exploring entirely only if the policy became fully deterministic, i.e., assigned probability 1.0 to a single action and 0.0 to everything else. In practice with softmax this essentially never happens — softmax always assigns some nonzero probability to every (valid) action, so a stochastic policy never fully stops exploring.
<!-- KEY CONCEPT: Softmax outputs are always > 0 for finite logits. True zero probability only happens via masking (invalid actions) or if a logit goes to -infinity. This means stochastic policies have a built-in exploration floor, unlike epsilon-greedy where you explicitly set epsilon=0 to stop exploring. -->

### Exercises

- [ ] **Thought experiment:** You have a board where column 3 is the only winning move. What should Q(s, 3) look like in a DQN? What should π(3|s) look like in a policy network? Now imagine a board where columns 2 and 4 are *equally* good. How does each approach handle this?
**Single winning move (column 3):** Q(s, 3) should be high (close to +1, the win reward). The other Q-values shouldn't be zero — they should reflect the actual expected outcome of those moves (likely negative, since not playing the winning move lets the opponent gain advantage). π(3|s) should be very close to 1.0, with the remaining probability mass spread thinly across other actions.

**Equally good columns 2 and 4:** This is where the approaches differ meaningfully. DQN computes Q(s,2) ≈ Q(s,4) — both high and roughly equal. But the derived policy is deterministic (argmax), so it always picks whichever has the slightly higher Q-value due to noise. DQN cannot naturally express "both are equally good — randomize between them." A policy network handles this naturally: π(2|s) ≈ π(4|s) ≈ 0.5, directly representing the ambiguity. This is a core advantage of stochastic policies.
<!-- KEY CONCEPT: Stochastic policies can represent multimodal action preferences (multiple equally good actions). DQN's argmax collapses this to a single choice. This matters in games where flexibility and unpredictability are strategic advantages. -->
- [ ] **Write it down:** In your own words, explain one advantage and one disadvantage of policy gradient methods compared to DQN. Refer back to this at the end of the chapter to see if your understanding changed.
Advantage: stochasticity baked in, no hyperparameter for exploration rate. The policy naturally explores and can represent uncertainty over equally good actions.
Disadvantage: High variance in gradient estimates — REINFORCE reinforces *all* actions in a winning game (even bad ones that happened to precede a win), so learning is noisy and sensitive to hyperparameters like learning rate. Also less sample-efficient than DQN since episodes are used once and discarded (no replay buffer).
<!-- KEY CONCEPT: The variance problem is the central weakness of policy gradients and the motivation for everything in Modules 10-11 (baselines, advantage estimation, PPO clipping). Keep this in mind — it's the thread that ties the whole chapter together. -->

---

## Module 8 — The Policy Network

**Goal:** Build a neural network that outputs action probabilities instead of Q-values. Understand how masking and softmax interact.

### Key Concepts

- **Softmax output**: The final layer produces 7 raw scores (logits). Softmax converts them to probabilities that sum to 1. This is how the network parameterizes a valid probability distribution.
- **Log-probabilities**: In practice, you'll work with `log_softmax` instead of `softmax` for numerical stability. PyTorch's `Categorical` distribution handles this, but you should understand *why* — multiplying many small probabilities together underflows to zero; summing log-probabilities avoids this.
- **Action masking for policy networks**: Invalid columns must have probability 0. The approach is the same as DQN — set logits to `-inf` before softmax — but the interpretation is different. In DQN, masking prevents selecting a bad action. In a policy network, masking ensures zero probability mass is wasted on impossible moves, which makes the gradient signal cleaner.

### Network Architecture

```
Input: Board state as tensor, shape (batch, 3, 6, 7)
       Same 3-channel encoding from Chapter 1 (reuse encode_state)

→ Conv2d(3, 64, kernel_size=4, padding=1)  → ReLU
→ Conv2d(64, 64, kernel_size=3, padding=1) → ReLU
→ Flatten
→ Linear(*, 128) → ReLU
→ Linear(128, 7)                            ← raw logits

At inference:
  Mask invalid actions → Softmax → Sample from distribution

At training:
  Mask invalid actions → Log-softmax → Use log-probs in loss
```

Notice this is nearly identical to the Q-network. The only difference is what the output *means* and how it's used. This is the point — the architecture is secondary to the algorithm.

### What to Ask Claude to Build

**Suggested prompt:** *"Create a PolicyNetwork class in models/policy_network.py. Use the same convolutional architecture as models/q_network.py (two conv layers followed by two linear layers), but the output represents action logits instead of Q-values. Include a method get_action_probs(x, valid_actions) that masks invalid actions by setting their logits to negative infinity, then applies softmax to return a valid probability distribution. Also include a method get_log_probs(x, valid_actions) that returns masked log-probabilities using log_softmax. Use the existing encode_state function from env/encoding.py for state preprocessing."*

### After Claude Generates the Code

1. **Compare it to Q-network side by side.** The layers should be identical. The difference is downstream — how the output is interpreted and used.
2. **Feed a random board through the network.** Call `get_action_probs()` and verify: do the probabilities sum to 1? Are invalid columns exactly 0?
3. **Test the masking.** Create a board where only 2 columns are valid. Feed it through the network. The 5 invalid columns should have probability exactly 0, and the 2 valid columns should sum to 1.
4. **Ask Claude:** *"Why do we mask before softmax rather than after? What would go wrong if we applied softmax first and then zeroed out invalid actions?"*

### Exercises

- [ ] Run a forward pass on 5 different board states. Print the probability distribution for each. Do any columns consistently get higher probability? (They shouldn't — the network is randomly initialized.)
- [ ] **Experiment:** What happens if you don't mask invalid actions? Feed a board with a full column through the unmasked network. What probability does the full column get? Why is this dangerous?
- [ ] **Predict before running:** If you initialize two PolicyNetwork instances with different random seeds, will they produce the same probabilities for the same board? Why or why not?

### Suggested File Structure

```
c4/
├── models/
│   ├── q_network.py           # From Chapter 1 (unchanged)
│   └── policy_network.py      # New: PolicyNetwork class
```

---

## Module 9 — REINFORCE

**Goal:** Implement the simplest policy gradient algorithm. Understand the policy gradient theorem, why it works, and where it struggles.

### Key Concepts — The Policy Gradient Theorem

This is the theoretical core of the entire chapter. Take time with it.

- **Objective**: Maximize the expected total reward: `J(θ) = E[R₁ + R₂ + ... + Rₜ]` where the expectation is over trajectories generated by policy πθ.
- **The key insight**: We can estimate the *gradient* of this objective without knowing the environment dynamics. The policy gradient theorem says:

  ```
  ∇J(θ) = E[ Σₜ ∇log π(aₜ|sₜ; θ) · Gₜ ]
  ```

  where `Gₜ = Rₜ + Rₜ₊₁ + ... + Rₜ` is the **return** — the total reward from time step t onward.

- **In plain language**: For each action the agent took, compute how much total reward followed. If the return was high, increase the probability of that action. If the return was low (or negative), decrease it. The `∇log π` term tells the optimizer *how* to push the probabilities, and `Gₜ` tells it *how much* to push.
- **Return** `Gₜ`: The sum of rewards from step t to the end of the episode. In Connect Four, rewards are sparse — only the final step gets +1 or -1. So `Gₜ = γ^(T-t) · R_final` where T is the episode length and t is the current step. Earlier moves get a smaller (more discounted) signal.
- **Monte Carlo**: REINFORCE is a Monte Carlo method — it waits until the episode is *complete* before computing returns and updating the policy. No bootstrapping, no replay buffer. Simple but high variance.

### What to Ask Claude to Build

Build in stages:

1. **REINFORCE agent:** *"Create a ReinforceAgent class in agents/reinforce_agent.py that extends the Agent base class from agents/base.py. It should use the PolicyNetwork from models/policy_network.py. During an episode, it stores (log_prob, reward) pairs for each step. At the end of the episode, it computes discounted returns, calculates the policy gradient loss as -sum(log_prob * return), and updates the network. Use gamma=0.99 and learning_rate=1e-3. Include an update() method that is called once per episode after the episode completes. Don't use a replay buffer — REINFORCE processes each episode once and discards it."*

2. **Training loop:** *"Create training/train_reinforce.py that trains the REINFORCE agent by playing games against the heuristic agent. The structure should mirror training/train.py but adapted for REINFORCE: collect a full episode, then call agent.update() once. Log average return and win rate. Evaluate every 500 episodes using the same evaluate functions. Use the existing checkpoint infrastructure from training/checkpoint.py to save models."*

### After Claude Generates the Code

1. **Find the loss computation.** This is the heart of REINFORCE. You should see something like:
   ```python
   loss = 0
   for log_prob, G in zip(saved_log_probs, returns):
       loss -= log_prob * G
   ```
   Make sure you can explain every term: what `log_prob` is, what `G` is, and why the negative sign is there (gradient *ascent* on reward, implemented as gradient *descent* on negative reward).

2. **Trace the return computation.** Starting from the last step, returns are computed backwards:
   ```python
   G = 0
   for r in reversed(rewards):
       G = r + gamma * G
   ```
   For a 20-move Connect Four game that ends in a win (+1), what is the return for the very first move? What about the last move? Does the discounting feel appropriate?

3. **Compare to DQN training.** DQN trains on individual transitions sampled from a buffer. REINFORCE trains on *entire episodes* and then throws them away. Ask Claude: *"Why can't REINFORCE reuse old episodes like DQN does with its replay buffer?"*

### Hyperparameters (Starting Points)

| Parameter              | Value     |
|------------------------|-----------|
| Learning rate          | 1e-3      |
| Discount factor (γ)    | 0.99      |
| Max episodes           | 20,000    |
| Eval interval          | 500       |

### Exercises

- [ ] Train REINFORCE vs. the random agent for 5,000 episodes. Plot the win rate over time. Does it learn? How does the learning curve compare to DQN from Chapter 1?
- [ ] **Inspect the gradient signal:** After a winning 20-move game, print the return assigned to each move. The last move should be close to 1.0. The first move should be ~0.99^19 ≈ 0.83. Is this a reasonable credit assignment?
- [ ] **Observe the variance:** Run 3 training runs with different random seeds. Plot all 3 win rate curves on the same graph. How much do they differ? This variance is REINFORCE's biggest weakness.
- [ ] **Experiment — learning rate sensitivity:** Try learning rates of 1e-2, 1e-3, and 1e-4. Which learns fastest? Which is most stable? REINFORCE is notoriously sensitive to this.
- [ ] **Break it to understand it:** Set gamma to 0.0. Now every move except the very last one gets zero return. What happens to training? Why?
- [ ] Train REINFORCE vs. the heuristic agent. Is it harder than vs. random? Compare the win rate curve.

---

## Module 10 — Baselines and Variance Reduction

**Goal:** Understand why raw REINFORCE has high variance and fix it with a baseline. This module bridges REINFORCE and actor-critic methods.

### Key Concepts

- **The variance problem**: REINFORCE updates work, but the gradient estimate is extremely noisy. In Connect Four, the return for every move in a winning game is positive, so *all* moves get reinforced — even bad ones that happened to precede a win. Over many episodes, the good moves get reinforced *more* than the bad ones, so learning happens — but it's slow and unstable.
- **Baseline**: Subtract a baseline `b(s)` from the return: `∇J ≈ Σ ∇log π(a|s) · (Gₜ - b(sₜ))`. This doesn't change the expected gradient (the math works out), but it *dramatically* reduces variance. Moves that got higher-than-expected returns get reinforced. Moves that got lower-than-expected returns get penalized. The signal is now about *relative* quality, not absolute.
- **Value function as baseline**: The best baseline is an estimate of the expected return from state s — the **value function** V(s). If we have a good V(s), then `Gₜ - V(sₜ)` is called the **advantage**: how much *better* this action was compared to what we'd expect on average.
- **Advantage** `A(s, a) = Gₜ - V(sₜ)`: Positive advantage means "this action was better than average." Negative means "worse than average." This is a much cleaner learning signal than raw returns.
- **Actor-Critic**: When you use a learned value function as the baseline, you have two networks — the **actor** (policy) and the **critic** (value function). The critic estimates V(s), the actor uses the advantage to update the policy. This is the actor-critic framework, and it's the foundation for PPO.

### What to Ask Claude to Build

1. **Value network:** *"Create a ValueNetwork class in models/value_network.py. Use the same convolutional architecture as PolicyNetwork but with a single scalar output instead of 7. The output should be an estimate of V(s) — the expected return from state s. Use a tanh activation on the output to bound it to [-1, 1] since Connect Four returns are in that range."*

2. **REINFORCE with baseline:** *"Update agents/reinforce_agent.py to add a ValueNetwork as a baseline. Add a flag use_baseline=True that defaults to on. In the update() method: (1) compute returns as before, (2) estimate V(s) for each state using the value network, (3) compute advantages as returns - V(s), (4) use advantages instead of raw returns in the policy loss, (5) train the value network with MSE loss between its predictions and the actual returns. Use a separate optimizer for the value network with learning rate 1e-3."*

3. **Comparison script:** *"Create a notebook or script at notebooks/reinforce_baseline_comparison.py that trains REINFORCE with and without the baseline for 10,000 episodes each, then plots both win rate curves on the same graph. Use the same random seed for the environment so the comparison is fair."*

### After Claude Generates the Code

1. **Inspect the advantage values.** During a training episode, print the return, V(s), and advantage for each move. Early in training, V(s) will be near zero (untrained), so advantages will look like raw returns. After a few thousand episodes, V(s) should be more accurate, and advantages should be centered around zero.
2. **Watch the value network learn.** For a nearly-won board state, V(s) should be close to +1. For a nearly-lost state, close to -1. For an opening-move state, close to 0 (uncertain). Check these after training.
3. **Ask Claude:** *"Mathematically, why doesn't subtracting the baseline change the expected gradient? Show me the proof sketch."*

### Exercises

- [ ] Run the comparison script. The baseline version should learn faster and with less variance. Quantify: what's the win rate after 5,000 episodes with vs. without the baseline?
- [ ] **Probe the value network:** After training, feed it the empty board (game start). What value does it predict? Feed it a board where the agent is about to win. What about a board where the agent is about to lose? Do these make sense?
- [ ] **Experiment — constant baseline:** Instead of a learned V(s), try subtracting a constant (e.g., 0.0 or the running average return). Does this help compared to no baseline? How does it compare to the learned baseline?
- [ ] **Predict before running:** If you make the value network much larger (4 conv layers, 256 hidden units) while keeping the policy network small, will training improve? Think about what happens when the baseline is very accurate vs. inaccurate.

### Suggested File Structure

```
c4/
├── models/
│   ├── q_network.py           # Chapter 1
│   ├── policy_network.py      # Module 8
│   └── value_network.py       # New: V(s) estimator
├── agents/
│   ├── dqn_agent.py           # Chapter 1
│   └── reinforce_agent.py     # Updated: now supports baseline
└── notebooks/
    ├── 01_pytorch_basics.py   # Chapter 1
    └── 02_reinforce_baseline_comparison.py  # New
```

---

## Module 11 — Proximal Policy Optimization (PPO)

**Goal:** Build a PPO agent — the most widely used policy gradient algorithm in practice. Understand the problems it solves and the mechanics of clipped surrogate objectives.

### Key Concepts — What Goes Wrong Without PPO

Before learning *what* PPO does, understand *why* it's needed:

- **The step-size problem**: REINFORCE (even with a baseline) computes a gradient and takes a step. But how big a step? Too small and learning is slow. Too large and the policy changes drastically — actions that were rare become common, and the training data (collected under the *old* policy) is no longer representative. This can cause catastrophic performance collapse.
- **Trust region intuition**: The idea is to limit *how much the policy can change* in a single update. If the old policy said "play column 3 with 60% probability" and the gradient says "increase column 3," we want to nudge it to maybe 65%, not jump to 99%. PPO enforces this constraint cheaply.

### Key Concepts — PPO Mechanics

- **Probability ratio** `r(θ) = π_new(a|s) / π_old(a|s)`: How much more (or less) likely the action is under the updated policy compared to the policy that collected the data. If r = 1, the policy hasn't changed. If r = 2, the action is now twice as likely.
- **Clipped surrogate objective**:
  ```
  L = min(r(θ) · A, clip(r(θ), 1-ε, 1+ε) · A)
  ```
  The `clip` prevents the ratio from going too far from 1. With ε = 0.2, the ratio is clamped to [0.8, 1.2]. If the advantage is positive (good action), the objective incentivizes increasing the probability — but only up to 1.2× the old probability. If the advantage is negative (bad action), the probability is decreased — but only down to 0.8× the old probability. This is the "proximal" part of PPO.
- **Multiple epochs**: Unlike REINFORCE, which uses each episode once, PPO can replay the same batch of episodes for several gradient steps (typically 3–10 epochs). The clipping ensures these repeated updates don't push the policy too far.
- **Generalized Advantage Estimation (GAE)**: A more sophisticated way to compute advantages that balances bias and variance:
  ```
  δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)          (TD residual)
  Aₜ = δₜ + (γλ)·δₜ₊₁ + (γλ)²·δₜ₊₂ + ...
  ```
  The λ parameter (typically 0.95) controls the trade-off. λ = 1 gives Monte Carlo returns (high variance, no bias). λ = 0 gives one-step TD (low variance, high bias). GAE interpolates between the two.
- **Entropy bonus**: Add the policy's entropy to the objective to encourage exploration. High entropy = spread-out probabilities = more exploration. The bonus decays over training as the policy becomes more confident.

### What to Ask Claude to Build

Build PPO in stages. Each piece has a clear purpose.

1. **Rollout buffer:** *"Create a RolloutBuffer class in training/rollout_buffer.py. Unlike the DQN replay buffer (which stores individual transitions indefinitely), this buffer stores complete episodes of (state, action, log_prob, reward, value, done) tuples, computes returns and GAE advantages when finalized, and is cleared after each training update. Include a finalize() method that computes discounted returns and GAE advantages in-place using gamma=0.99 and gae_lambda=0.95. Include a get_batches(batch_size) method that yields random mini-batches for multi-epoch training."*

2. **PPO agent:** *"Create a PPOAgent class in agents/ppo_agent.py that extends the Agent base class. It should use the PolicyNetwork for the actor and the ValueNetwork for the critic. Include methods: (1) select_action(state, valid_actions) — sample from policy, return action and store log_prob and value estimate, (2) store_transition(state, action, log_prob, value, reward, done), (3) update(rollout_buffer) — run multiple epochs of mini-batch updates using the clipped surrogate objective for the actor and MSE loss for the critic, with an entropy bonus. Clip ratio epsilon=0.2, value loss coefficient=0.5, entropy coefficient=0.01, 4 update epochs, batch size 64."*

3. **Training loop:** *"Create training/train_ppo.py that trains the PPO agent against the heuristic agent. The structure: collect N complete episodes into the rollout buffer, finalize (compute returns and advantages), run PPO update, clear buffer, repeat. Collect 16 episodes per update batch. Log policy loss, value loss, entropy, clip fraction, and win rate. Evaluate every 500 episodes. Reuse the checkpoint infrastructure from training/checkpoint.py."*

### Hyperparameters (Starting Points)

| Parameter              | Value     |
|------------------------|-----------|
| Learning rate (actor)  | 3e-4      |
| Learning rate (critic) | 1e-3      |
| Discount factor (γ)    | 0.99      |
| GAE λ                  | 0.95      |
| Clip ratio (ε)         | 0.2       |
| Value loss coefficient | 0.5       |
| Entropy coefficient    | 0.01      |
| Update epochs          | 4         |
| Episodes per update    | 16        |
| Batch size             | 64        |

### After Claude Generates the Code

This is the most complex module. Take it piece by piece.

1. **GAE computation:** Find the `finalize()` method in the rollout buffer. Trace through the advantage computation step by step. For a 3-move episode with rewards [0, 0, 1], compute the advantage for each step by hand using γ=0.99 and λ=0.95. Does your hand calculation match what the code produces?

2. **The clipped objective:** Find the loss computation in `update()`. You should see:
   ```python
   ratio = (new_log_prob - old_log_prob).exp()
   surr1 = ratio * advantage
   surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
   policy_loss = -torch.min(surr1, surr2).mean()
   ```
   Ask yourself: when does the `min` choose `surr1` vs `surr2`? Draw it out for positive and negative advantages.

3. **Clip fraction:** This metric tracks what fraction of samples hit the clip boundary. It should be low early (policy changes are small) and may increase as the policy improves. If it's consistently above 0.3, the updates are too aggressive.

4. **Ask Claude to explain:** *"Walk me through one PPO update step. What happens to a transition where the old policy had log_prob=-1.2 for an action, the new policy has log_prob=-0.8, and the advantage is +0.5? Compute the ratio and the clipped loss."*

### Exercises

- [ ] Train PPO vs. the random agent for 10,000 episodes. Plot win rate, policy loss, value loss, and entropy over time. Entropy should decrease as the policy becomes more confident.
- [ ] **Monitor the clip fraction.** If it's near zero, the clip isn't doing anything (updates are already small enough). If it's above 0.3, consider reducing the learning rate. What do you see?
- [ ] **Experiment — remove the clip:** Set ε to a very large value (e.g., 10.0) so the clip never activates. This makes PPO equivalent to vanilla policy gradient with multiple epochs. Does training become unstable?
- [ ] **Experiment — change the number of epochs:** Try 1, 4, and 10 update epochs per batch. More epochs means more gradient steps on the same data. What's the sweet spot?
- [ ] **Experiment — entropy coefficient:** Try 0.0 (no entropy bonus), 0.01 (default), and 0.1 (strong exploration). How does this affect the speed of convergence and the final win rate?
- [ ] Train PPO vs. the heuristic agent. Compare the learning curve to REINFORCE with baseline (Module 10) and to DQN (Chapter 1).

### Suggested File Structure

```
c4/
├── agents/
│   ├── dqn_agent.py           # Chapter 1
│   ├── reinforce_agent.py     # Modules 9–10
│   └── ppo_agent.py           # New: PPO agent
├── models/
│   ├── q_network.py           # Chapter 1
│   ├── policy_network.py      # Module 8
│   └── value_network.py       # Module 10
├── training/
│   ├── replay_buffer.py       # Chapter 1 (DQN)
│   ├── rollout_buffer.py      # New: episode buffer with GAE
│   ├── train.py               # Chapter 1 (DQN training)
│   ├── train_reinforce.py     # Module 9
│   └── train_ppo.py           # New: PPO training loop
```

---

## Module 12 — Head-to-Head: DQN vs. Policy Gradient

**Goal:** Compare your DQN, REINFORCE, and PPO agents rigorously. Develop intuition for when each approach shines.

### Key Concepts

- **Sample efficiency**: How many episodes does each algorithm need to reach a given win rate? DQN reuses experience (replay buffer), so it often needs fewer total episodes. PPO uses each batch for multiple epochs but discards it after. REINFORCE uses each episode exactly once.
- **Wall-clock time**: Sample efficiency isn't the whole story. DQN trains on individual transitions (fast per step but many steps). PPO collects batches and runs multiple epochs (more computation per update but fewer updates). Which finishes first?
- **Final performance**: After sufficient training, do they reach the same win rate, or does one plateau higher?
- **Behavioral differences**: Even at the same win rate, the agents may play *differently*. A DQN agent plays deterministically (max Q). A policy gradient agent plays stochastically (samples from probabilities). Watch their games.

### What to Ask Claude to Build

**Suggested prompt:** *"Create a comparison script at notebooks/03_algorithm_comparison.py. It should: (1) load trained checkpoints for DQN, REINFORCE-with-baseline, and PPO agents, (2) evaluate each against random and heuristic opponents using the existing evaluate functions from training/evaluate.py, (3) run round-robin tournaments between all agents (each pair plays 500 games), (4) print a results table with win rates, and (5) plot a bar chart comparing win rates across matchups. Also create a function that plays one game between two agents and prints the board after every move, so I can watch them play."*

### Exercises

- [ ] Run the full comparison. Create a table like this and fill it in:

  | Agent | vs. Random | vs. Heuristic | vs. DQN | vs. REINFORCE | vs. PPO |
  |-------|-----------|--------------|---------|---------------|---------|
  | DQN   |           |              | —       |               |         |
  | REINFORCE |       |              |         | —             |         |
  | PPO   |           |              |         |               | —       |

- [ ] **Watch games.** Use the board-printing function to watch 3 games of DQN vs. PPO. Note any visible differences in play style. Does one play more aggressively? More defensively?
- [ ] **Inspect policies.** For the same board position, print the DQN Q-values and the PPO action probabilities side by side. Do they agree on which column is best? When they disagree, who's right?
- [ ] **Experiment — training budget:** Give each algorithm the same wall-clock training time (e.g., 5 minutes) instead of the same number of episodes. Which achieves the highest win rate under a fixed time budget?
- [ ] **Write a paragraph** comparing the three algorithms based on your experiments. Cover: ease of implementation, sample efficiency, final performance, and training stability. Which would you choose for a new board game project and why?

---

## Module 13 — Self-Play with PPO

**Goal:** Train the PPO agent through self-play and observe how it discovers increasingly sophisticated strategies.

### Key Concepts

- **Self-play curriculum**: Instead of training against a fixed opponent, the agent plays against a previous version of itself. As the agent improves, its opponent improves too — an automatic curriculum that scales with the agent's ability.
- **Opponent update schedule**: Freeze a copy of the current policy as the opponent. Update the opponent every N episodes to a newer checkpoint. Too frequent updates make the opponent a moving target (unstable). Too infrequent updates let the agent overfit to a stale opponent.
- **Policy collapse in self-play**: A risk unique to self-play — the agent discovers a narrow strategy that beats the current opponent, and the opponent (being a copy) adopts the same strategy. Both players do the same thing, and neither explores alternatives. The entropy bonus is critical to preventing this.
- **ELO tracking**: To measure improvement over time, track a running ELO rating. Each time the current agent beats a frozen previous version, its ELO goes up. This gives a single number that measures absolute improvement even though the opponents keep changing.

### What to Ask Claude to Build

**Suggested prompt:** *"Create training/train_ppo_selfplay.py that trains the PPO agent through self-play. Structure: (1) initialize the PPO agent and a frozen copy as the opponent, (2) play episodes where the agent and opponent alternate as Player 1 and Player 2 (collect transitions for both perspectives), (3) update the PPO agent, (4) every 1,000 episodes, freeze the current policy as the new opponent, (5) every 500 episodes, evaluate against the random agent, heuristic agent, and the previous frozen opponent. Track and log ELO ratings. Increase the entropy coefficient to 0.02 to prevent policy collapse during self-play. Save checkpoints at each opponent update."*

### After Claude Generates the Code

1. **Watch the ELO curve.** It should generally trend upward, though there will be dips when the opponent is refreshed (the agent temporarily struggles against its improved self).
2. **Watch for policy collapse.** If the entropy drops to near zero and the win rate against heuristic stops improving, the agent may have collapsed. Check if it always plays the same opening moves.
3. **Compare to DQN self-play.** If you ran self-play training in Chapter 1, compare the learning curves. Policy gradient self-play should be smoother (stochastic policy explores more naturally) but possibly slower.

### Exercises

- [ ] Run self-play training for 20,000 episodes. Plot ELO over time. Is the trend upward?
- [ ] **Monitor the entropy.** Plot entropy over training. If it drops below 0.1, the policy is becoming too deterministic for self-play. Try increasing the entropy coefficient.
- [ ] After self-play training, evaluate against the heuristic agent. Does self-play produce a stronger agent than training against the heuristic directly?
- [ ] **Experiment — opponent update frequency:** Try updating the frozen opponent every 500, 1,000, and 5,000 episodes. Which produces the strongest final agent?
- [ ] Load checkpoints from different points in training and play them against each other. Can the later checkpoint consistently beat the earlier one?
- [ ] Play against your self-play-trained agent yourself. Can you spot strategies it has learned that the heuristic agent doesn't use?

---

## Module 14 — Improvements and Next Steps

**Goal:** Refine your policy gradient agents and explore what comes next.

### PPO Improvements

Each of these is a targeted change. Implement, measure, and compare — don't just bolt things on.

- **Shared actor-critic backbone:** Instead of separate networks for the policy and value function, use a single convolutional backbone with two heads (one for logits, one for value). This lets the value function benefit from features the policy has learned, and vice versa. *"Create a models/actor_critic_network.py that has a shared conv backbone with two output heads — a policy head (7 logits) and a value head (scalar). Update PPOAgent to use this instead of separate PolicyNetwork and ValueNetwork."*

- **Reward shaping:** Connect Four's sparse reward (only at game end) makes credit assignment hard for all algorithms. Add small intermediate rewards — e.g., +0.01 for creating a three-in-a-row, -0.01 for letting the opponent create one. These give earlier gradient signal. *"Add optional reward shaping to the environment: small positive reward for creating 3-in-a-row threats, small negative for allowing opponent threats. Make it toggleable so we can compare with and without."*

- **Learning rate scheduling:** Decrease the learning rate over training. A cosine schedule or linear decay can help the agent make large updates early (when the policy is bad) and small refinements later (when the policy is good). *"Add a cosine learning rate schedule to the PPO training loop. Start at 3e-4 and decay to 1e-5 over the full training run."*

### After Each Improvement

1. **Measure the impact** — don't guess. Compare training curves and final performance.
2. **Check for regressions** — an improvement in one setting may hurt in another (e.g., reward shaping helps vs. random but not vs. self-play).
3. **Understand *why*** — ask Claude to explain the mechanism. *"Why does sharing the backbone between actor and critic help? When might it hurt?"*

### Beyond Policy Gradients (Future Exploration)

- **AlphaZero-Style Agent**: Combine a neural network (policy + value head) with Monte Carlo Tree Search (MCTS). The network's policy output guides MCTS search, and MCTS results improve the network. This is the state-of-the-art for board games.
- **Multi-Agent RL**: Train two separate agents that co-evolve through competition. Each agent has its own network and optimizer. This avoids the symmetry assumptions of self-play.
- **Curriculum Learning**: Automatically adjust opponent difficulty based on the agent's current skill level. Start easy, ramp up smoothly.

### Exercises

- [ ] Implement the shared backbone. Does it train faster? Compare parameter count to the separate-network version.
- [ ] Implement reward shaping. Compare win rate curves with and without. Does it help more for REINFORCE or PPO?
- [ ] **Research exercise:** Read the [PPO paper](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017). List two implementation details from the paper that your implementation doesn't include. Are they relevant for Connect Four?
- [ ] **Capstone:** Train your best agent (whichever algorithm and configuration performed best) with your best settings. Run a final evaluation against all opponents. Write a one-page summary of what you built, what worked, what didn't, and what you'd try next.

---

## Appendix — Recommended Resources

| Resource | What It Covers |
|----------|---------------|
| [Spinning Up in Deep RL — Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) | Clear derivation of the policy gradient theorem |
| [Policy Gradient Methods for RL with Function Approximation](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) (Sutton et al., 1999) | The original policy gradient theorem paper |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) | The PPO paper — accessible and well-written |
| [High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438) (Schulman et al., 2015) | The GAE paper — explains bias-variance trade-off in advantage estimation |
| [David Silver's RL Lectures — Policy Gradient](https://www.davidsilver.uk/teaching/) (Lecture 7) | University-level policy gradient theory |
| [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) | Practical PPO implementation guide — what the paper doesn't tell you |

---

## Full Project Structure (End of Chapter 2)

```
c4/
├── pyproject.toml
├── game.md
├── lesson_plan.md              # Chapter 1
├── lesson_plan_chapter_2.md    # This file
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
│   ├── dqn_agent.py            # Chapter 1
│   ├── reinforce_agent.py      # Modules 9–10
│   └── ppo_agent.py            # Module 11
├── models/
│   ├── __init__.py
│   ├── q_network.py            # Chapter 1
│   ├── policy_network.py       # Module 8
│   ├── value_network.py        # Module 10
│   └── actor_critic_network.py # Module 14
├── training/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── replay_buffer.py        # Chapter 1
│   ├── rollout_buffer.py       # Module 11
│   ├── evaluate.py
│   ├── train.py                # Chapter 1
│   ├── train_reinforce.py      # Module 9
│   ├── train_ppo.py            # Module 11
│   └── train_ppo_selfplay.py   # Module 13
├── notebooks/
│   ├── 01_pytorch_basics.py
│   ├── 02_reinforce_baseline_comparison.py  # Module 10
│   └── 03_algorithm_comparison.py           # Module 12
├── checkpoints/
└── tests/
```
