# Connect Four RL Agent — Lesson Plan

A progressive curriculum for building a reinforcement learning agent that learns to play Connect Four. You'll use **Claude Code** to generate most of the implementation, which means your job is to *direct* the AI, *read and understand* the code it produces, *run experiments* to build intuition, and *debug* when things go wrong. The RL and PyTorch concepts matter more than typing the code yourself.

## How to Learn with AI Coding Tools

Using an AI to write the code doesn't mean you skip the learning — it changes *where* the learning happens:

- **Before generating**: Know what you want to build and why. The Key Concepts sections prepare you to give good instructions and evaluate the output.
- **After generating**: Read every line. If you can't explain what a line does, stop and ask Claude to explain it. The code is not done until you understand it.
- **Experimentation is the real teacher**: Change a hyperparameter, remove a component, break something on purpose. Watching what happens when you change things builds deeper intuition than writing it from scratch ever could.
- **Prompting well is a skill**: Each module includes suggested prompts. As you get comfortable, you'll learn what level of detail produces the best results.

---

## Module 1 — Game Environment

**Goal:** Build a fully functional Connect Four game in pure Python that can serve as an RL environment.

### Key Concepts

Read and understand these *before* you ask Claude to generate anything. They define the vocabulary you'll use for the rest of the project.

- **State**: The current board configuration (a 6x7 grid where each cell is empty, player 1, or player 2).
- **Action**: Choosing a column (0–6) to drop a disc into.
- **Reward**: A signal the agent receives after each action (+1 for winning, 0 otherwise).
- **Episode**: One complete game from start to finish.
- **Environment**: The system the agent interacts with — it accepts actions and returns new states and rewards.

### What to Ask Claude to Build

Ask Claude to create a `ConnectFourEnv` class with:

- A 2D numpy array board of shape `(6, 7)` with values `0` (empty), `1` (player 1), `2` (player 2).
- `reset()` — Clear the board, return the initial state.
- `get_valid_actions()` — Return columns that aren't full.
- `step(action)` — Drop a disc, check for win/draw, return `(obs, reward, done, info)`.
- `check_winner()` — Scan horizontal, vertical, and diagonal lines of four.
- `render()` — Print the board in a readable format.
- Automatic alternation between player 1 and player 2.

**Suggested prompt:** *"Create a Connect Four environment class in env/connect_four.py. Reference game.md for the rules. The class should follow the OpenAI Gym style interface with reset(), step(), and render() methods. Include get_valid_actions() and check_winner(). Use numpy for the board."*

### After Claude Generates the Code

1. **Read the `check_winner()` method carefully.** This is the most logic-heavy part. Trace through one direction (e.g., horizontal) by hand. How does it scan for four in a row? Could it miss an edge case?
2. **Read the `step()` method.** Follow the control flow: what happens when a player drops a disc into a full column? What reward does each player get on a win?
3. **Ask Claude questions** about anything you don't understand: *"Explain how the diagonal check works in check_winner()"*.

### Exercises

- [ ] Open a Python REPL, create an environment, and play a full game by calling `step()` manually. Get comfortable with the state/action/reward loop.
- [ ] Ask Claude to generate unit tests, then **read the tests before running them** — do they cover all four win directions? Edge cases like full columns and draws?
- [ ] **Break something on purpose:** comment out the diagonal win check and run the tests. Confirm the right test fails.
- [ ] Ask Claude to add a `clone()` method. Before looking at the implementation, predict: why would a deep copy be needed instead of a shallow copy?

### Suggested File Structure

```
c4/
├── game.md
├── lesson_plan.md
└── env/
    ├── __init__.py
    └── connect_four.py      # ConnectFourEnv class
```

---

## Module 2 — PyTorch Fundamentals

**Goal:** Understand the PyTorch building blocks so you can read and modify the neural network code in later modules.

You don't need to be fluent in PyTorch from scratch — you need to be fluent enough to *read* what Claude generates, *understand* what each piece does, and *modify* it when experiments call for changes.

### Key Concepts

- **Tensor**: Like a numpy array, but it can live on a GPU and track gradients. You'll convert board states to tensors.
- **Autograd**: PyTorch records operations on tensors so it can compute gradients automatically. This is how neural networks learn — you don't write the calculus yourself.
- **nn.Module**: The base class for neural networks. You define layers in `__init__` and the forward pass in `forward()`. Everything in PyTorch is built from these.
- **Optimizer**: Adjusts model weights based on gradients (e.g., Adam). Think of it as the "learning" step.
- **Loss Function**: Measures how wrong the model is (e.g., MSELoss). The optimizer uses this to decide which direction to push the weights.

### What to Ask Claude to Build

Ask Claude to create a small exploration script or notebook:

**Suggested prompt:** *"Create a script at notebooks/01_pytorch_basics.py that demonstrates: (1) creating tensors from numpy arrays, (2) converting a Connect Four board state into a float tensor, (3) building a small 2-layer neural network with nn.Module that takes a flattened board (42 inputs) and outputs 7 values, (4) running a forward pass with a random board, (5) a minimal training loop that fits the network to 10 random input-target pairs and prints the loss each step. Add comments explaining each section."*

### After Claude Generates the Code

1. **Focus on the nn.Module class.** Identify `__init__` (where layers are defined) and `forward` (where data flows through them). This pattern repeats in every neural network you'll see.
2. **Focus on the training loop.** Find these five steps — they appear in every PyTorch training loop:
   - `optimizer.zero_grad()` (clear old gradients)
   - Forward pass (feed data through the network)
   - Compute loss (compare output to target)
   - `loss.backward()` (compute new gradients)
   - `optimizer.step()` (update weights)
3. **Ask Claude to explain** anything that isn't clear: *"What does optimizer.zero_grad() do and why is it needed?"*

### Exercises

- [ ] Run the script. Verify the loss decreases over iterations.
- [ ] **Experiment:** Change the network from 2 layers to 1 layer. Does it still converge? What about 4 layers?
- [ ] **Experiment:** Change the learning rate from 1e-3 to 1.0. What happens? What about 1e-6? This builds intuition for a critical hyperparameter.
- [ ] **Predict before running:** If you change `nn.ReLU` to `nn.Sigmoid`, will the network still learn? Run it and check.
- [ ] Make sure you can answer: *What is the shape of the tensor at each point in the forward pass?* Add print statements if needed.

---

## Module 3 — Agents and Baselines

**Goal:** Create an agent framework and build simple agents to serve as baselines and training opponents.

### Key Concepts

- **Policy**: A strategy that maps states to actions. A random policy picks uniformly; a learned policy picks intelligently.
- **Baseline**: A simple agent you measure your RL agent against. If your DQN can't beat a random agent, something is wrong.
- **Win rate**: The primary evaluation metric — percentage of games won over many episodes.

### What to Ask Claude to Build

Generate the agent framework in stages so you can understand each piece:

1. **Agent base class and random agent:** *"Create an Agent base class in agents/base.py with a select_action(state, valid_actions) method. Then create a RandomAgent in agents/random_agent.py that picks uniformly from valid actions."*
2. **Heuristic agent:** *"Create a HeuristicAgent in agents/heuristic_agent.py that: (1) takes a winning move if one exists, (2) blocks the opponent's winning move if one exists, (3) otherwise prefers the center column. It needs access to the environment's check_winner logic."*
3. **Game runner and evaluation:** *"Create a play_game(agent1, agent2, env) function that plays a full episode and returns the winner. Create an evaluate(agent1, agent2, n_games) function that plays many games and returns win/loss/draw statistics."*

### After Claude Generates the Code

1. **Read the heuristic agent carefully.** How does it check for winning/blocking moves? Does it actually try each valid column and test the result, or does it use some other approach?
2. **Think about what's missing** from the heuristic: it doesn't think ahead more than one move. A human would consider traps (two ways to win simultaneously). Keep this in mind — it's a gap the DQN should eventually learn to exploit.

### Exercises

- [ ] Run 1,000 games of random vs. random. Verify player 1 wins slightly more often (first-move advantage). Ask Claude to explain *why* going first is an advantage.
- [ ] Run 1,000 games of heuristic vs. random. The heuristic should dominate. What's the actual win rate?
- [ ] Ask Claude to add a `HumanAgent` that reads input from the terminal. Play a few games against the heuristic agent yourself. Can you beat it? Where does it make mistakes?
- [ ] **Predict before running:** What happens when heuristic plays against heuristic? Does player 1 always win? Run it and check.

### Suggested File Structure

```
c4/
├── game_runner.py             # play_game() and evaluate()
├── play.py                    # Terminal script to play vs an agent
└── agents/
    ├── __init__.py
    ├── base.py                # Agent base class
    ├── random_agent.py
    ├── heuristic_agent.py
    └── human_agent.py
```

---

## Module 4 — Deep Q-Network (DQN)

**Goal:** Understand Q-learning theory and build a neural network that learns Q-values for Connect Four through self-play. This is the core of the project.

### Key Concepts — Q-Learning Theory

- **Q-Value** `Q(s, a)`: The expected future reward from taking action `a` in state `s` and then playing optimally. High Q-value = good move.
- **Bellman Equation**: `Q(s, a) = r + γ * max_a' Q(s', a')` — the Q-value of an action equals the immediate reward plus the discounted best future Q-value.
- **Discount Factor** `γ` (gamma): How much the agent values future rewards vs. immediate ones. Typically 0.95–0.99.
- **Epsilon-Greedy Exploration**: With probability `ε`, pick a random action (explore); otherwise pick the action with the highest Q-value (exploit). Decay `ε` over time.
- **Temporal Difference (TD) Learning**: Update Q-values based on the difference between the predicted Q-value and the Bellman target.

### Why a Neural Network?

Connect Four has roughly **4.5 trillion** possible board states. A Q-table that stores a value for every (state, action) pair would be impossibly large. Instead, we use **function approximation** — a neural network that takes a board state as input and outputs approximate Q-values for each action. The network *generalizes* across similar board positions, estimating Q-values for states it has never seen before.

### Key Concepts — DQN

- **Experience Replay**: Store transitions `(s, a, r, s', done)` in a buffer and sample random mini-batches for training. This breaks correlation between consecutive experiences and stabilizes learning.
- **Target Network**: A second copy of the Q-network whose weights are updated slowly (or copied periodically). Without this, training is unstable because the target keeps moving.
- **Loss Function**: MSE between predicted Q-values and Bellman targets: `L = (Q(s,a;θ) - [r + γ * max_a' Q(s',a';θ⁻)])²` where `θ⁻` are the target network weights.

### Network Architecture

```
Input: Board state as tensor, shape (batch, 3, 6, 7)
       3 channels: player 1 pieces, player 2 pieces, current player indicator

→ Conv2d(3, 64, kernel_size=4, padding=1)  → ReLU
→ Conv2d(64, 64, kernel_size=3, padding=1) → ReLU
→ Flatten
→ Linear(*, 128) → ReLU
→ Linear(128, 7)                            ← Q-value for each column

Mask invalid actions by setting their Q-values to -infinity before selecting.
```

### What to Ask Claude to Build

Build this in stages. Generate one piece at a time, read and understand it, then move on.

1. **State encoding:** *"Create a function encode_state(board, current_player) that converts a numpy board (6x7) into a PyTorch tensor of shape (3, 6, 7). Channel 0: binary mask of current player's pieces. Channel 1: binary mask of opponent's pieces. Channel 2: all ones if player 1's turn, all zeros if player 2's."*

2. **Q-Network:** *"Create a QNetwork class in models/q_network.py using the architecture described in lesson_plan.md Module 4. Include a method that masks invalid actions by setting their Q-values to negative infinity."*

3. **Replay buffer:** *"Create a ReplayBuffer class in training/replay_buffer.py. It should store (state, action, reward, next_state, done) transitions in a fixed-size buffer and support sampling random mini-batches as PyTorch tensors."*

4. **DQN Agent:** *"Create a DQNAgent class in agents/dqn_agent.py that combines the QNetwork, a target network, the replay buffer, and epsilon-greedy action selection. Include methods: select_action, store_transition, train_step, and update_target_network. Reference the hyperparameters in lesson_plan.md Module 4."*

5. **Training loop:** *"Create a training script at training/train.py that trains the DQN agent by playing against a random agent. Log training loss and evaluate win rate every 500 episodes. Include a progress display."*

### Hyperparameters (Starting Points)

| Parameter              | Value     |
|------------------------|-----------|
| Learning rate          | 1e-4      |
| Discount factor (γ)    | 0.99      |
| Replay buffer size     | 100,000   |
| Batch size             | 64        |
| Epsilon start          | 1.0       |
| Epsilon end            | 0.05      |
| Epsilon decay steps    | 100,000   |
| Target network update  | Every 1,000 steps |

### After Claude Generates the Code

This is the most code-heavy module. Take time to understand each piece:

1. **Think about Q-values in Connect Four.** Imagine a board where you can win by dropping in column 3. The Q-value `Q(board, col=3)` should be high (+1), while other columns should be lower. The heuristic agent from Module 3 does this by brute force — the DQN will learn to approximate it.
2. **State encoding:** Feed a sample board through `encode_state()` and print each channel. Does channel 0 correctly show the current player's pieces? Does channel 2 correctly indicate whose turn it is?
3. **Q-Network:** Run a forward pass on a random board. You should get 7 values (one per column). They'll be random at first — that's expected.
4. **Replay buffer:** Add 10 fake transitions, sample a batch of 3, and inspect the shapes. Each component should be a tensor with the batch dimension first.
5. **train_step():** This is the heart of DQN. Find these steps in the code:
   - Sample a batch from the replay buffer
   - Compute Q-values for the current states: `Q(s, a; θ)`
   - Compute target Q-values using the target network: `r + γ * max Q(s', a'; θ⁻)`
   - Compute the loss between predicted and target
   - Backpropagate and update weights
6. **Ask Claude to explain** anything you can't trace: *"Walk me through what happens in train_step() line by line. Why do we use torch.no_grad() when computing targets?"*

### Exercises

- [ ] **On paper:** Given a 3-state MDP with known rewards and transitions, compute Q-values by hand using the Bellman equation. This should take 10–15 minutes and is worth every second.
- [ ] **Thought experiment:** Imagine storing a Q-value for every (board state, column) pair in Connect Four. With ~4.5 trillion states and 7 columns, how much memory would that take?
- [ ] Before training, run a forward pass and verify the output shape is `(1, 7)` for a single board.
- [ ] Train for 1,000 episodes. Is the loss trending down? If not, ask Claude to help debug.
- [ ] **Experiment — remove experience replay:** Modify the agent to train on each transition immediately (batch size 1, no buffer). How does the loss curve change? This demonstrates *why* replay matters.
- [ ] **Experiment — remove the target network:** Use the same network for both predictions and targets. What happens to training stability?
- [ ] Save a checkpoint. Load it in a fresh script and play a game against the loaded agent. Does it work?
- [ ] Inspect Q-values for a nearly-won board state. Does the network assign a high value to the winning column?

### Suggested File Structure

```
c4/
├── agents/
│   └── dqn_agent.py          # DQNAgent class
├── models/
│   ├── __init__.py
│   └── q_network.py           # QNetwork (nn.Module)
├── training/
│   ├── __init__.py
│   ├── replay_buffer.py       # ReplayBuffer class
│   └── train.py               # Training loop
└── checkpoints/               # Saved model weights
```

---

## Module 5 — Training and Evaluation

**Goal:** Train a strong agent through a curriculum of increasingly difficult opponents and rigorously evaluate its performance.

### Training Curriculum

1. **Phase 1 — vs. Random Agent** (warm-up): The DQN agent learns basic tactics: take winning moves, avoid obvious losses. Train until win rate exceeds 85%.
2. **Phase 2 — vs. Heuristic Agent**: The opponent now blocks wins and plays center. The agent must learn deeper strategy. Train until win rate exceeds 70%.
3. **Phase 3 — Self-Play**: The agent plays against a frozen copy of itself. Periodically update the opponent to the latest checkpoint. This drives the agent to discover increasingly sophisticated strategies.

### Evaluation Metrics

- **Win rate** vs. random agent (should be 90%+).
- **Win rate** vs. heuristic agent (should be 75%+).
- **Average game length** — strong agents tend to win faster.
- **Loss curve** — should decrease over training and stabilize.
- **Q-value distribution** — spot-check that Q-values for obviously good/bad moves are sensible.

### What to Ask Claude to Build

**Suggested prompt:** *"Update training/train.py to implement a 3-phase training curriculum: Phase 1 trains vs random until 85% win rate, Phase 2 trains vs heuristic until 70% win rate, Phase 3 does self-play. Add matplotlib plotting for loss curve and win rate over time. Save checkpoints between phases. Create training/evaluate.py with functions to run evaluation games and generate performance reports."*

### After Claude Generates the Code

1. **Watch the win rate curves.** The most revealing moment is the transition between phases — win rate should drop when the opponent gets harder, then climb back up.
2. **Watch actual games.** Ask Claude to add a mode that prints the board after each move. Watch 5 games at each phase. Do the agent's moves look intelligent? Can you spot strategies it has learned?

### Exercises

- [ ] Run the full training curriculum. This will take a while — watch the metrics as it trains.
- [ ] Plot win rate vs. random, win rate vs. heuristic, and training loss on the same figure.
- [ ] **Watch and evaluate:** Observe 5 games of your trained agent vs. the heuristic agent. Note specific moves that seem smart or dumb.
- [ ] **Experiment:** What happens if you skip Phase 1 and go straight to self-play? Is it better or worse?
- [ ] **Experiment:** Double the replay buffer size. Does it help? What about halving it?
- [ ] Play against your trained agent yourself using the HumanAgent. Can you beat it?

---

## Module 6 — Improvements and Next Steps

**Goal:** Refine the agent and explore what comes after DQN.

### DQN Improvements

Each of these is a targeted change you can ask Claude to implement, then measure the impact:

- **Double DQN**: Use the online network to *select* the best action but the target network to *evaluate* it. This reduces Q-value overestimation. *"Update the DQN agent to use Double DQN. In train_step, use the online network to pick the best action for the next state, but use the target network to evaluate that action's Q-value."*

- **Dueling DQN**: Split the network into two streams — one estimates the state value `V(s)`, the other estimates the advantage `A(s, a)`. They combine as `Q(s, a) = V(s) + A(s, a) - mean(A)`. *"Modify QNetwork to use a dueling architecture. After the conv layers, split into a value stream and an advantage stream, then combine them."*

- **Prioritized Experience Replay**: Sample transitions with high TD error more frequently. *"Update ReplayBuffer to use prioritized sampling based on TD error. Use proportional prioritization with importance sampling weights."*

### After Each Improvement

Don't just implement and move on. For each change:
1. **Understand *why* it helps** — ask Claude to explain the problem it solves.
2. **Measure the impact** — compare training curves and final win rates against the vanilla DQN.
3. **Check for regressions** — does the improvement help in one phase but hurt in another?

### Beyond DQN (Future Exploration)

- **Policy Gradient Methods (REINFORCE, PPO)**: Instead of learning Q-values, directly learn a policy that outputs action probabilities. A fundamentally different approach to RL.
- **AlphaZero-Style Agent**: Combine a neural network (policy + value head) with Monte Carlo Tree Search (MCTS). This is the state-of-the-art for board games but significantly more complex.

### Exercises

- [ ] Implement Double DQN. Compare win rates — does it improve?
- [ ] Implement Dueling DQN. Ask Claude to add logging that prints the value and advantage streams separately for a sample board. Do the values make intuitive sense?
- [ ] **Research exercise:** Read the [AlphaZero paper](https://arxiv.org/abs/1712.01815) (or a summary). List three key differences between AlphaZero's approach and what you built.

---

## Appendix — Recommended Resources

| Resource | What It Covers |
|----------|---------------|
| [PyTorch Tutorials](https://pytorch.org/tutorials/) | Official tutorials for tensors, autograd, and nn.Module |
| [Spinning Up in Deep RL](https://spinningup.openai.com/) (OpenAI) | Excellent RL fundamentals with clear explanations |
| [David Silver's RL Lectures](https://www.davidsilver.uk/teaching/) | University-level RL theory (free on YouTube) |
| [Human-level control through deep RL](https://www.nature.com/articles/nature14236) (Mnih et al., 2015) | The original DQN paper |
| [Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013) | The earlier DQN preprint — more accessible |

---

## Full Project Structure (Final)

```
c4/
├── pyproject.toml             # Project config (numpy, torch, matplotlib)
├── game.md                    # Game rules reference
├── lesson_plan.md             # This file
├── game_runner.py             # play_game() and evaluate()
├── play.py                    # Terminal script to play vs an agent
├── env/
│   ├── __init__.py
│   └── connect_four.py        # ConnectFourEnv
├── agents/
│   ├── __init__.py
│   ├── base.py                # Agent base class
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   ├── human_agent.py
│   └── dqn_agent.py           # DQN agent
├── models/
│   ├── __init__.py
│   └── q_network.py           # QNetwork (nn.Module)
├── training/
│   ├── __init__.py
│   ├── replay_buffer.py
│   ├── train.py               # Main training entry point
│   └── evaluate.py            # Evaluation and plotting
├── notebooks/
│   └── 01_pytorch_basics.py
├── checkpoints/
│   └── ...
└── tests/
    ├── test_connect_four.py   # Environment tests
    ├── test_game_runner.py    # Game runner tests
    └── test_replay_buffer.py
```
