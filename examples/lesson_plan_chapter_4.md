# Connect Four RL Agent — Chapter 4: From Local to Cloud

A progressive curriculum for deploying your RL training pipeline to **Databricks**, tracking experiments with **MLflow**, and bringing trained models back to your local machine for play. Chapters 1–3 built and trained agents entirely on your laptop. That works — until it doesn't. Long training runs tie up your machine for hours, you lose track of which hyperparameters produced which results, and you can't leverage GPU acceleration. This chapter fixes all three problems.

You already have working DQN, REINFORCE, and PPO agents with training loops, checkpoints, and evaluation. Nothing in this chapter changes how those algorithms work. Instead, you'll wrap them in infrastructure that lets you train faster, track better, and work from anywhere.

This chapter is as much about **MLOps intuition** as it is about code. Every ML project eventually outgrows a single laptop. Understanding the deployment boundary — what runs locally vs. what runs in the cloud, how artifacts move between them, how you track what happened — is a skill that transfers to any ML project, not just RL.

---

## Module 23 — Why Cloud Training?

**Goal:** Understand the practical limitations of local training and what cloud infrastructure solves.

### Key Concepts

- **Compute bottleneck**: Your RL training loops are CPU-bound. A single REINFORCE episode involves ~15 forward passes through a ~389K-parameter CNN, followed by a full backward pass. At 20,000 episodes, that's 300,000+ forward passes. On CPU, each forward pass takes ~2ms; on a GPU, it takes ~0.05ms. The math is stark: a training run that takes 30 minutes on CPU finishes in under 2 minutes on GPU. This matters more as you scale to longer runs and larger networks.

- **The GPU advantage for neural networks**: GPUs were designed for parallel matrix multiplication — exactly what neural network forward and backward passes do. A CPU processes matrix operations sequentially across 8–16 cores. A GPU processes them in parallel across thousands of CUDA cores. For small matrices (like a 6x7 board), the speedup is modest. For batched operations (PPO updates with batch_size=64, DQN training on replay buffer batches), the speedup is dramatic.

- **Experiment tracking**: After 20 training runs with different hyperparameters, can you answer: which learning rate produced the highest win rate against the heuristic? What entropy coefficient prevented collapse? Right now, this information lives in scattered terminal output and JSON files in timestamped checkpoint directories. **MLflow** is a lightweight experiment tracker that logs hyperparameters, metrics over time, and artifacts (models, plots) to a single searchable database. It's the difference between a lab notebook and a pile of Post-it notes.

- **The training/inference split**: Training is compute-hungry and runs for minutes to hours. Inference (playing a game) is cheap — one forward pass per move. This creates a natural split: train on powerful cloud hardware, download the trained model, play locally. Your model checkpoint (a ~5MB `.pt` file) is the artifact that crosses the boundary. Everything else — the training loop, the opponents, the replay buffers — stays in the cloud.

- **Databricks as ML infrastructure**: Databricks provides managed Spark clusters, but for this project we're using it as a **GPU job runner** with built-in MLflow. You define a job (what code to run, what cluster to use), deploy it, and trigger it. Databricks spins up a GPU VM, runs your training, logs results to MLflow, and tears down the VM. You pay for compute only while training runs.

### No Code This Module

Make sure you can answer these before moving on:

1. Your REINFORCE training loop processes one episode at a time — no batching. Will a GPU help much for REINFORCE specifically? What about PPO, which updates on batches of 64 transitions across 4 epochs? What about DQN, which trains on batches of 64 from the replay buffer every 4 steps?
<!-- REINFORCE: modest GPU benefit — each forward/backward pass is a single sample, so GPU parallelism is underutilized. The speedup comes from faster individual matrix ops, not parallelism. PPO: significant benefit — batch updates parallelize well across GPU cores. DQN: significant benefit — replay buffer sampling + batch training is the ideal GPU workload. The lesson: GPU ROI depends on how much batched computation your algorithm does. -->

2. You currently save checkpoints to `checkpoints/run_YYYYMMDD_HHMMSS_*/model.pt` on your local filesystem. If training runs on a cloud VM, where do those files go when the VM is terminated? How do you get them back?
<!-- They're destroyed when the VM terminates. Cloud compute is ephemeral — the filesystem is temporary. You must explicitly copy artifacts to durable storage (MLflow artifact store, cloud storage bucket, or database) before the VM shuts down. This is the fundamental lesson of cloud ML: your code must save results to persistent storage, not just the local disk. -->

3. Why not just buy a GPU for your local machine instead of using cloud compute?
<!-- Cost and utilization. A good ML GPU costs $1,000–$3,000 and sits idle most of the time. Cloud GPU instances cost $0.50–$3.00/hour and you pay only when training. For this project (30-minute training runs a few times a week), cloud compute costs pennies. For 24/7 production training, a local GPU makes more sense. The break-even point depends on utilization rate. -->

### Exercises

- [ ] **Estimate the cost.** An Azure Standard_NC6s_v3 GPU instance costs roughly $0.90/hour. If your REINFORCE training run takes 15 minutes on GPU, how much does one run cost? How about 50 experimental runs while tuning hyperparameters?
- [ ] **Catalog your experiments.** Without looking at file timestamps, can you answer: what hyperparameters did your best REINFORCE run use? How about your best PPO run? If this is hard, you've experienced the problem MLflow solves.
- [ ] **Thought experiment.** You want to run a hyperparameter sweep: 3 learning rates x 3 entropy coefficients x 2 opponents = 18 runs. How long would this take sequentially on your laptop? On Databricks, you could run all 18 as parallel jobs. What's the wall-clock time difference?

---

## Module 24 — GPU Support in PyTorch

**Goal:** Make your existing code device-aware so the same training scripts run transparently on CPU (local) or GPU (Databricks).

### Key Concepts

- **Device abstraction**: PyTorch uses `torch.device` to represent where tensors and models live — either `cpu` or `cuda` (NVIDIA GPU). Every tensor has a device. Every model's parameters are tensors, so models have a device too. Operations between tensors on different devices fail with an error. The entire pipeline — encoding, model, loss computation — must be on the same device.

- **The `.to(device)` pattern**: Moving a model to GPU: `model.to(torch.device("cuda"))`. Moving a tensor to GPU: `tensor.to(device)`. This is a one-line change per object, but it must be applied consistently everywhere tensors are created or models are initialized. Miss one spot and you get a runtime error: `Expected all tensors to be on the same device`.

- **Auto-detection**: Rather than hardcoding a device, detect at startup:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
  This single line makes your code portable — it uses GPU when available, falls back to CPU when not. Your local machine (no GPU) runs on CPU. Databricks (GPU cluster) runs on CUDA. Same code, different hardware.

- **`map_location` for cross-device loading**: When you `torch.save()` a model on GPU, the checkpoint remembers it came from CUDA. If you `torch.load()` that checkpoint on a CPU-only machine, it fails unless you specify `map_location="cpu"`. This is the most common bug when moving models between local and cloud — the fix is a single argument.

- **Where tensors are created**: The tricky part isn't moving models — it's finding every place a tensor is created during training. In your codebase, tensors appear in:
  - `encode_state()` — creates the 3-channel board tensor
  - `agent.update()` — creates returns, advantages, loss tensors
  - Replay/rollout buffers — sample batches as tensors
  
  Each of these must produce tensors on the correct device.

### What to Ask Claude to Build

Build in stages so you can test each piece:

1. **Device utility:** *"Create a utility function `get_device()` in training/device.py that returns `torch.device('cuda')` if CUDA is available, otherwise `torch.device('cpu')`. Print the detected device when called."*

2. **Update encode_state:** *"Update env/encoding.py's `encode_state()` function to accept an optional `device` parameter (default None). When provided, move the output tensor to that device. When None, behavior is unchanged (CPU tensor). This preserves backward compatibility — existing code that doesn't pass device still works."*

3. **Update agents:** *"Update ReinforceAgent, DQNAgent, and PPOAgent to accept a `device` parameter in `__init__`. Default to `None`, which means auto-detect using `get_device()`. Move all networks to the device in the constructor. Pass the device to `encode_state()` in `select_action()`. Ensure tensors created in `update()` (returns, advantages) are on the correct device. Update `torch.load()` calls in `.load()` to use `map_location=self.device` so models trained on GPU can load on CPU and vice versa."*

4. **Update training loops:** *"Update `training/train.py`, `training/train_reinforce.py`, and `training/train_ppo.py` to detect the device at the start of `train()`, pass it to the agent constructor, and print it in the training header."*

### After Claude Generates the Code

1. **Trace the device flow.** Starting from `train()`, follow the device through: `train()` → `agent.__init__()` → `self.policy_net.to(device)` → `select_action()` → `encode_state(board, player, device)` → tensor on device → model forward pass → output on device. Every step must be on the same device. If you see a `torch.tensor(...)` call without a `device=` argument, that tensor is on CPU — it will crash on GPU if it interacts with a CUDA tensor.

2. **Check the update method.** The `ReinforceAgent.update()` method creates `returns_t = torch.tensor(returns)`. This defaults to CPU. After the change, it should be `torch.tensor(returns, device=self.device)`. Find every `torch.tensor()` call in the agents and verify they specify the device.

3. **Check the replay buffer.** `ReplayBuffer.sample()` converts numpy arrays to tensors. These tensors must be on the device. Find the `torch.tensor()` or `torch.as_tensor()` calls in `sample()` and verify they move to the right device.

4. **Test locally.** Run `uv run python -m training.train_reinforce 100`. It should print `device=cpu` and work exactly as before. Nothing should break — the default behavior is unchanged.

5. **Ask Claude:** *"If I accidentally leave one `torch.tensor()` call on CPU while the model is on GPU, what error message will I get? Show me an example."*

### Exercises

- [ ] Run `uv run python -c "from training.device import get_device; print(get_device())"`. It should print `cpu` on your local machine.
- [ ] Run `uv run python -m training.train_reinforce 200` and verify it prints `device=cpu` in the header and trains normally. No regressions.
- [ ] **Audit the codebase.** Search for every `torch.tensor(` call across all Python files. For each one, verify it either (a) specifies `device=`, (b) is in code that doesn't run during training, or (c) is moved to device immediately after creation. This is the most common source of device mismatch bugs.
- [ ] **Predict before running.** If you call `model.to("cuda")` on a machine without a GPU, what happens? Try it: `uv run python -c "import torch; m = torch.nn.Linear(3,3); m.to('cuda')"`. Read the error message carefully — you'll see it if your Databricks cluster doesn't have a GPU.
- [ ] **Thought experiment.** Moving a tensor from CPU to GPU (and back) has overhead — data must cross the PCIe bus. For a single board encoding (3x6x7 = 126 floats = 504 bytes), this transfer takes ~10 microseconds. For a batch of 64 (32 KB), it's similar — the bus bandwidth is high but the latency is fixed. What does this mean for REINFORCE (one forward pass at a time) vs. PPO (batches of 64)?
<!-- REINFORCE pays the fixed latency cost on every single forward pass during an episode, getting minimal parallelism benefit. PPO amortizes the transfer latency over a batch of 64, and gets massive parallelism benefit from the GPU. This is why REINFORCE benefits less from GPU than batch methods. The transfer overhead is negligible for both — it's the computation parallelism that matters. -->

### Suggested File Structure

```
c4/
├── training/
│   ├── device.py              # New: get_device() utility
│   ├── train.py               # Updated: device detection
│   ├── train_reinforce.py     # Updated: device detection
│   └── train_ppo.py           # Updated: device detection
├── env/
│   └── encoding.py            # Updated: optional device parameter
├── agents/
│   ├── dqn_agent.py           # Updated: device parameter
│   ├── reinforce_agent.py     # Updated: device parameter
│   └── ppo_agent.py           # Updated: device parameter
```

---

## Module 25 — Experiment Tracking with MLflow

**Goal:** Add MLflow logging to your training loops so every run's hyperparameters, metrics, and artifacts are tracked in a searchable database.

### Key Concepts

- **The experiment tracking problem**: You've run dozens of training experiments. Each produced a `metadata.json`, `metrics.json`, and `model.pt` in a timestamped directory. To compare runs, you'd need to open each `metadata.json`, extract the hyperparameters, open each `metrics.json`, extract the final win rate, and build a comparison table by hand. MLflow does this automatically.

- **MLflow's data model**: MLflow organizes work into **experiments** (groups of related runs) and **runs** (individual training executions). Each run logs:
  - **Parameters**: Hyperparameters set at the start (learning rate, gamma, opponent type). Logged once per run.
  - **Metrics**: Values that change over training (loss, win rate, entropy). Logged at each eval interval with a step number, creating time-series curves.
  - **Artifacts**: Files produced by the run (model checkpoint, training plots, metadata). Uploaded at the end.
  - **Tags**: Free-form labels (algorithm name, git commit hash).

- **MLflow on Databricks**: Databricks includes MLflow as a built-in service — no setup required. When your training code calls `mlflow.log_param(...)`, it automatically routes to the Databricks workspace's tracking server. The MLflow UI is accessible from the Databricks sidebar. Locally, MLflow can either connect to the Databricks tracking server (if configured) or fall back to a local `mlruns/` directory.

- **What to log and what not to log**: Log things that help you compare runs and reproduce results. Log hyperparameters (all of them — disk is cheap, forgotten parameters are expensive). Log metrics at regular intervals (every eval, not every episode — too much data is noise). Log the model checkpoint and the training summary plot as artifacts. Don't log intermediate checkpoints (waste of storage), raw episode data (too large), or information already in git (code, architecture).

- **The MLflow run lifecycle**:
  ```python
  mlflow.start_run(run_name="reinforce_lr1e-3_optimal")
  mlflow.log_params({"lr": 1e-3, "gamma": 0.99, "opponent": "optimal"})
  # ... training loop ...
  mlflow.log_metrics({"win_rate": 72.5, "loss": 0.34}, step=5000)
  # ... more training ...
  mlflow.log_artifact("checkpoints/run_.../model.pt")
  mlflow.log_artifact("checkpoints/run_.../training_summary.png")
  mlflow.end_run()
  ```

### What to Ask Claude to Build

1. **Add MLflow dependency:** *"Add `mlflow>=2.20.0` to pyproject.toml's dependencies list."*

2. **MLflow utility module:** *"Create training/mlflow_utils.py with helper functions that wrap common MLflow operations: (1) `setup_mlflow(experiment_name)` that sets the experiment, falling back to local file tracking if the Databricks tracking server is unreachable, (2) `log_training_start(params, run_name)` that starts a run and logs hyperparameters, (3) `log_metrics(metrics_dict, step)` that logs a dictionary of metrics, (4) `log_artifacts(run_dir)` that logs all files in a checkpoint directory as artifacts, (5) `end_run()` that closes the current run. Handle errors gracefully — if MLflow is unavailable, print a warning and continue training (logging should never crash training)."*

3. **Integrate into training loops:** *"Update training/train_reinforce.py to use the MLflow utilities. At the start of train(), call setup_mlflow and log_training_start with all hyperparameters from agent.get_hyperparams(). At each eval interval, call log_metrics with the current win rate, loss, entropy, and alpha values. At the end of training, call log_artifacts with the run directory to upload model.pt, metrics.json, metadata.json, and the training summary plot. Call end_run() in a finally block so the run closes even if training crashes. Do the same for training/train.py (DQN) and training/train_ppo.py (PPO)."*

### After Claude Generates the Code

1. **Read `mlflow_utils.py` carefully.** The fallback behavior is critical — when running locally without a Databricks connection, MLflow should fall back to local file storage (`mlruns/` directory), not crash. Find the try/except that handles this. What exception does it catch?

2. **Find the `end_run()` call.** It should be in a `finally` block:
   ```python
   try:
       # ... training loop ...
   finally:
       end_run()
   ```
   Without `finally`, a training crash leaves the MLflow run in a "running" state forever. This is a common MLflow bug that clutters the UI.

3. **Check what gets logged as parameters vs. metrics.** Parameters are things you set *before* training (learning rate, gamma, opponent type). Metrics are things you *measure during* training (loss, win rate). Logging a metric as a parameter (e.g., logging final win rate as a parameter) prevents you from seeing how it evolved over time. Logging a parameter as a metric (e.g., logging learning rate at every step) wastes space since it never changes.

4. **Test locally.** Run `uv run python -m training.train_reinforce 500`. After it finishes, check if a `mlruns/` directory was created. If so, MLflow logged locally. Run `uv run mlflow ui` and open `http://localhost:5000` to browse your runs. You should see the hyperparameters, the metrics curves, and the artifacts.

5. **Ask Claude:** *"What's the difference between `mlflow.log_metric` and `mlflow.log_metrics`? When would I use each?"*

### Exercises

- [ ] Run a short training (`uv run python -m training.train_reinforce 1000`) and verify MLflow logs appear. If `mlflow ui` works, explore the UI — click on the run, look at the parameters, click "Metrics" to see the curves.
- [ ] Run two training sessions with different learning rates. In the MLflow UI (or `mlruns/` directory), find both runs and compare their win rate curves. This is the power of experiment tracking — two clicks instead of digging through directories.
- [ ] **Experiment — artifact size.** Check the size of your `model.pt` files in the checkpoint directories. MLflow uploads these as artifacts. For a ~5MB model, this is negligible. What if your model were 5GB? Think about when artifact logging becomes expensive and how you'd handle it.
- [ ] **Read the MLflow docs.** Search for `mlflow.pytorch.log_model()`. This is an alternative to logging the raw `.pt` file — it logs the model in MLflow's standard format, which enables serving and model registry features. For this project, logging the raw `.pt` file is simpler. But understand that the option exists.
- [ ] **Thought experiment.** MLflow logs metrics as time series (value + step). Your current `metrics.json` also stores time series. After adding MLflow, you have the same data in two places. Is this redundant? Should you remove `metrics.json`? Think about what happens if MLflow is unavailable, and whether local tools (like your `save_plots()` function) need the JSON.
<!-- Keep both. metrics.json is the local source of truth that works offline and feeds save_plots(). MLflow is the comparison/search layer. Redundancy is cheap; data loss is expensive. If you remove metrics.json and MLflow is down during a training run, you lose everything. -->

### Suggested File Structure

```
c4/
├── pyproject.toml             # Updated: mlflow dependency
├── training/
│   ├── mlflow_utils.py        # New: MLflow helper functions
│   ├── train.py               # Updated: MLflow logging
│   ├── train_reinforce.py     # Updated: MLflow logging
│   └── train_ppo.py           # Updated: MLflow logging
```

---

## Module 26 — Packaging and Deployment with Databricks Asset Bundles

**Goal:** Package your project as a deployable artifact and define Databricks jobs that run your training scripts on GPU clusters.

### Key Concepts

- **The deployment problem**: Your training scripts run via `uv run python -m training.train_reinforce`. This assumes your source code, virtual environment, and dependencies are all present on the machine. A Databricks cluster is a fresh VM with none of this. You need to **package** your code so it can be installed on the cluster, and **define a job** that tells Databricks what to run and what hardware to use.

- **Python wheels**: A wheel (`.whl` file) is Python's standard package format — a zip file containing your code and metadata. When you install a package with `pip install`, you're usually installing a wheel. By packaging your project as a wheel, you can install it on any machine (including a Databricks cluster) with `pip install c4-0.1.0-py3-none-any.whl`. All your modules (`agents/`, `models/`, `training/`, `env/`) become importable.

- **Entry points**: A wheel can define **console entry points** — shell commands that map to Python functions. Instead of `python -m training.train_reinforce`, you get `train-reinforce` as a standalone command. Databricks' `python_wheel_task` uses these entry points to run your code.

- **Databricks Asset Bundles (DABs)**: A DAB is a declarative configuration (your `databricks.yml`) that describes everything Databricks needs: what code to deploy, what jobs to define, what clusters to create. The `databricks bundle deploy` command reads this file, builds your wheel, uploads it to the workspace, and creates the job definitions. `databricks bundle run <job>` triggers a job.

- **Cluster configuration**: Each job specifies a cluster — what VM type, what Databricks runtime, how many workers. For RL training:
  - **Single node** (no Spark workers) — your training is pure PyTorch, not distributed
  - **GPU runtime** (e.g., `15.4.x-gpu-ml-scala2.12`) — includes PyTorch, CUDA, and MLflow pre-installed
  - **GPU VM type** (e.g., `Standard_NC6s_v3` on Azure) — the smallest GPU option with a Tesla K80
  - **Ephemeral cluster** — created when the job starts, destroyed when it finishes. You only pay for active compute time.

- **Job parameters and variables**: DABs support variables that you can override at runtime. Define defaults in `databricks.yml` and override when launching: `databricks bundle run train_reinforce --params episodes=20000,opponent=optimal`. This lets you run hyperparameter sweeps without editing the config file.

### What to Ask Claude to Build

1. **Add entry points to pyproject.toml:** *"Add a [project.scripts] section to pyproject.toml that defines entry points for each training script: `train-dqn`, `train-reinforce`, and `train-ppo`. Each entry point should map to a `main()` function in the corresponding training module. Then wrap the existing `if __name__ == '__main__':` code in each training script into a `main()` function that the entry points can call."*

2. **Update databricks.yml with jobs:** *"Update databricks.yml to define three Databricks jobs — one for each training algorithm (DQN, REINFORCE, PPO). Each job should: (1) use a python_wheel_task that references the c4 wheel and the corresponding entry point, (2) create a single-node GPU cluster using the 15.4.x-gpu-ml-scala2.12 runtime and Standard_NC6s_v3 node type, (3) accept episode count and opponent type as configurable parameters with sensible defaults. Add an artifacts section that tells the bundle to build the wheel from the project root. Add a variables section with defaults for episodes (10000) and opponent (heuristic). Pass these variables to the job task parameters."*

### After Claude Generates the Code

1. **Read the updated pyproject.toml.** The `[project.scripts]` section should look like:
   ```toml
   [project.scripts]
   train-dqn = "training.train:main"
   train-reinforce = "training.train_reinforce:main"
   train-ppo = "training.train_ppo:main"
   ```
   This means: when someone runs the command `train-reinforce`, Python calls the `main()` function in `training/train_reinforce.py`.

2. **Read the updated training scripts.** The existing code under `if __name__ == "__main__":` should now be inside a `main()` function. The `if __name__` block should just call `main()`. This is a small but important refactor — without it, the entry points can't call the code.

3. **Read databricks.yml carefully.** Understand each section:
   - `artifacts` — tells the bundle to build a wheel from your project
   - `resources.jobs` — defines the three training jobs
   - `new_cluster` — specifies the GPU cluster configuration
   - `variables` — defines configurable parameters with defaults

4. **Validate the bundle.** Run `databricks bundle validate`. This checks that `databricks.yml` is syntactically correct and all referenced files exist. It doesn't deploy anything — it's a dry run.

5. **Build the wheel locally.** Run `uv build`. This creates a `.whl` file in the `dist/` directory. Inspect it: `unzip -l dist/c4-0.1.0-py3-none-any.whl`. Verify your modules (`agents/`, `models/`, `training/`, `env/`) are included.

6. **Test the entry point locally.** After `uv build && uv pip install dist/c4-*.whl`, run `train-reinforce 100`. It should work identically to `uv run python -m training.train_reinforce 100`. If it doesn't, the `main()` refactor has a bug.

7. **Ask Claude:** *"What does `spark_conf: spark.databricks.cluster.profile: singleNode` mean in the cluster config? Why do we need it when num_workers is 0?"*

### Exercises

- [ ] Run `databricks bundle validate` and fix any errors. This is the first step before deploying.
- [ ] Build the wheel with `uv build` and inspect the contents. Is every Python file you need included? Are test files excluded? (They should be — tests don't need to run on the cluster.)
- [ ] Test the entry points locally: `uv run train-reinforce 200`. Verify it works. Then test with arguments: `uv run train-reinforce 500 --opponent optimal --reward-shaping`. All CLI arguments should still work.
- [ ] **Read the cluster config.** Look up the `Standard_NC6s_v3` VM on Azure's pricing page. How much GPU memory does it have? How many CUDA cores? Is this sufficient for your model size (~389K parameters)?
- [ ] **Thought experiment.** Your `databricks.yml` defines ephemeral clusters — they're created per-job and destroyed after. An alternative is a **persistent cluster** that stays running between jobs. What are the trade-offs? When would each approach be better?
<!-- Ephemeral: pay only for active training, no idle costs, guaranteed clean state each run. But startup time (~5 minutes for GPU cluster) adds overhead. Persistent: instant job start, can share across runs, but you pay while idle. For occasional training runs (this project), ephemeral is better. For rapid iteration (running 20 experiments in a row), persistent saves the 5-minute spin-up per job. -->
- [ ] **Predict before running.** The Databricks ML runtime includes PyTorch and MLflow pre-installed. But your `pyproject.toml` also lists `torch` as a dependency in the wheel. Will there be a conflict? Think about what happens when `pip install c4-0.1.0.whl` tries to install torch on a cluster that already has it.
<!-- pip will see that torch is already installed and satisfies the version requirement, so it skips it. No conflict. The wheel's dependency list serves as a minimum version guarantee, not a forced reinstall. This is by design — ML runtimes pre-install optimized, CUDA-linked versions of PyTorch. Installing from pip would give a CPU-only or different CUDA version. -->

### Suggested File Structure

```
c4/
├── pyproject.toml             # Updated: [project.scripts] entry points
├── databricks.yml             # Updated: jobs, artifacts, variables
├── training/
│   ├── train.py               # Updated: main() wrapper
│   ├── train_reinforce.py     # Updated: main() wrapper
│   └── train_ppo.py           # Updated: main() wrapper
```

---

## Module 27 — The Full Workflow: Train, Monitor, Download, Play

**Goal:** Execute the complete cycle: deploy to Databricks, run GPU training, monitor in MLflow, download the model, and play against it locally.

### Key Concepts

- **The deploy-run-monitor-download cycle**: This is the core loop of cloud ML development:
  1. **Deploy**: `databricks bundle deploy` — builds wheel, uploads to workspace, creates/updates jobs
  2. **Run**: `databricks bundle run train_reinforce` — triggers the job on a GPU cluster
  3. **Monitor**: Watch training in the Databricks UI or MLflow experiment tracker
  4. **Download**: Pull the trained model from MLflow to your local machine
  5. **Play**: Load the model locally and play against it

- **Model portability across devices**: A model trained on GPU saves tensors tagged as CUDA. Loading on CPU requires `map_location="cpu"` (handled in Module 24). The weights themselves are identical — GPU vs. CPU affects computation speed, not model quality. A model trained on GPU and loaded on CPU produces the exact same predictions.

- **MLflow artifact retrieval**: MLflow provides both a CLI and Python API for downloading artifacts:
  ```bash
  # CLI
  mlflow artifacts download --run-id <run_id> --dst-path checkpoints/
  
  # Python
  client = mlflow.tracking.MlflowClient()
  client.download_artifacts(run_id, "", "checkpoints/")
  ```
  On Databricks, each MLflow run has a unique `run_id` (a UUID). You find it in the MLflow UI or from the job output.

- **Connecting local MLflow to Databricks**: For the download to work from your local machine, MLflow needs to know where the tracking server is. Two options:
  - Set `MLFLOW_TRACKING_URI=databricks` in your environment (uses the Databricks CLI's auth)
  - Call `mlflow.set_tracking_uri("databricks")` in your script
  
  Both require the Databricks CLI to be authenticated (`databricks auth login`).

### What to Ask Claude to Build

1. **Download script:** *"Create download_model.py at the project root. It should accept an MLflow run ID as a command-line argument and download all artifacts from that run to a local directory. Default destination is a new subdirectory under checkpoints/ named after the run ID. Use mlflow.set_tracking_uri('databricks') to connect to the Databricks workspace. Print what was downloaded and where. If the run ID is not provided, list the 5 most recent runs from the /c4/training experiment with their run IDs, names, and key metrics so the user can pick one."*

2. **Update play.py:** *"Update play.py to accept a --mlflow-run flag. When provided, download the model from that MLflow run (using the logic from download_model.py) to a temporary local directory, then load and play against it. This is a convenience shortcut so the user doesn't need to run download_model.py separately."*

### After Claude Generates the Code

1. **Read download_model.py.** The script has two modes: (a) with a run ID, download artifacts; (b) without a run ID, list recent runs. The listing mode is important — after a training job finishes, you need the run ID to download the model. The listing should show enough information (name, win rate, timestamp) to pick the right run.

2. **Check the tracking URI.** The script sets `mlflow.set_tracking_uri("databricks")`. This relies on Databricks CLI authentication. If you haven't authenticated, it will fail. Verify by running `databricks auth login` first.

3. **Check the map_location.** When `play.py` loads a GPU-trained model, `torch.load()` must use `map_location="cpu"` (or `map_location=device` as implemented in Module 24). Verify this is in place — without it, loading a GPU checkpoint on your CPU laptop will crash.

### Executing the Full Workflow

This is the hands-on part. Follow these steps in order.

**Step 1: Deploy**
```bash
databricks bundle deploy
```
This builds the wheel, uploads it to your Databricks workspace, and creates the job definitions. Watch the output — it should show the wheel being built and the jobs being created or updated.

**Step 2: Run**
```bash
databricks bundle run train_reinforce
```
This triggers the REINFORCE training job. By default it uses the parameters from `databricks.yml` (10,000 episodes, heuristic opponent). To customize:
```bash
databricks bundle run train_reinforce --params episodes=5000,opponent=optimal
```
The command prints a URL to the job run in the Databricks UI.

**Step 3: Monitor**
Open the URL from Step 2 in your browser. You'll see:
- The cluster spinning up (~3–5 minutes for GPU)
- The training output (same as your local terminal)
- After completion: a link to the MLflow run

In the MLflow UI, explore:
- Parameters tab — all hyperparameters
- Metrics tab — click any metric to see the time-series curve
- Artifacts tab — your model.pt, plots, and metadata

**Step 4: Download**
```bash
uv run python download_model.py
```
This lists recent runs. Find the one you just ran, note its run ID, then:
```bash
uv run python download_model.py <run_id>
```
This downloads the model checkpoint and other artifacts to `checkpoints/<run_id>/`.

**Step 5: Play**
```bash
uv run python play.py trained --run checkpoints/<run_id>
```
Or the shortcut:
```bash
uv run python play.py trained --mlflow-run <run_id>
```

### Exercises

- [ ] **Execute the full workflow.** Deploy, run a short REINFORCE training (2,000 episodes), monitor in the UI, download the model, and play against it. This is the core deliverable of this chapter.
- [ ] **Compare GPU vs. CPU training time.** Run the same training (2,000 episodes, same hyperparameters) locally and on Databricks. Compare the elapsed time reported in `metadata.json`. How much faster is GPU?
- [ ] **Run a hyperparameter comparison.** Deploy 3 training jobs with different learning rates (1e-4, 3e-4, 1e-3). After all finish, open the MLflow UI and use the "Compare" feature to see all three runs' win rate curves on the same graph. Which learning rate won?
- [ ] **Test cross-device loading.** Take the GPU-trained `model.pt` you downloaded and load it in a Python REPL:
  ```python
  import torch
  checkpoint = torch.load("checkpoints/<run_id>/model.pt", map_location="cpu", weights_only=True)
  print(checkpoint.keys())
  ```
  This should work. Now try without `map_location` — what error do you get?
- [ ] **Break the workflow intentionally.** Run `databricks bundle run train_reinforce` and then immediately cancel the job from the UI. What happens to the MLflow run? Is it stuck in "running" state? If so, find the run in the UI and manually end it. This is why the `finally` block from Module 25 matters.

### Suggested File Structure

```
c4/
├── download_model.py          # New: MLflow artifact download CLI
├── play.py                    # Updated: --mlflow-run flag
```

---

## Module 28 — Synthesis: Local vs. Cloud, and When Each Matters

**Goal:** Develop a mental model for when to train locally vs. in the cloud, and how this workflow scales to larger ML projects.

### No Code This Module

This is the conceptual capstone. Revisit what you've built with fresh eyes.

### The Decision Framework

After this chapter, you should ask these questions before starting any training run:

1. **How long will it take?** If < 5 minutes on CPU, train locally. No point spinning up a cluster for something your laptop handles while you get coffee.

2. **Am I iterating on code or hyperparameters?** Code changes need the edit-run-debug loop to be fast — train locally with small episode counts. Hyperparameter sweeps benefit from parallel cloud jobs.

3. **Do I need to compare this run to previous runs?** If yes, MLflow logging should be on. If you're just debugging a crash, you don't need experiment tracking — run locally without MLflow.

4. **Will someone else need this model?** If the model needs to be shared, reproducible, or deployed, train on Databricks where the artifact is in MLflow. If it's a throwaway experiment, train locally.

### What You Built vs. Production ML

Your cloud training setup is a simplified version of what production ML teams use. Here's what scales and what would need to change:

| This project | Production | Why |
|---|---|---|
| Single-node GPU | Multi-GPU / multi-node | Larger models don't fit on one GPU |
| Manual `bundle run` | Scheduled jobs or CI/CD triggers | Automated retraining on new data |
| MLflow experiment tracking | MLflow + model registry + serving | Deploy models as API endpoints |
| Download model manually | Model registry + automated deployment | Models move from training to serving without human intervention |
| One developer | Team with roles (ML engineer, data engineer, MLOps) | Specialization as scale increases |

Your project's simplicity is intentional — single-node training with manual deployment is the right starting point. But every piece has a production-grade extension, and the concepts (packaging, job definitions, artifact tracking, cross-device portability) are the same.

### Exercises

- [ ] **Cost analysis.** Look at your Databricks workspace usage (if available). How much did your training runs cost in compute? Compare to the cost of a local GPU (e.g., an RTX 4070 at ~$600). How many cloud training runs would you need before buying a GPU breaks even?

- [ ] **Design exercise.** A teammate wants to run a large hyperparameter sweep: 5 learning rates x 4 entropy coefficients x 3 opponents = 60 combinations. Design a workflow using Databricks jobs. Should each combination be a separate job? A single job with a loop? How would you aggregate results? Sketch the `databricks.yml` structure.

- [ ] **Reflection.** What was the hardest part of this chapter? Likely candidates: (a) getting device handling right across every file, (b) understanding the wheel packaging, (c) configuring Databricks authentication. Each of these is a common pain point in ML deployment. Which would you study more deeply?

- [ ] **Write a one-paragraph guide** for a teammate who wants to train a model on Databricks for the first time. Assume they have the project cloned and `databricks` CLI installed. What are the exact commands?
<!-- databricks auth login (authenticate), databricks bundle deploy (build + upload), databricks bundle run train_reinforce (run job), wait for completion + note the MLflow run ID, uv run python download_model.py <run_id> (download model), uv run python play.py trained --run checkpoints/<run_id> (play). That's 5 commands. -->

- [ ] **The meta-question.** This chapter didn't change any RL algorithm. The same REINFORCE, DQN, and PPO code runs on both CPU and GPU. The same model checkpoint works on both. Everything you added — device handling, MLflow, packaging, job definitions — is infrastructure. Is this infrastructure worth the effort for a project this size? At what scale does it become essential? Think about this in terms of: number of training runs, number of collaborators, and model size.

---

## Appendix — Recommended Resources

| Resource | What It Covers |
|----------|---------------|
| [MLflow Documentation](https://mlflow.org/docs/latest/) | Official docs — experiments, tracking, model registry |
| [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles/index.html) | DAB configuration reference |
| [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) | Device management, memory, best practices |
| [Python Packaging Guide](https://packaging.python.org/en/latest/) | Wheels, entry points, pyproject.toml |
| [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html) | Databricks-specific MLflow features |
| [Azure NC-series VMs](https://learn.microsoft.com/en-us/azure/virtual-machines/nc-series) | GPU VM specs and pricing |

---

## Full Project Structure (End of Chapter 4)

```
c4/
├── pyproject.toml                     # Updated: mlflow dep, entry points
├── databricks.yml                     # Updated: jobs, artifacts, variables
├── game.md
├── lesson_plan.md                     # Chapter 1
├── lesson_plan_chapter_2.md           # Chapter 2
├── lesson_plan_chapter_3.md           # Chapter 3
├── lesson_plan_chapter_4.md           # This file
├── download_model.py                  # New: MLflow artifact download
├── game_runner.py
├── play.py                            # Updated: --mlflow-run flag
├── env/
│   ├── __init__.py
│   ├── connect_four.py
│   └── encoding.py                    # Updated: optional device param
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   ├── human_agent.py
│   ├── optimal_agent.py
│   ├── dqn_agent.py                   # Updated: device parameter
│   ├── reinforce_agent.py             # Updated: device parameter
│   └── ppo_agent.py                   # Updated: device parameter
├── models/
│   ├── __init__.py
│   ├── q_network.py
│   ├── policy_network.py
│   └── value_network.py
├── training/
│   ├── __init__.py
│   ├── device.py                      # New: get_device() utility
│   ├── mlflow_utils.py                # New: MLflow helpers
│   ├── checkpoint.py
│   ├── replay_buffer.py
│   ├── prioritized_replay_buffer.py
│   ├── n_step_buffer.py
│   ├── rollout_buffer.py
│   ├── reward_shaping.py
│   ├── evaluate.py
│   ├── train.py                       # Updated: device, MLflow, main()
│   ├── train_reinforce.py             # Updated: device, MLflow, main()
│   └── train_ppo.py                   # Updated: device, MLflow, main()
├── notebooks/
├── checkpoints/
└── tests/
```
