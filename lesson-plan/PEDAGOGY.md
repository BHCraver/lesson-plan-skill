# Pedagogy — The Techniques That Make These Plans Teach

The structure in TEMPLATE.md is a skeleton. *This* is what makes a lesson plan
build intuition instead of just listing facts. Use these techniques liberally;
the example chapters lean on every one of them. Examples below are real, drawn
from the Connect Four RL chapters.

---

## 1. Intuition pumps (non-negotiable)

Every abstract concept gets a concrete, sticky analogy *before or alongside* the
formal definition. The analogy carries the intuition; the math confirms it.

- Value function → **a real-estate appraiser**: doesn't decide what to buy, just
  estimates the price tag. (prediction vs. control)
- n-step return → **a sandwich**: bread is observed rewards (honest), filling is
  the bootstrap estimate (biased). Picking n is choosing the bread-to-filling ratio.
- Eligibility trace → **a glowing afterimage** on each parameter that flares when
  used and dims geometrically; a surprise updates every parameter by how brightly
  it glows.
- MC vs. TD → **two friends predicting a chess result**: one refuses to guess until
  the king topples; the other blurts "+0.8" mid-game — fast but only as good as
  their judgment.

Rule of thumb: if a concept has no analogy, you haven't finished teaching it.

## 2. The "why it matters" and "when it breaks" pair

For each concept, answer two questions the learner will actually hit:
- **Why does this matter for what I'm building?** (ties theory to their project)
- **When is this wrong / when does it break?** (the contrapositive is underrated —
  e.g. "if your agent isn't learning, *prediction* may be broken even when the
  policy update is fine.")

State deep points as memorable one-liners: *"bias survives averaging; variance dies under it."*

## 3. Suggested prompts (the AI-pair-programming engine)

The learner directs an AI coding agent; they don't type the implementation. Every build
step ends with a copy-pasteable instruction in *italics*, specific enough to
produce working code and to teach what a good prompt looks like:

> **Suggested prompt:** *"Create notebooks/04_nstep_comparison.py. The script should
> (a) launch four DQN training runs with n_step ∈ {1,3,5,10}, sharing seed/lr/arch,
> (b) tag each MLflow run, (c) pull win-rate histories and produce an overlay plot."*

Stage builds: "Generate one piece at a time, read it, then move on." Name the file,
the precise spec, and constraints. Reference concepts by name so the learner connects
prompt to theory.

## 4. Active-learning exercises (mix all of these)

Checkbox exercises, varied by cognitive mode:

- **Hands-on run** with a success criterion: *"Train for 1,000 episodes. Is the loss trending down?"*
- **Predict before running**: force a falsifiable prediction first, then test.
  *"If you change ReLU to Sigmoid, will it still learn? Run it and check."*
- **Break it on purpose**: *"Comment out the diagonal win check and run the tests.
  Confirm the right test fails."* / *"Remove the target network — what happens to stability?"*
- **Compute by hand**: a small calculation makes the math physical.
  *"In a 15-move game ending in a win, γ=0.99: compute the 1-step, 3-step, and MC return for move 10."*
- **Thought experiment**: extreme/edge cases. *"A game lasting exactly 2 moves — any
  difference between MC and TD? What about 1,000 moves?"*
- **Design exercise**: transfer to a new problem. *"A friend's game has 500-step
  episodes, dense reward, continuous actions. Where on each axis? Which algorithm?"*

## 5. Hidden answers for self-checking

After a hard conceptual question, embed the answer in an HTML comment so the learner
can attempt it first, then verify:

```md
2. REINFORCE waits for the full episode. Why does this eliminate bias but increase variance?
<!-- Both games end in +1, so the final return is the same. But the *path* differs...
Monte Carlo can't tell the difference. The "no bias" part: you observed the real return. -->
```

Use these on compute-by-hand and conceptual questions, not on open-ended reflection.

## 6. "You already built this"

The most satisfying moments connect new theory back to code the learner wrote
chapters ago without understanding it:

> *"Open your PPO rollout buffer's finalize(). The line `gae = delta + gamma *
> gae_lambda * gae` is the backward-view eligibility trace. You built TD(λ) without
> knowing it."*

Hunt for these. They convert abstract theory into recognition.

## 7. Framing: learning happens around the code, not in typing it

Open project chapters with the contract (adapt to the topic):
- **Before generating**: know what you want and why — Key Concepts prepare you to
  give good instructions and judge the output.
- **After generating**: read every line; if you can't explain it, ask the agent to.
- **Experimentation is the real teacher**: change a hyperparameter, remove a
  component, break something — that builds intuition writing-from-scratch can't.
- **Prompting is a skill** that improves with practice.

## 8. Voice and pacing

- Direct, second-person, intellectually serious but warm. No filler.
- Bold the term being defined on first use.
- Vary module types (build / concept / experiment / synthesis) so the chapter breathes.
- End the chapter with a synthesis or decision framework that generalizes beyond the
  specific project — the learner should leave with transferable judgment, not just a
  finished artifact.
- Close with a resources table and the full project structure tree.
