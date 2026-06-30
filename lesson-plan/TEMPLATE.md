# Lesson Plan — Structural Template

Copy this skeleton. Every chapter and every module follows it. Adapt headings to the domain, but keep the rhythm: **concept → build → inspect → exercise.**

---

## Chapter skeleton

```md
# <Project Name> — Chapter N: <Evocative Title That Names the Big Idea>

<2–4 paragraph intro. Connect to prior chapters. Pose the motivating
questions this chapter answers — phrase them as things the learner has
*done but not yet understood*. State what's new and what's reflection.>

### A note on infrastructure   <!-- only if continuing a project that has changed -->

<Reconcile the plan with how the repo actually looks now: consolidated
entry points, new tooling, renamed files. The conceptual material is
unaffected — only build instructions and paths move.>

---

## Part I: <Theme>          <!-- Parts are optional; use for long chapters -->

---

## Module K — <Title>
...modules...

---

## Appendix — Recommended Resources

| Resource | What It Covers |
|----------|---------------|
| [Title](url) | One-line description |

---

## Full Project Structure (End of Chapter N)

```
<tree reflecting the ACTUAL repo layout, with inline comments marking
which file came from which module/chapter>
```
```

---

## Module skeleton

```md
## Module K — <Title>

**Goal:** <One sentence. What intuition or capability the learner walks away with.>

### Key Concepts

<The conceptual spine. This is where the teaching happens. For each idea:
- Define it plainly.
- Add an intuition pump — a concrete analogy (see PEDAGOGY.md).
- Explain *why it matters* and *when it breaks*.
- Where possible, tie it to something the learner already built.
Use bullet lists, bold the term being defined, and inline math/code where it clarifies.>

### What to Build        <!-- or "What to Ask the Agent to Build"; or "No Code This Module" -->

<Concrete build steps, staged so each piece is understood before the next.
Each step gets a copy-pasteable instruction to the AI coding agent in italics:>

**Suggested prompt:** *"Create <file> that does <precise spec>. Reference
<concept> from this lesson plan. <constraints>."*

### After <Generating / Running the Experiment>

<What to look at and verify. Numbered. Point at the specific method, line,
or curve. Tell the learner what a healthy result looks like and what a
broken one looks like.>

### Exercises

- [ ] <Hands-on run with a concrete success criterion.>
- [ ] **Experiment — <name>:** <change one thing, observe, explain.>
- [ ] **Predict before running:** <make a falsifiable prediction, then test it.>
- [ ] **Compute by hand:** <a small calculation that makes the math concrete.>
      <!-- answer hidden so the learner can self-check -->
- [ ] **Thought experiment:** <an extreme or edge case to reason about.>
- [ ] **Design exercise:** <apply the concept to a hypothetical new problem.>

### Suggested File Structure     <!-- when the module introduces new files -->

```
<the slice of the tree this module adds>
```
```

---

## Module type variants

- **Build module** — heavy "What to Ask the Agent to Build", staged, each piece inspected.
- **Concept module** — replace the build section with `### No Code This Module` and a set of questions (each with a hidden `<!-- answer -->`), plus reflection exercises that revisit existing code with new vocabulary.
- **Experiment module** — a sweep or comparison: a thin script that runs variations, then heavy "After Running" analysis of the curves/results.
- **Synthesis module** — capstone with a decision framework and design-document exercises; usually no new code.

A good chapter mixes these — don't make every module a build module.
