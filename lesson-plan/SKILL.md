---
name: lesson-plan
description: Generate a hands-on, project-based lesson plan (one chapter file) that teaches a technical topic through AI-pair-programming — the learner directs Claude Code to build, then reads, experiments, and debugs to build deep intuition. Use when the user wants to create a lesson plan, learning curriculum, study plan, course chapter, or tutorial for a technical topic (programming, CS, ML, math), or says "teach me X", "make a lesson plan", "next chapter".
---

# Lesson Plan

Produce a lesson-plan chapter that teaches a technical topic by **directing Claude Code to build a real project**, then reading, experimenting, and debugging. The learning happens in the *experiments and exercises*, not in typing code. Intuition is the goal.

Output: a single markdown file per chapter (e.g. `lesson_plan_chapter_N.md`).

## Workflow

1. **Interview the learner first — always.** Do not draft until you've asked. See [Interview](#interview).
2. **Inspect existing context.** If this continues a project, read the prior `lesson_plan*.md` files and skim the repo so module file paths, naming, and infrastructure are accurate. Match the established voice and structure.
3. **Draft the chapter** following [TEMPLATE.md](TEMPLATE.md) for structure and [PEDAGOGY.md](PEDAGOGY.md) for the techniques that make it teach. Write to `lesson_plan_chapter_N.md` (or the project's convention).
4. **Review with the learner.** Show the module outline first for sign-off, then expand. Ask what's missing, too shallow, or too deep.

## Interview

Ask these before drafting. Group them with AskUserQuestion where it helps; let the learner skip what doesn't apply. Probe — the answers drive everything.

- **Topic & artifact.** What do you want to learn, and what will you build to learn it? (Every chapter is anchored to a concrete project.)
- **Starting point.** What do you already know? What did the previous chapter(s) cover? (Calibrates concept depth and what to assume.)
- **Concepts to internalize.** Which specific ideas must the learner gain *intuition* for — not just use? List them; each becomes a module's conceptual spine.
- **Scope of this chapter.** New project or continuing one? Roughly how many modules? Where does this chapter start and stop?
- **Theory vs. build balance.** Some modules can be pure concept (no code), others heavy build. How much new code should the learner direct vs. reflect on existing code?
- **Difficulty & pace.** Beginner building first intuition, or advanced learner connecting deep ideas? Short focused chapter or comprehensive?

If the learner is vague, ask follow-ups. A precise interview is the difference between a generic outline and a chapter that teaches.

## What makes these plans work

Read [PEDAGOGY.md](PEDAGOGY.md) before drafting — it's the heart of the skill. The signatures:

- **Intuition pumps** — every abstract concept gets a concrete analogy (the appraiser, the sandwich, the glowing afterimage).
- **Suggested prompts** — copy-pasteable instructions to Claude Code in *italics*, so the learner practices directing the AI.
- **"Predict before running"** and **"break it on purpose"** exercises — operational knowledge beats reading knowledge.
- **Hidden answers** in `<!-- HTML comments -->` after hard questions, so learners can self-check.
- **Connect new theory to code already written** — "you built X without knowing it."
- **Checkbox exercises** mixing hands-on runs, compute-by-hand, thought experiments, and design questions.

## Review checklist

- [ ] Interview happened and answers shaped the modules
- [ ] Every module: Goal → Key Concepts → What to Build → After → Exercises
- [ ] Each key concept has at least one intuition pump
- [ ] File paths / infrastructure match the actual repo (if continuing)
- [ ] Hard questions have hidden `<!-- answers -->`
- [ ] Appendix of resources + full project structure at the end
