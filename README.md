# Lesson Plan Skill

An **agent skill** that generates **hands-on, project-based lesson plans** for technical topics — programming, CS, ML, math, and more. It works with any AI coding agent that reads the [`SKILL.md`](lesson-plan/SKILL.md) format.

The premise: you learn a topic best by *directing an AI coding agent to build a real project*, then reading, experimenting, and breaking what you built. The code-typing isn't the lesson — the **experiments and exercises** are. Intuition is the goal.

## What it produces

A single markdown chapter file (e.g. `lesson_plan_chapter_N.md`) structured around a concrete project, where every module follows a **Goal → Key Concepts → What to Build → After → Exercises** arc.

The chapters lean on a few signature techniques:

- **Intuition pumps** — every abstract concept gets a concrete analogy.
- **Suggested prompts** — copy-pasteable instructions for your coding agent, so you practice directing the AI.
- **"Predict before running"** and **"break it on purpose"** exercises — operational knowledge over reading knowledge.
- **Hidden answers** in HTML comments after hard questions, for self-checking.
- **Connecting new theory to code already written** — "you built X without knowing it."


## How to use it

Ask your agent for a lesson plan and the skill kicks in. Triggers include:

- "teach me X"
- "make a lesson plan / curriculum / study plan"
- "next chapter"

The skill **always interviews you first** — topic and artifact, your starting point, the concepts to internalize, scope, and pace — then drafts a module outline for sign-off before expanding it into a full chapter.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`lesson-plan/SKILL.md`](lesson-plan/SKILL.md) | Entry point: workflow, interview questions, review checklist |
| [`lesson-plan/PEDAGOGY.md`](lesson-plan/PEDAGOGY.md) | The teaching techniques that make the plans work |
| [`lesson-plan/TEMPLATE.md`](lesson-plan/TEMPLATE.md) | Chapter and module structure |
| [`bin/install.js`](bin/install.js) | The `npx` installer |
