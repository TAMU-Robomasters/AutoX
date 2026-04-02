# Context

`Context` is the shared state object passed through modules inside an engine.

In AutoX, each module:

- reads required fields from context (`inputs`)
- computes one step
- writes results back to context (`outputs`)

Why this exists:

- keeps module interfaces small
- makes pipeline data flow explicit
- lets state evolve across steps (for example, previous target history)

Example: `AutoAimContext` stores fields like `panels`, `target_panel`, and `prev_target_panel`, so detection and selection modules can cooperate without tightly coupling their implementations.
