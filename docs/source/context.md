# Context

A **Context** is a plain `@dataclass` (base class in `src/core/module.py`) that
holds the shared state for one engine. Every module in that engine reads its
`inputs` from the context and writes its `outputs` back to it, so modules cooperate
through named fields instead of by calling each other.

The concrete contexts live in `src/types/autoaim.py`:

- `AutoAimContext` — the older/simpler pipeline (panels + a selected target panel).
- `FullStateAutoAimContext` — what the production `FullStateAutoAimEngine` uses. It
  holds the full world model for a frame: `panels`, `target_robot`, the kinematic
  `estimate`, the ballistic `solution`, and the intermediate values the estimation
  and ballistics modules pass between each other.

## How it's used

```python
from dataclasses import dataclass
from typing import Optional
from src.core.module import Context

@dataclass
class AutoAimContext(Context):
    panels: Optional[list] = None
    target_panel: Optional[object] = None
```

A module declares the field names it touches as strings:

```python
# detector writes "panels"; targeting reads "panels", writes "target_robot"
ClassicalDetectorModule(ctx)   # inputs=[],          outputs=["panels"]
TargetingModule(ctx)           # inputs=["panels"],  outputs=["target_robot"]
```

## Why fields are declared by name

Declaring `inputs`/`outputs` as strings is what lets the engine **check the
pipeline before it runs**. `Engine.__init__` calls `_validate_wiring()`, which
enforces two things:

- every input/output name a module declares is an actual field on the context
  dataclass, and
- every input a module reads is produced by some other module's output.

A typo'd field name therefore raises `ValueError` at construction, not three steps
into a run. The flip side: when you add a new module input or output, you must add
the matching field to the context dataclass, or wiring validation will reject it.

## Why a shared context instead of passing arguments

Pipeline stages need overlapping state, and that state grows over time (e.g. the
estimator keeps history between frames). A single typed context means you can add a
new intermediate value without changing any method signatures, and you can inspect
the entire per-frame world state in one place. The context is effectively the
schema for an engine's pipeline.
