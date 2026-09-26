"""The agent loop under measurement: retrieve, choose, call, grade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from atg.catalog import Catalog
from atg.events import Trajectory
from atg.render import Format, ToolView, view
from atg.retrieval.strategies import Allowed, Shortlist, Strategy
from atg.tasks import Action, Clarify, Grade, Outcome, Task, ToolCall, grade


class Decider(Protocol):
    """Whatever stands in for the model: the simulator, or real Gemini."""

    def __call__(self, views: Sequence[ToolView], query: str) -> Action: ...


@dataclass(slots=True)
class Attempt:
    """One (task, strategy) measurement point."""

    task_id: str
    strategy: str
    task_kind: str
    catalog_size: int
    k: int
    fmt: Format
    gold_tool: str
    gold_rank: int | None
    retrieved: bool
    action_tool: str | None
    outcome: Outcome
    success: bool
    detail: str
    prompt_tokens: int
    stages: int
    hallucinated: tuple[str, ...] = ()

    @property
    def recall_hit(self) -> bool:
        """Whether recall@k counts this as a hit: gold in the shortlist."""
        return self.retrieved


@dataclass(slots=True)
class RunStats:
    attempts: list[Attempt] = field(default_factory=list)

    def add(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)


def run_attempt(
    task: Task,
    strategy: Strategy,
    catalog: Catalog,
    decider: Decider,
    *,
    k: int,
    allowed: Allowed = None,
    catalog_size: int | None = None,
    trajectory: Trajectory | None = None,
) -> Attempt:
    """Retrieve candidates for one task, let the model act, grade the action."""
    shortlist: Shortlist = strategy.retrieve(task.query, k, allowed)
    gold_rank = shortlist.gold_rank(task.gold_tool)
    retrieved = task.gold_tool in shortlist.candidates
    size = catalog_size if catalog_size is not None else (len(allowed) if allowed else len(catalog))

    if trajectory is not None:
        trajectory.emit(
            "decision",
            kind="retrieval",
            strategy=strategy.name,
            task_id=task.task_id,
            catalog_size=size,
            k=k,
            candidates=list(shortlist.candidates),
            gold_rank=gold_rank,
            prompt_tokens=shortlist.prompt_tokens,
            stages=shortlist.stages,
        )
        trajectory.emit(
            "llm_call",
            task_id=task.task_id,
            strategy=strategy.name,
            query=task.query,
            n_candidates=len(shortlist.candidates),
            fmt=shortlist.fmt,
            prompt_tokens=shortlist.prompt_tokens,
        )

    views = [view(catalog.get(name), shortlist.fmt) for name in shortlist.candidates]
    action: Action = decider(views, task.query)

    if trajectory is not None:
        trajectory.emit(
            "llm_result",
            task_id=task.task_id,
            action=_action_repr(action),
        )

    result = grade(task, action, catalog, gold_rank=gold_rank, k=k)

    if trajectory is not None:
        if isinstance(action, ToolCall):
            trajectory.emit("tool_call", task_id=task.task_id, tool=action.tool, args=action.args)
            trajectory.emit(
                "tool_result",
                task_id=task.task_id,
                tool=action.tool,
                ok=result.success,
                outcome=result.outcome,
            )
        elif isinstance(action, Clarify):
            trajectory.emit(
                "human_gate",
                task_id=task.task_id,
                reason="missing_required_argument",
                tool=action.tool,
                missing=action.missing,
            )
        else:
            trajectory.emit("error", task_id=task.task_id, reason="no action produced")

    return Attempt(
        task_id=task.task_id,
        strategy=strategy.name,
        task_kind=task.kind,
        catalog_size=size,
        k=k,
        fmt=shortlist.fmt,
        gold_tool=task.gold_tool,
        gold_rank=gold_rank,
        retrieved=retrieved,
        action_tool=action.tool if action is not None else None,
        outcome=result.outcome,
        success=result.success,
        detail=result.detail,
        prompt_tokens=shortlist.prompt_tokens,
        stages=shortlist.stages,
        hallucinated=result.hallucinated,
    )


def _action_repr(action: Action) -> dict[str, object]:
    if action is None:
        return {"kind": "none"}
    if isinstance(action, Clarify):
        return {"kind": "clarify", "tool": action.tool, "missing": action.missing}
    return {"kind": "tool_call", "tool": action.tool, "args": action.args}


AllowedBuilder = Callable[[Task, int], frozenset[str]]
