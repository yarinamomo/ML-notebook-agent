from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from contextlib import contextmanager
from typing import Callable, Iterator
import threading
import time

console = Console()
status_renderable = Text("")

# Global flag to enable/disable UI
UI_ENABLED = True

__all__ = [
    "agent",
    "system",
    "info",
    "warn",
    "error",
    "wait",
    "wait_llm",
    "wait_tool",
    "console",
    "status_renderable",
    "format_eta",
    "progress_live",
    "get_progress_advance_fn",
    "set_ui_enabled",
    "is_ui_enabled",
]


def set_ui_enabled(enabled: bool) -> None:
    """Enable or disable all UI output globally."""
    global UI_ENABLED
    UI_ENABLED = enabled


def is_ui_enabled() -> bool:
    """Check if UI is currently enabled."""
    return UI_ENABLED


def ui_enabled_only(func):
    """Decorator that makes a function return None if UI is disabled."""

    def wrapper(*args, **kwargs):
        if not UI_ENABLED:
            return None
        return func(*args, **kwargs)

    return wrapper


@ui_enabled_only
def agent(
    step: int, output: str, *, action: str | None = None, return_code: int | None = None
) -> None:
    if action and return_code is not None:
        console.print(
            Panel(
                f"Executed {action}, with Return Code {return_code}",
                title=f"🔧Tool Executed (Step {step})",
                style="yellow",
                title_align="left",
            )
        )
    console.print(
        Panel(
            _truncate(output),
            title=f"🤖Agent Response (Step {step})",
            style="green",
            title_align="left",
        )
    )


@ui_enabled_only
def system(step: int, content: str, actions: str = "", reasoning: str = "") -> None:
    parts = []
    if reasoning:
        # Show reasoning in a dim indented block, like a blockquote
        reasoning_lines = _truncate(reasoning, max_lines=20).split("\n")
        reasoning_block = "\n".join(f"  {line}" for line in reasoning_lines)
        parts.append(f"[dim italic]💭 Reasoning:\n{reasoning_block}[/dim italic]")
    parts.append(_truncate(content or ""))
    if actions:
        parts.append(f"[yellow]Actions:\n {_truncate(actions)}[/yellow]")
    console.print(
        Panel(
            "\n\n".join(parts),
            title=f"🧠LLM Response (Step {step})",
            style="cyan",
            title_align="left",
        )
    )


@ui_enabled_only
def info(message: str) -> None:
    console.print(f"[bold cyan]▶[/] {message}")


@ui_enabled_only
def warn(message: str) -> None:
    console.print(f"[bold yellow]⚠[/] {message}")


@ui_enabled_only
def error(message: str) -> None:
    console.print(f"[bold red]✖[/] {message}")


# ---------- Status / waiting helpers ----------


@contextmanager
def wait(label: str, *, color: str = "cyan"):
    """
    Generic spinner for indeterminate waits.
    """
    if not UI_ENABLED:
        yield
        return

    status_renderable.plain = f"{label}…"
    status_renderable.style = color
    try:
        yield
    finally:
        status_renderable.plain = ""
        status_renderable.style = ""


@contextmanager
def wait_llm():
    """
    Spinner specifically for LLM calls.
    """
    with wait("🧠Calling LLM", color="cyan"):
        yield


@contextmanager
def wait_tool(command: str | None = None):
    """
    Spinner for tool / action execution.
    """
    label = f"🔧Executing {command}" if command else "tool"
    with wait(label, color="yellow"):
        yield


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@contextmanager
def progress_live(
    total: int, description: str, force_display: bool = False
) -> Iterator[tuple[Progress, TaskID]]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total} runs"),
        TimeElapsedColumn(),
        TextColumn("ETA {task.fields[eta]}", style="cyan"),
        console=console,
        transient=False,
        auto_refresh=False,
    )

    with Live(
        Group(status_renderable, progress),
        console=console,
        transient=False,
        refresh_per_second=10,
    ):
        task_id = progress.add_task(
            description,
            total=total,
            eta="--:--:--",
        )
        yield progress, task_id


def get_progress_advance_fn(
    progress: Progress,
    task_id: TaskID,
    total_iterations: int,
) -> Callable[[str], None]:
    """Return a thread-safe function that advances progress and recomputes ETA."""
    start_time = time.monotonic()
    completed_count = 0
    lock = threading.Lock()

    def advance_fn(description: str) -> None:
        nonlocal completed_count
        with lock:
            completed_count += 1
            elapsed = time.monotonic() - start_time
            avg_per_run = elapsed / completed_count
            remaining = avg_per_run * (total_iterations - completed_count)
            progress.update(
                task_id,
                advance=1,
                description=description,
                eta=format_eta(remaining),
            )

    return advance_fn


def _truncate(content: str, max_lines: int = 50, max_chars_per_line: int = 200) -> str:
    if content is None:
        content = ""
    lines = content.split("\n")
    was_truncated_lines = len(lines) > max_lines

    # Truncate lines
    truncated_lines = lines[:max_lines]

    # Truncate each line if too long
    was_truncated_chars = any(
        len(line) > max_chars_per_line for line in truncated_lines
    )
    truncated_lines = [line[:max_chars_per_line] for line in truncated_lines]

    result = "\n".join(truncated_lines)

    # Add truncation notices
    notices = []
    if was_truncated_lines:
        notices.append(
            f"[dim yellow]... ({len(lines) - max_lines} more lines)[/dim yellow]"
        )
    if was_truncated_chars:
        notices.append("[dim yellow]... (some lines truncated)[/dim yellow]")

    if notices:
        result += "\n\n" + "\n".join(notices)

    return result
