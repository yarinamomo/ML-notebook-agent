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
from typing import Iterator

console = Console()
status_renderable = Text("")

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
]



def agent(step: int, output: str, *, action: str | None = None, return_code: int | None = None) -> None:
    if action and return_code is not None:
        console.print(Panel(f"Executed {action}, with Return Code {return_code}", title=f"🔧Tool Executed (Step {step})", style="yellow", title_align="left"))
    console.print(Panel(_truncate(output), title=f"🤖Agent Response (Step {step})", style="green", title_align="left"))

def system(step: int, content: str, actions: str = "", reasoning: str = "") -> None:
    parts = []
    if reasoning:
        # Show reasoning in a dim indented block, like a blockquote
        reasoning_lines = _truncate(reasoning, max_lines=20).split("\n")
        reasoning_block = "\n".join(f"  {line}" for line in reasoning_lines)
        parts.append(f"[dim italic]💭 Reasoning:\n{reasoning_block}[/dim italic]")
    parts.append(_truncate(content))
    if actions:
        parts.append(f"[yellow]Actions:\n {_truncate(actions)}[/yellow]")
    console.print(Panel("\n\n".join(parts), title=f"🧠LLM Response (Step {step})", style="cyan", title_align="left"))


def info(message: str) -> None:
    console.print(f"[bold cyan]▶[/] {message}")

def warn(message: str) -> None:
    console.print(f"[bold yellow]⚠[/] {message}")

def error(message: str) -> None:
    console.print(f"[bold red]✖[/] {message}")


# ---------- Status / waiting helpers ----------

@contextmanager
def wait(label: str, *, color: str = "cyan"):
    """
    Generic spinner for indeterminate waits.
    """
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
def progress_live(total: int, description: str) -> Iterator[tuple[Progress, TaskID]]:
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

def _truncate(content: str, max_lines: int = 50, max_chars_per_line: int = 200) -> str:
    lines = content.split('\n')
    was_truncated_lines = len(lines) > max_lines
    
    # Truncate lines
    truncated_lines = lines[:max_lines]
    
    # Truncate each line if too long
    was_truncated_chars = any(len(line) > max_chars_per_line for line in truncated_lines)
    truncated_lines = [line[:max_chars_per_line] for line in truncated_lines]
    
    result = '\n'.join(truncated_lines)
    
    # Add truncation notices
    notices = []
    if was_truncated_lines:
        notices.append(f"[dim yellow]... ({len(lines) - max_lines} more lines)[/dim yellow]")
    if was_truncated_chars:
        notices.append("[dim yellow]... (some lines truncated)[/dim yellow]")
    
    if notices:
        result += "\n\n" + "\n".join(notices)
    
    return result