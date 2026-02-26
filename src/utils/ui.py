from rich.console import Console
from rich.panel import Panel
from contextlib import contextmanager
from src.utils.summary_logger import get_logger

console = Console()

__all__ = ["agent","system","info","warn","error", "wait","wait_llm","wait_tool", "console"]



def agent(output: str, *, action: str | None = None, return_code: int | None = None, step: int | None = None) -> None:
    if action and return_code is not None:
        console.print(Panel(f"Executed {action}, with Return Code {return_code}", title=f"🔧Tool Executed (Step {step})", style="yellow", title_align="left"))
        get_logger().log_operation(action or "(no action)", output, return_code if return_code is not None else -1, step or 0)
    console.print(Panel(_truncate(output), title=f"🤖Agent Response (Step {step})", style="green", title_align="left"))

def system(content: str, actions: str = "", step: int | None = None, reasoning: str = "") -> None:
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
    get_logger().log_llm_response(content, reasoning, step or 0)


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
    with console.status(f"[{color}]{label}…[/]"):
        yield

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