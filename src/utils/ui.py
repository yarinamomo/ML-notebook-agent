from rich.console import Console
from rich.panel import Panel
from contextlib import contextmanager

console = Console()

def agent(output: str, action: str | None = None, return_code: int | None = None) -> None:
    subtitle = ""
    if action and return_code is not None:
        console.print(Panel(f"Executed {action}, with Return Code {return_code}", title="🔧Tool Exectued", style="yellow"))
    console.print(Panel(output[:500], title="🤖Agent Response", style="green"))

def system(content: str)-> None:
    console.print(Panel(content[:500], title="🧠LLM Response",style="cyan"))



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
