from io import StringIO
import re
import tokenize
from typing import List, Optional, cast

from nbformat import NotebookNode

from src.utils.nb_types import CellExecutionResult, CellOutput, NotebookCell
from src.utils.nbformat_helper import get_cell_source


def remove_comments(source_code: str) -> str:
    tokens = tokenize.generate_tokens(StringIO(source_code).readline)
    result = []
    last_lineno = -1
    last_col = 0

    for token in tokens:
        tok_type = token.type
        tok_string = token.string
        start_line, start_col = token.start

        if tok_type == tokenize.COMMENT:
            continue

        if start_line > last_lineno:
            result.append('\n' * (start_line - last_lineno - 1))
            last_col = 0

        if start_col > last_col:
            result.append(' ' * (start_col - last_col))

        result.append(tok_string)
        last_lineno, last_col = token.end

    cleaned_code = ''.join(result)
    # Optional: strip trailing spaces from each line
    cleaned_code = '\n'.join(line.rstrip() for line in cleaned_code.splitlines())
    return cleaned_code #normalize_whitespace(cleaned_code)


def format_cell_source_for_llm(index: int, cell: NotebookCell) -> str:
    return f"# --- [CELL {index}]: ---\n{remove_comments(get_cell_source(cell))}"


def format_exec_result_for_llm(exec_result: Optional[CellExecutionResult], if_truncate: bool = False, max_words: int = 500, no_runtime_output: bool = False) -> str:
    """Converts execution result into LLM-compatible text format."""
    if exec_result is None or not isinstance(exec_result, dict):
        return "(No output)"

    # Add metadata header
    metadata_parts = []
    if 'execution_count' in exec_result:
        metadata_parts.append(f"Execution Count: {exec_result['execution_count']}")
    if 'status' in exec_result:
        metadata_parts.append(f"Status: {exec_result['status']}")
    
    text_parts = []
    raw_outputs = exec_result.get('outputs', [])
    if no_runtime_output:
        raw_outputs = [o for o in raw_outputs if o.get('msg_type', o.get('output_type', '')) == 'error']
    cleaned_oututs = _clean_outputs(raw_outputs)

    for out in cleaned_oututs:
        msg_type = out.get('msg_type', out.get('output_type', ''))
        content = out.get('content', out)
        
        if msg_type == 'stream':
            stream_text = _clean_ansi_codes(content.get('text', ''))
            if if_truncate:
                stream_text = _truncate_stream_output(stream_text, max_words=max_words)
            text_parts.append(stream_text)
        elif msg_type in ('execute_result', 'display_data'):
            data = content.get('data', {})
            
            # Note presence of images but don't include their data
            if 'image/png' in data:
                text_parts.append('[Image: PNG output]')
            elif 'image/jpeg' in data:
                text_parts.append('[Image: JPEG output]')
            
            # Include text representations
            if 'text/plain' in data:
                text = _clean_ansi_codes(data['text/plain'])
                if if_truncate:
                    text = _truncate_cell_output(text, max_words=max_words)
                text_parts.append(text)
            elif 'text/html' in data:
                text = _clean_ansi_codes(data['text/html'])
                if if_truncate:
                    text = _truncate_cell_output(text, max_words=max_words)
                text_parts.append(text)
        elif msg_type == 'error':
            ename = content.get('ename', 'Error')
            evalue = content.get('evalue', 'Unknown error')
            text_parts.append(f"\n---ERROR---: {ename}: {evalue}\n")
    
    outputs = ''.join(text_parts)
    
    # Combine metadata and outputs
    result_parts = []
    if metadata_parts:
        result_parts.append(' | '.join(metadata_parts))
    if outputs:
        result_parts.append(outputs)
    
    return '\n'.join(result_parts) if result_parts else "(No output)"


def format_initial_notebook(notebook: NotebookNode, no_runtime_output: bool = False) -> str:   
    formatted_texts = []
    for i, cell in enumerate(cast(List[NotebookCell], notebook.cells)):
        source = format_cell_source_for_llm(i, cell)
        cell_outputs: List[CellOutput] = cell.get("outputs", [])
        has_error = any(output.get("output_type") == "error" for output in cell_outputs)

        filtered_outputs = [output for output in cell_outputs if output.get("output_type") == "error"] if no_runtime_output else cell_outputs
        exec_result: CellExecutionResult = {
            "outputs": filtered_outputs,
            "execution_count": cell.get("execution_count", None),
            "status": "error" if has_error else "ok",
            "done": True
        }
        formatted_texts.extend([source, "\n OUTPUT:\n", format_exec_result_for_llm(exec_result, if_truncate=True, no_runtime_output=no_runtime_output), "\n\n"])
    return "\n\n".join(formatted_texts)


def _clean_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes and Unicode box-drawing characters from text."""
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    
    # Remove Unicode box-drawing characters (used in model summaries)
    # Unicode ranges: Box Drawing (U+2500–U+257F), Block Elements (U+2580–U+259F)
    box_drawing = re.compile(r'[\u2500-\u257F\u2580-\u259F]+')
    text = box_drawing.sub('', text)
    
    return text


def _truncate_cell_output(output: str, max_words: int = 500) -> str:
    """Truncate text with a head-and-tail strategy for LLM processing."""
    words = output.split()
    if len(words) <= max_words:
        return output

    head_words = max_words // 2
    tail_words = max_words - head_words
    remaining_words = len(words) - max_words
    head = ' '.join(words[:head_words])
    tail = ' '.join(words[-tail_words:])
    return (
        f"{head}\n\n"
        f"[OUTPUT TRUNCATED - {remaining_words} more words omitted for brevity]\n\n"
        f"{tail}"
    )


def _truncate_stream_output(output: str, max_words: int = 500) -> str:
    """Truncate stream output while preserving both the beginning and end."""
    truncated = _truncate_cell_output(output, max_words=max_words)
    if truncated == output:
        return output
    return truncated.replace('[OUTPUT TRUNCATED', '[STREAM TRUNCATED', 1)


def _clean_outputs(outputs: list[CellOutput]) -> list[CellOutput]:
    cleaned_outputs = []
    handled_streams = set()    
    for out in outputs:
        msg_type = out.get('msg_type', out.get('output_type', ''))
        content = out.get('content', out)
        if msg_type == 'stream':
            name = content.get('name', "Unknown")
            if name in handled_streams:
                continue  # Skip duplicate stream output
            else:
                text = _get_all_stream_text(name, outputs)
                out.get("content", {})["text"] = text
                handled_streams.add(name)
                cleaned_outputs.append(out)
        else:
            cleaned_outputs.append(out)

    return cleaned_outputs


def _get_all_stream_text(name: str, outputs: List[CellOutput]) -> str:
    stream_text = ""
    for out in outputs:
        msg_type = out.get('msg_type', out.get('output_type', ''))
        content = out.get('content', out)
        if msg_type == 'stream' and content.get('name') == name:
            stream_text += content.get('text', '')
    return stream_text
