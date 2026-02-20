import re
import base64
from typing import TypedDict, List


class CellOutput(TypedDict):
    output_type: str
    name: str
    text: str


class CellExecutionResult(TypedDict):
    execution_count: int
    status: str
    outputs: List[CellOutput]

def format_for_llm(exec_result: CellExecutionResult, if_truncate: bool = False, max_words: int = 500) -> str:
    """Converts execution result into LLM-compatible text format."""
    # Add metadata header
    metadata_parts = []
    if 'execution_count' in exec_result:
        metadata_parts.append(f"Execution Count: {exec_result['execution_count']}")
    if 'status' in exec_result:
        metadata_parts.append(f"Status: {exec_result['status']}")
    
    text_parts = []
    
    for out in exec_result.get('outputs', []):
        msg_type = out.get('msg_type', out.get('output_type', ''))
        content = out.get('content', out)
        
        if msg_type == 'stream':
            text_parts.append(_clean_ansi_codes(content.get('text', '')))
        elif msg_type in ('execute_result', 'display_data'):
            data = content.get('data', {})
            
            # Note presence of images but don't include their data
            if 'image/png' in data:
                text_parts.append('[Image: PNG output]')
            elif 'image/jpeg' in data:
                text_parts.append('[Image: JPEG output]')
            
            # Include text representations
            if 'text/plain' in data:
                text_parts.append(_clean_ansi_codes(data['text/plain']))
            elif 'text/html' in data:
                text_parts.append(_clean_ansi_codes(data['text/html']))
        elif msg_type == 'error':
            ename = content.get('ename', 'Error')
            evalue = content.get('evalue', 'Unknown error')
            text_parts.append(f"\n---ERROR---: {ename}: {evalue}\n")
    
    outputs = ''.join(text_parts)
    if if_truncate:
        outputs = _truncate_cell_output(outputs, max_words=max_words)
    
    # Combine metadata and outputs
    result_parts = []
    if metadata_parts:
        result_parts.append(' | '.join(metadata_parts))
    if outputs:
        result_parts.append(outputs)
    
    return '\n'.join(result_parts) if result_parts else "(No output)"


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
    """Truncate cell output to a reasonable word limit for LLM processing."""
    words = output.split()
    if len(words) <= max_words:
        return output
    
    truncated = ' '.join(words[:max_words])
    remaining_words = len(words) - max_words
    return f"{truncated}\n\n[OUTPUT TRUNCATED - {remaining_words} more words omitted for brevity]"
