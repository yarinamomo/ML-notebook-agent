import re
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def clean_output(outputs):
        """
        Consolidates stream outputs and handles carriage returns (\r) 
        to reduce verbose progress bars (like tqdm/PyMC/Rich).
        """
        if not outputs:
            return []

        stdout_text = []
        stderr_text = []
        non_stream_outputs = []
        
        # Track display_data messages - Rich progress bars send many of these
        # We only want to keep the LAST display_data for progress-like outputs
        display_data_msgs = []
        
        for msg in outputs:
            if msg['msg_type'] == 'stream':
                content = msg['content']
                if content['name'] == 'stdout':
                    stdout_text.append(content['text'])
                else:
                    stderr_text.append(content['text'])
            elif msg['msg_type'] == 'display_data':
                display_data_msgs.append(msg)
            else:
                non_stream_outputs.append(msg)

        
        cleaned_outputs = []
        
        if stdout_text:
            merged_stdout = ''.join(stdout_text)
            cleaned_stdout = _clean_stream_text(merged_stdout)
            if cleaned_stdout.strip():
                cleaned_outputs.append({
                    'msg_type': 'stream',
                    'content': {'name': 'stdout', 'text': cleaned_stdout}
                })
        
        if stderr_text:
            merged_stderr = ''.join(stderr_text)
            cleaned_stderr = _clean_stream_text(merged_stderr)
            if cleaned_stderr.strip():
                cleaned_outputs.append({
                    'msg_type': 'stream',
                    'content': {'name': 'stderr', 'text': cleaned_stderr}
                })
        
        # Filter display_data: remove progress-like displays entirely, keep others
        last_progress_msg = None
        last_progress_idx = 0
        idx = 0
        for msg in display_data_msgs:
            if not _is_progress_display(msg):
                idx += 1
                cleaned_outputs.append(msg)
            else:
                last_progress_idx = idx
                last_progress_msg = msg

        if last_progress_msg:
            cleaned_outputs.insert(last_progress_idx, last_progress_msg)
        
        cleaned_outputs.extend(non_stream_outputs)
        
        return cleaned_outputs


def _clean_stream_text(text: str) -> str:
    """Process a merged stream text to handle \r and ANSI codes."""
    # Remove ANSI escape codes
    text = ansi_escape.sub('', text)
            
    # Simulate a terminal line buffer to handle \r correctly
    # Split by \n first to preserve actual newlines
    lines = text.split('\n')
    final_lines = []
    for line in lines:
        if '\r' in line:
                    # Simulate carriage return: only keep what's visible after all \r processing
                    # Each \r moves cursor to start of line, subsequent text overwrites
            segments = line.split('\r')
                    # The last non-empty segment is what would be displayed
            visible = ''
            for segment in segments:
                if segment:
                            # This segment overwrites from the beginning
                    visible = segment
                    if visible.strip():  # Only add non-empty lines
                        final_lines.append(visible)
                elif line.strip():  # Only add non-empty lines
                    final_lines.append(line)
            
    return '\n'.join(final_lines)
        
def _is_progress_display(msg):
    """PYMC SPECIFIC FUNCTION: Check if a display_data message looks like a progress bar."""
    data = msg.get('content', {}).get('data', {})
    text = data.get('text/plain', '')
    progress_indicators = ['Progress', 'Draws', 'Divergences', 'Step size', 
                           'Sampling', 'it/s', 'draws/s', '━', '╸', '╺', 
                           'Output()', 'Elapsed', 'Remaining']
    return any(indicator in text for indicator in progress_indicators)