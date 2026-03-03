<script>
  import * as Diff from 'diff';

  /** @type {{ oldCode: string, newCode: string }} */
  let { oldCode, newCode } = $props();

  /**
   * Compute line-level diff using the diff library
   */
  let diffLines = $derived.by(() => {
    const changes = Diff.diffLines(oldCode || '', newCode || '');
    const result = [];
    
    for (let i = 0; i < changes.length; i++) {
      const change = changes[i];
      const lines = change.value.split('\n');
      
      // Remove last empty line if it exists (from split)
      if (lines[lines.length - 1] === '') {
        lines.pop();
      }
      
      if (change.added) {
        // Check if previous change was a removal (potential modification)
        const prevResult = result[result.length - 1];
        if (prevResult && prevResult.type === 'remove' && lines.length === 1 && prevResult.lines.length === 1) {
          // Single line change - compute word diff
          const oldLine = prevResult.lines[0];
          const newLine = lines[0];
          
          // Check if lines are similar enough for inline diff
          const maxLen = Math.max(oldLine.length, newLine.length);
          if (maxLen > 0 && Math.abs(oldLine.length - newLine.length) < maxLen * 0.8) {
            const wordDiff = Diff.diffWordsWithSpace(oldLine, newLine);
            
            // Replace the last remove with a modify
            result[result.length - 1] = {
              type: 'modify',
              oldLine: oldLine,
              newLine: newLine,
              wordDiff: wordDiff
            };
            continue;
          }
        }
        
        // Regular addition
        for (const line of lines) {
          result.push({ type: 'add', line: line });
        }
      } else if (change.removed) {
        // Removal
        for (const line of lines) {
          result.push({ type: 'remove', lines: [line] });
        }
      } else {
        // Context (unchanged)
        for (const line of lines) {
          result.push({ type: 'context', line: line });
        }
      }
    }
    
    return result;
  });

  let addCount = $derived(diffLines.filter(l => l.type === 'add' || l.type === 'modify').length);
  let removeCount = $derived(diffLines.filter(l => l.type === 'remove' || l.type === 'modify').length);
</script>

<div class="diff-view">
  <div class="diff-header">
    <span class="diff-title">Cell Edit Diff</span>
    <span class="diff-stats">
      <span class="additions">+{addCount}</span>
      <span class="deletions">-{removeCount}</span>
    </span>
  </div>
  <div class="diff-content">
    {#each diffLines as line}
      {#if line.type === 'context'}
        <div class="diff-line context">
          <span class="diff-gutter"> </span>
          <span class="diff-text">{line.line || ' '}</span>
        </div>
      {:else if line.type === 'add'}
        <div class="diff-line add">
          <span class="diff-gutter">+</span>
          <span class="diff-text">{line.line || ' '}</span>
        </div>
      {:else if line.type === 'remove'}
        <div class="diff-line remove">
          <span class="diff-gutter">-</span>
          <span class="diff-text">{line.lines[0] || ' '}</span>
        </div>
      {:else if line.type === 'modify'}
        <!-- Show old line with deletions highlighted -->
        <div class="diff-line modify-old">
          <span class="diff-gutter">-</span>
          <span class="diff-text">
            {#each line.wordDiff as part}
              {#if part.removed}
                <span class="diff-deleted">{part.value}</span>
              {:else if !part.added}
                <span class="diff-equal">{part.value}</span>
              {/if}
            {/each}
          </span>
        </div>
        <!-- Show new line with additions highlighted -->
        <div class="diff-line modify-new">
          <span class="diff-gutter">+</span>
          <span class="diff-text">
            {#each line.wordDiff as part}
              {#if part.added}
                <span class="diff-inserted">{part.value}</span>
              {:else if !part.removed}
                <span class="diff-equal">{part.value}</span>
              {/if}
            {/each}
          </span>
        </div>
      {/if}
    {/each}
  </div>
</div>

<style>
  .diff-view {
    border: 1px solid #30363d;
    border-radius: 6px;
    overflow: hidden;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 12px;
  }

  .diff-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 10px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
  }

  .diff-title {
    font-size: 11px;
    font-weight: 600;
    color: #d29922;
  }

  .diff-stats {
    font-size: 11px;
  }
  .additions { color: #3fb950; margin-right: 8px; }
  .deletions { color: #f85149; }

  .diff-content {
    overflow-x: auto;
  }

  .diff-line {
    display: flex;
    line-height: 1.6;
    min-height: 20px;
  }

  .diff-line.add {
    background: rgba(46, 160, 67, 0.15);
  }
  .diff-line.remove {
    background: rgba(248, 81, 73, 0.15);
  }
  .diff-line.modify-old {
    background: rgba(248, 81, 73, 0.15);
  }
  .diff-line.modify-new {
    background: rgba(46, 160, 67, 0.15);
  }
  .diff-line.context {
    background: transparent;
  }

  .diff-gutter {
    flex-shrink: 0;
    width: 24px;
    text-align: center;
    user-select: none;
    color: #484f58;
  }
  .diff-line.add .diff-gutter { color: #3fb950; }
  .diff-line.remove .diff-gutter { color: #f85149; }
  .diff-line.modify-old .diff-gutter { color: #f85149; }
  .diff-line.modify-new .diff-gutter { color: #3fb950; }

  .diff-text {
    flex: 1;
    padding-left: 4px;
    white-space: pre;
    color: #c9d1d9;
  }
  .diff-line.add .diff-text { color: #aff5b4; }
  .diff-line.remove .diff-text { color: #ffa198; }
  .diff-line.modify-old .diff-text { color: #ffa198; }
  .diff-line.modify-new .diff-text { color: #aff5b4; }
  
  /* Inline diff highlighting */
  .diff-equal {
    color: inherit;
  }
  
  .diff-deleted {
    background: rgba(248, 81, 73, 0.4);
    color: #ffa198;
    font-weight: 600;
  }
  
  .diff-inserted {
    background: rgba(46, 160, 67, 0.4);
    color: #aff5b4;
    font-weight: 600;
  }
</style>
