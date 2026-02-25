<script>
  /**
   * Compute a simple line-by-line diff between two strings.
   * Returns array of { type: 'add'|'remove'|'context', line: string }
   */
  export function computeDiff(oldText, newText) {
    const oldLines = (oldText || '').split('\n');
    const newLines = (newText || '').split('\n');

    // Simple LCS-based diff
    const m = oldLines.length;
    const n = newLines.length;

    // Build LCS table
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (oldLines[i - 1] === newLines[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    // Backtrack to get diff
    const result = [];
    let i = m, j = n;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && oldLines[i - 1] === newLines[j - 1]) {
        result.unshift({ type: 'context', line: oldLines[i - 1] });
        i--; j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        result.unshift({ type: 'add', line: newLines[j - 1] });
        j--;
      } else {
        result.unshift({ type: 'remove', line: oldLines[i - 1] });
        i--;
      }
    }

    return result;
  }

  /** @type {{ oldCode: string, newCode: string }} */
  let { oldCode, newCode } = $props();

  let diffLines = $derived(computeDiff(oldCode, newCode));
</script>

<div class="diff-view">
  <div class="diff-header">
    <span class="diff-title">Cell Edit Diff</span>
    <span class="diff-stats">
      <span class="additions">+{diffLines.filter(l => l.type === 'add').length}</span>
      <span class="deletions">-{diffLines.filter(l => l.type === 'remove').length}</span>
    </span>
  </div>
  <div class="diff-content">
    {#each diffLines as line, i}
      <div class="diff-line" class:add={line.type === 'add'} class:remove={line.type === 'remove'} class:context={line.type === 'context'}>
        <span class="diff-gutter">{line.type === 'add' ? '+' : line.type === 'remove' ? '-' : ' '}</span>
        <span class="diff-text">{line.line || ' '}</span>
      </div>
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

  .diff-text {
    flex: 1;
    padding-left: 4px;
    white-space: pre;
    color: #c9d1d9;
  }
  .diff-line.add .diff-text { color: #aff5b4; }
  .diff-line.remove .diff-text { color: #ffa198; }
</style>
