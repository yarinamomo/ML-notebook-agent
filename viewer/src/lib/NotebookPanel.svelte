<script>
  import DiffView from './DiffView.svelte';

  /** @type {{ cells: any[], step: any, steps: any[], codeChanges: any[], currentStepIndex: number }} */
  let { cells, step, steps, codeChanges, currentStepIndex } = $props();

  let cellsContainer = $state(null);

  // Get the most recent cell statuses at or before the current step
  let currentCellStatuses = $derived.by(() => {
    if (!steps || !step) return null;
    
    // Find the most recent step with cell_statuses up to current step
    const currentStep = step.step;
    for (let i = currentStep; i >= 1; i--) {
      const s = steps.find(s => s.step === i);
      if (s && s.cell_statuses) {
        return s.cell_statuses;
      }
    }
    return null;
  });

  // Compute the current state of cells based on the current step
  let cellsWithCurrentState = $derived.by(() => {
    if (!cells || !codeChanges) return cells;
    
    // Build a map of cell_index -> current code
    const cellStateMap = new Map();
    
    // Initialize with original code
    cells.forEach(cell => {
      cellStateMap.set(cell.index, cell.original_code);
    });
    
    // Apply changes up to and including the current step
    const currentStep = step ? step.step : 0;
    codeChanges.forEach(change => {
      if (change.step <= currentStep) {
        cellStateMap.set(change.cell_index, change.code);
      }
    });
    
    // Track which cells have been edited up to this point
    const editedCells = new Set(
      codeChanges.filter(c => c.step <= currentStep).map(c => c.cell_index)
    );
    
    // Return cells with computed state
    return cells.map(cell => {
      const status = currentCellStatuses ? currentCellStatuses[cell.index.toString()] : null;
      return {
        ...cell,
        code: cellStateMap.get(cell.index) || cell.original_code,
        state: editedCells.has(cell.index) ? 'edited' : 'unchanged',
        executionStatus: status,
      };
    });
  });

  // The cell index being edited in the current step
  let editedCellIndex = $derived(
    step && step.edit ? step.edit.cell_index : -1
  );

  // Auto-scroll to edited cell
  $effect(() => {
    if (cellsContainer && editedCellIndex >= 0) {
      const el = cellsContainer.querySelector(`[data-cell-index="${editedCellIndex}"]`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  });
</script>

<div class="notebook-header">
  <span class="notebook-title">Notebook Cells</span>
  <span class="cell-count">{cells.length} cells</span>
  {#if editedCellIndex >= 0}
    <span class="edit-indicator">✏️ Editing Cell {editedCellIndex}</span>
  {/if}
</div>

<div class="cells-container" bind:this={cellsContainer}>
  {#each cellsWithCurrentState as cell}
    <div
      class="cell-card"
      class:edited={cell.state === 'edited'}
      class:active-edit={cell.index === editedCellIndex}
      data-cell-index={cell.index}
    >
      <div class="cell-header">
        <span class="cell-index">Cell {cell.index}</span>
        <div class="cell-badges">
          {#if cell.executionStatus}
            <span class="cell-badge status-badge status-{cell.executionStatus}">
              {cell.executionStatus}
            </span>
          {/if}
          {#if cell.state === 'edited'}
            <span class="cell-badge edit-badge">edited</span>
          {/if}
        </div>
      </div>

      {#if cell.index === editedCellIndex && step.edit}
        <!-- Show diff for this step's edit -->
        <DiffView oldCode={step.edit.old_code} newCode={step.edit.new_code} />
      {:else}
        <!-- Show current code -->
        <pre class="cell-code">{cell.code}</pre>
      {/if}
    </div>
  {/each}
</div>

<style>
  .notebook-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    flex-shrink: 0;
  }

  .notebook-title {
    font-weight: 600;
    font-size: 13px;
    color: #58a6ff;
  }

  .cell-count {
    font-size: 12px;
    color: #8b949e;
  }

  .edit-indicator {
    font-size: 12px;
    color: #d29922;
    margin-left: auto;
  }

  .cells-container {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }

  .cell-card {
    margin-bottom: 6px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    overflow: hidden;
    transition: all 0.15s ease;
  }

  .cell-card.edited {
    border-color: #30363d;
  }

  .cell-card.active-edit {
    border-color: #d29922;
    box-shadow: 0 0 0 1px rgba(210, 153, 34, 0.3), 0 0 12px rgba(210, 153, 34, 0.1);
  }

  .cell-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 10px;
    background: #0d1117;
    border-bottom: 1px solid #21262d;
  }

  .cell-index {
    font-size: 11px;
    font-weight: 600;
    color: #8b949e;
  }

  .cell-badges {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .cell-badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
  }
  
  .edit-badge {
    background: #3d2e00;
    color: #d29922;
  }

  .status-badge {
    font-weight: 600;
    text-transform: lowercase;
  }

  .status-ok {
    background: #0f2e0f;
    color: #3fb950;
  }

  .status-error {
    background: #3d1f1f;
    color: #f85149;
  }

  .status-timeout {
    background: #3d2e00;
    color: #d29922;
  }

  .cell-code {
    padding: 8px 10px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 12px;
    line-height: 1.5;
    color: #c9d1d9;
    white-space: pre;
    overflow-x: auto;
    margin: 0;
    max-height: 400px;
    overflow-y: auto;
  }
</style>
