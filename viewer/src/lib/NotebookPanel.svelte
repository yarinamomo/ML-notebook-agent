<script>
  import DiffView from './DiffView.svelte';

  /** @type {{
   *  cells: any[],
   *  step: any,
   *  steps: any[],
   *  codeChanges: any[],
   *  currentStepIndex: number,
   *  mode?: string,
   *  referenceNotebook?: string[] | null
   * }} */
  let {
    cells,
    step,
    steps,
    codeChanges,
    currentStepIndex,
    mode = 'steps',
    referenceNotebook = null,
  } = $props();

  let cellsContainer = $state(null);
  let cellStates = $state({});
  let focusedCellIndex = $state(null);

  const cellViewModes = [
    { key: 'original', label: 'Original', isDiff: false },
    { key: 'diff_original_to_final', label: 'Orig → Final', isDiff: true, compareFrom: 'original', compareTo: 'final' },
    { key: 'final_only', label: 'Final', isDiff: false },
    { key: 'diff_final_to_reference', label: 'Final → Ref', isDiff: true, compareFrom: 'final', compareTo: 'reference' },
    { key: 'reference_only', label: 'Reference', isDiff: false },
    { key: 'diff_original_to_reference', label: 'Orig → Ref', isDiff: true, compareFrom: 'original', compareTo: 'reference' },
  ];

  function getCellState(cellIndex) {
    return cellStates[cellIndex] || 'original';
  }

  function setCellState(cellIndex, state) {
    cellStates = { ...cellStates, [cellIndex]: state };
  }

  function parseSerializedCell(cellData) {
    if (typeof cellData !== 'string') return { index: null, code: null };

    const lines = cellData.split('\n');
    if (!lines.length || !lines[0].startsWith('# --- [CELL ')) {
      return { index: null, code: null };
    }

    const match = lines[0].match(/\[CELL\s+(\d+)\]/);
    if (!match) return { index: null, code: null };

    const index = Number(match[1]);
    const codeLines = lines.slice(1);
    while (codeLines.length && !codeLines[0].trim()) {
      codeLines.shift();
    }

    return { index, code: codeLines.join('\n') };
  }

  let originalMap = $derived.by(() => {
    const map = new Map();
    if (!cells) return map;
    cells.forEach((cell) => {
      map.set(cell.index, cell.original_code || '');
    });
    return map;
  });

  let finalMap = $derived.by(() => {
    const map = new Map(originalMap);
    if (!codeChanges) return map;
    codeChanges.forEach((change) => {
      map.set(change.cell_index, change.code || '');
    });
    return map;
  });

  let referenceMap = $derived.by(() => {
    const map = new Map();
    if (!referenceNotebook) return map;

    referenceNotebook.forEach((rawCell) => {
      const parsed = parseSerializedCell(rawCell);
      if (parsed.index !== null) {
        map.set(parsed.index, parsed.code || '');
      }
    });

    return map;
  });

  let sortedCellIndices = $derived.by(() => {
    const indices = new Set();
    originalMap.forEach((_, idx) => indices.add(idx));
    finalMap.forEach((_, idx) => indices.add(idx));
    referenceMap.forEach((_, idx) => indices.add(idx));
    return [...indices].sort((a, b) => a - b);
  });

  // Get available view modes for a specific cell
  function getAvailableModesForCell(cellIndex) {
    const original = originalMap.get(cellIndex) || '';
    const final = finalMap.get(cellIndex) || '';
    const reference = referenceMap.get(cellIndex) || '';

    const hasAgentChanges = original !== final;
    const hasReferenceChanges = original !== reference;
    const hasBothChanges = hasAgentChanges && hasReferenceChanges;

    return cellViewModes.filter(mode => {
      if (!mode.isDiff) {
        // Show original if there are any changes
        if (mode.key === 'original') return hasAgentChanges || hasReferenceChanges;
        // Show final only if agent made changes
        if (mode.key === 'final_only') return hasAgentChanges;
        // Show reference only if reference differs from original
        if (mode.key === 'reference_only') return hasReferenceChanges;
      } else {
        // Show diff tabs based on what changed
        if (mode.compareFrom === 'original' && mode.compareTo === 'final') {
          // Show Original → Final only if agent made changes
          return hasAgentChanges;
        }
        if (mode.compareFrom === 'final' && mode.compareTo === 'reference') {
          // Show Final → Reference only if BOTH agent and reference made changes
          return hasBothChanges;
        }
        if (mode.compareFrom === 'original' && mode.compareTo === 'reference') {
          // Show Original → Reference only if reference differs
          return hasReferenceChanges;
        }
      }
      return true;
    });
  }

  // Check if cell has any changes
  function cellHasChanges(cellIndex) {
    const original = originalMap.get(cellIndex) || '';
    const final = finalMap.get(cellIndex) || '';
    const reference = referenceMap.get(cellIndex) || '';
    return original !== final || original !== reference;
  }

  // Get change types for badge display
  function getCellChangeTypes(cellIndex) {
    const original = originalMap.get(cellIndex) || '';
    const final = finalMap.get(cellIndex) || '';
    const reference = referenceMap.get(cellIndex) || '';
    
    return {
      hasAgentEdit: original !== final,
      hasReferenceEdit: original !== reference,
    };
  }

  // Get the most recent cell statuses at or before the current step
  let currentCellStatuses = $derived.by(() => {
    if (mode !== 'steps' || !steps || !step) return null;
    
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
    if (mode !== 'steps') return [];
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
    mode === 'steps' && step && step.edit ? step.edit.cell_index : -1
  );

  let verifyCells = $derived.by(() => {
    if (mode !== 'verify') return [];
    return sortedCellIndices.map(idx => ({ index: idx }));
  });

  // Get cell content based on its current state
  function getCellContent(cellIndex) {
    const state = getCellState(cellIndex);
    const original = originalMap.get(cellIndex) || '';
    const final = finalMap.get(cellIndex) || '';
    const reference = referenceMap.get(cellIndex) || '';

    if (state === 'original') {
      return { code: original, state: 'original', badgeType: null };
    }
    if (state === 'diff_original_to_final') {
      return {
        compareOld: original,
        compareNew: final,
        state: 'diff',
        badgeType: original !== final ? 'agent' : null,
      };
    }
    if (state === 'final_only') {
      return {
        code: final,
        state: 'final',
        badgeType: original !== final ? 'agent' : null,
      };
    }
    if (state === 'diff_final_to_reference') {
      return {
        compareOld: final,
        compareNew: reference,
        state: 'diff-ref',
        badgeType: final !== reference ? 'reference' : null,
      };
    }
    if (state === 'reference_only') {
      return {
        code: reference,
        state: 'reference',
        badgeType: original !== reference ? 'reference' : null,
      };
    }
    if (state === 'diff_original_to_reference') {
      return {
        compareOld: original,
        compareNew: reference,
        state: 'diff-ref',
        badgeType: original !== reference ? 'reference' : null,
      };
    }

    return { code: original, state: 'original', badgeType: null };
  }

  let displayedCells = $derived(mode === 'steps' ? cellsWithCurrentState : verifyCells);

  let notebookTitle = $derived(mode === 'steps' ? 'Notebook Cells' : 'Verify Notebook');

  let emptyVerifyMessage = $derived.by(() => {
    if (mode !== 'verify') return '';
    if (
      (activeVerifyTab === 'diff_final_to_reference' ||
        activeVerifyTab === 'diff_original_to_reference' ||
        activeVerifyTab === 'reference_only') &&
      (!referenceNotebook || referenceNotebook.length === 0)
    ) {
      return 'No reference fixed notebook found for this instance.';
    }
    return 'No cells to display for this verify state.';
  });

  function handleKeyNavigation(event) {
    if (mode !== 'verify') return;
    
    // Only handle keys if not in an input/textarea
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
    
    // Get cells with changes
    const cellsWithChanges = sortedCellIndices.filter(idx => cellHasChanges(idx));
    
    if (cellsWithChanges.length === 0) return;
    
    // Tab key: move focus between cells with changes
    if (event.key === 'Tab') {
      event.preventDefault();
      
      const currentIndex = cellsWithChanges.indexOf(focusedCellIndex);
      let nextIndex;
      
      if (event.shiftKey) {
        // Shift+Tab: previous cell
        nextIndex = currentIndex <= 0 ? cellsWithChanges.length - 1 : currentIndex - 1;
      } else {
        // Tab: next cell
        nextIndex = currentIndex >= cellsWithChanges.length - 1 ? 0 : currentIndex + 1;
      }
      
      focusedCellIndex = cellsWithChanges[nextIndex];
      
      // Scroll to focused cell
      const el = cellsContainer?.querySelector(`[data-cell-index="${focusedCellIndex}"]`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
      return;
    }
    
    // Arrow keys: change view mode for focused cell
    if (focusedCellIndex === null) {
      focusedCellIndex = cellsWithChanges[0] || null;
    }
    
    if (focusedCellIndex !== null && cellHasChanges(focusedCellIndex)) {
      const availableModes = getAvailableModesForCell(focusedCellIndex);
      const currentState = getCellState(focusedCellIndex);
      const currentModeIndex = availableModes.findIndex(m => m.key === currentState);
      
      if (event.key === 'ArrowLeft' && currentModeIndex > 0) {
        event.preventDefault();
        setCellState(focusedCellIndex, availableModes[currentModeIndex - 1].key);
      } else if (event.key === 'ArrowRight' && currentModeIndex < availableModes.length - 1) {
        event.preventDefault();
        setCellState(focusedCellIndex, availableModes[currentModeIndex + 1].key);
      }
    }
  }

  function focusCell(cellIndex) {
    if (cellHasChanges(cellIndex)) {
      focusedCellIndex = cellIndex;
    }
  }

  // Keyboard navigation for verify mode
  $effect(() => {
    if (mode !== 'verify') return;
    
    window.addEventListener('keydown', handleKeyNavigation);
    return () => {
      window.removeEventListener('keydown', handleKeyNavigation);
    };
  });

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
  <span class="notebook-title">{notebookTitle}</span>
  <span class="cell-count">{displayedCells.length} cells</span>
  {#if mode === 'steps' && editedCellIndex >= 0}
    <span class="edit-indicator">✏️ Editing Cell {editedCellIndex}</span>
  {/if}
  {#if mode === 'verify'}
    <span class="keyboard-hint">Tab: navigate edited cells | ←→: change view</span>
  {/if}
</div>

<div class="cells-container" bind:this={cellsContainer}>
  {#if displayedCells.length === 0}
    <div class="empty-state">{emptyVerifyMessage}</div>
  {:else}
  {#each displayedCells as cell}
    {@const cellContent = mode === 'verify' ? getCellContent(cell.index) : null}
    {@const availableModes = mode === 'verify' ? getAvailableModesForCell(cell.index) : []}
    {@const currentMode = mode === 'verify' ? getCellState(cell.index) : null}
    {@const hasChanges = mode === 'verify' ? cellHasChanges(cell.index) : false}
    {@const changeTypes = mode === 'verify' ? getCellChangeTypes(cell.index) : null}
    <div
      class="cell-card"
      class:edited={mode === 'steps' && (cell.state === 'edited' || cell.state === 'final' || cell.state === 'reference')}
      class:active-edit={mode === 'steps' && cell.index === editedCellIndex}
      class:focused-cell={mode === 'verify' && hasChanges && cell.index === focusedCellIndex}
      data-cell-index={cell.index}
      role={mode === 'verify' && hasChanges ? 'button' : undefined}
      tabindex={mode === 'verify' && hasChanges ? '0' : undefined}
      onclick={() => mode === 'verify' && hasChanges && focusCell(cell.index)}
      onkeydown={(e) => mode === 'verify' && hasChanges && (e.key === 'Enter' || e.key === ' ') && focusCell(cell.index)}
    >
      <div class="cell-header">
        <span class="cell-index">Cell {cell.index}</span>
        <div class="cell-badges">
          {#if mode === 'steps' && cell.executionStatus}
            <span class="cell-badge status-badge status-{cell.executionStatus}">
              {cell.executionStatus}
            </span>
          {/if}
          {#if mode === 'steps'}
            {#if cell.state === 'edited'}
              <span class="cell-badge edit-badge">edited</span>
            {:else if cell.state === 'final'}
              <span class="cell-badge edit-badge">final agent</span>
            {:else if cell.state === 'reference'}
              <span class="cell-badge edit-badge">reference fix</span>
            {/if}
          {/if}
          {#if mode === 'verify' && changeTypes}
            {#if changeTypes.hasAgentEdit}
              <span class="cell-badge edit-badge">edited</span>
            {/if}
            {#if changeTypes.hasReferenceEdit}
              <span class="cell-badge reference-badge">reference</span>
            {/if}
          {/if}
        </div>
      </div>

      {#if mode === 'verify' && hasChanges}
        <!-- Per-cell tabs (only if there are changes) -->
        <div class="cell-view-tabs">
          {#each availableModes as viewMode}
            <button
              type="button"
              class="cell-view-tab"
              class:active={currentMode === viewMode.key}
              onclick={(e) => {
                e.stopPropagation();
                setCellState(cell.index, viewMode.key);
              }}
            >
              {viewMode.label}
            </button>
          {/each}
        </div>
        
        <!-- Cell content based on current mode -->
        {#if cellContent?.compareOld !== undefined && cellContent?.compareNew !== undefined}
          <DiffView oldCode={cellContent.compareOld} newCode={cellContent.compareNew} />
        {:else if cellContent?.code !== undefined}
          <pre class="cell-code">{cellContent.code}</pre>
        {/if}
      {:else if mode === 'verify'}
        <!-- No changes, just show original code -->
        <pre class="cell-code">{originalMap.get(cell.index) || ''}</pre>
      {:else}
        <!-- Steps mode content -->
        {#if mode === 'steps' && cell.index === editedCellIndex && step.edit}
          <!-- Show diff for this step's edit -->
          <DiffView oldCode={step.edit.old_code} newCode={step.edit.new_code} />
        {:else}
          <!-- Show current code -->
          <pre class="cell-code">{cell.code}</pre>
        {/if}
      {/if}
    </div>
  {/each}
  {/if}
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

  .keyboard-hint {
    font-size: 11px;
    color: #6e7681;
    margin-left: auto;
    font-family: monospace;
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

  .cell-card.focused-cell {
    border-color: #58a6ff;
    box-shadow: 0 0 0 1px rgba(88, 166, 255, 0.4), 0 0 12px rgba(88, 166, 255, 0.15);
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

  .cell-view-tabs {
    display: flex;
    gap: 4px;
    padding: 6px 10px;
    background: #0d1117;
    border-bottom: 1px solid #21262d;
    overflow-x: auto;
    flex-wrap: wrap;
  }

  .cell-view-tab {
    border: 1px solid #30363d;
    background: #161b22;
    color: #8b949e;
    border-radius: 4px;
    font-size: 10px;
    padding: 3px 8px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s ease;
  }

  .cell-view-tab:hover {
    border-color: #58a6ff;
    color: #c9d1d9;
  }

  .cell-view-tab.active {
    background: #1f6feb;
    border-color: #1f6feb;
    color: #ffffff;
    font-weight: 600;
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

  .reference-badge {
    background: #1f3a1f;
    color: #3fb950;
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

  .empty-state {
    padding: 16px;
    color: #8b949e;
    font-size: 12px;
    border: 1px dashed #30363d;
    border-radius: 6px;
    background: #0d1117;
  }
</style>
