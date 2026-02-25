<script>
  import { onMount } from 'svelte';
  import ChatPanel from './lib/ChatPanel.svelte';
  import NotebookPanel from './lib/NotebookPanel.svelte';

  let data = $state(null);
  let loading = $state(true);
  let error = $state(null);

  // Selection state
  let selectedModel = $state('');
  let selectedLibrary = $state('');
  let selectedRun = $state('');
  let selectedInstance = $state('');
  let currentStepIndex = $state(0);

  // Derived: available options
  let models = $derived(
    data ? [...new Set(data.trajectories.map(t => t.model))].sort() : []
  );

  let libraries = $derived(
    data && selectedModel
      ? [...new Set(data.trajectories.filter(t => t.model === selectedModel).map(t => t.library))].sort()
      : []
  );

  let runs = $derived(
    data && selectedModel && selectedLibrary
      ? [...new Set(data.trajectories.filter(t => t.model === selectedModel && t.library === selectedLibrary).map(t => t.run))].sort()
      : []
  );

  let instances = $derived(
    data && selectedModel && selectedLibrary && selectedRun
      ? [...new Set(data.trajectories.filter(t => t.model === selectedModel && t.library === selectedLibrary && t.run === selectedRun).map(t => t.instance))].sort((a, b) => {
          const numA = parseInt(a.replace(/\D/g, ''));
          const numB = parseInt(b.replace(/\D/g, ''));
          return numA - numB;
        })
      : []
  );

  let currentTrajectory = $derived(
    data
      ? data.trajectories.find(t => t.model === selectedModel && t.library === selectedLibrary && t.run === selectedRun && t.instance === selectedInstance)
      : null
  );

  let currentStep = $derived(
    currentTrajectory && currentTrajectory.steps[currentStepIndex]
      ? currentTrajectory.steps[currentStepIndex]
      : null
  );

  // Load data
  onMount(async () => {
    try {
      const resp = await fetch('/data.json');
      const d = await resp.json();
      data = d;
      loading = false;
      // Auto-select first available
      if (d.trajectories.length > 0) {
        const first = d.trajectories[0];
        selectedModel = first.model;
        selectedLibrary = first.library;
        selectedRun = first.run;
        selectedInstance = first.instance;
        currentStepIndex = 0;
      }
    } catch (e) {
      error = e.message;
      loading = false;
    }
  });

  // Reset downstream selections when upstream changes
  $effect(() => {
    if (selectedModel && libraries.length && !libraries.includes(selectedLibrary)) {
      selectedLibrary = libraries[0];
    }
  });
  $effect(() => {
    if (selectedLibrary && runs.length && !runs.includes(selectedRun)) {
      selectedRun = runs[0];
    }
  });
  $effect(() => {
    if (selectedRun && instances.length && !instances.includes(selectedInstance)) {
      selectedInstance = instances[0];
      currentStepIndex = 0;
    }
  });

  // Reset step index when trajectory changes
  let prevTrajectoryKey = $state('');
  $effect(() => {
    const key = currentTrajectory?.key || '';
    if (key !== prevTrajectoryKey) {
      prevTrajectoryKey = key;
      currentStepIndex = 0;
    }
  });

  function onSelectStep(index) {
    currentStepIndex = index;
  }

  function handleKeydown(e) {
    if (!currentTrajectory) return;
    // Don't interfere with form controls
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
      e.preventDefault();
      if (currentStepIndex < currentTrajectory.steps.length - 1) {
        currentStepIndex++;
      }
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      e.preventDefault();
      if (currentStepIndex > 0) {
        currentStepIndex--;
      }
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="app">
  <!-- Top Bar -->
  <header class="topbar">
    <div class="topbar-left">
      <h1>ML Agent Trajectory Viewer</h1>
    </div>
    <div class="topbar-selectors">
      <label>
        <span>Model</span>
        <select bind:value={selectedModel}>
          {#each models as m}
            <option value={m}>{m}</option>
          {/each}
        </select>
      </label>
      <label>
        <span>Library</span>
        <select bind:value={selectedLibrary}>
          {#each libraries as l}
            <option value={l}>{l}</option>
          {/each}
        </select>
      </label>
      <label>
        <span>Run</span>
        <select bind:value={selectedRun}>
          {#each runs as r}
            <option value={r}>{r}</option>
          {/each}
        </select>
      </label>
      <label>
        <span>Instance</span>
        <select bind:value={selectedInstance}>
          {#each instances as inst}
            <option value={inst}>{inst}</option>
          {/each}
        </select>
      </label>
    </div>
    {#if currentTrajectory}
      <div class="topbar-meta">
        <span class="badge" class:success={currentTrajectory.metadata.success} class:failure={!currentTrajectory.metadata.success}>
          {currentTrajectory.metadata.status}
        </span>
        <span class="meta-item">Steps: {currentTrajectory.metadata.total_steps}</span>
        <span class="meta-item">Cost: ${currentTrajectory.metadata.cost?.toFixed(4)}</span>
        <span class="meta-item">Time: {currentTrajectory.metadata.execution_time}s</span>
        <span class="meta-item">Edits: {currentTrajectory.metadata.cells_edited}</span>
      </div>
    {/if}
  </header>

  <!-- Main Content -->
  {#if loading}
    <div class="center-message">Loading trajectory data...</div>
  {:else if error}
    <div class="center-message error">Error: {error}</div>
  {:else if !currentTrajectory}
    <div class="center-message">No trajectory selected</div>
  {:else}
    <div class="main-content">
      <div class="panel chat-panel">
        <ChatPanel
          steps={currentTrajectory.steps}
          {currentStepIndex}
          {onSelectStep}
        />
      </div>
      <div class="panel notebook-panel">
        <NotebookPanel
          cells={currentTrajectory.notebook_cells}
          step={currentStep}
        />
      </div>
    </div>
  {/if}
</div>

<style>
  :global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    overflow: hidden;
    height: 100vh;
  }
  :global(#app) {
    height: 100vh;
  }

  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  .topbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 8px 16px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    flex-shrink: 0;
    flex-wrap: wrap;
  }

  .topbar-left h1 {
    font-size: 14px;
    font-weight: 600;
    color: #58a6ff;
    white-space: nowrap;
  }

  .topbar-selectors {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .topbar-selectors label {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: #8b949e;
  }

  .topbar-selectors select {
    background: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
    cursor: pointer;
  }
  .topbar-selectors select:hover {
    border-color: #58a6ff;
  }

  .topbar-meta {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-left: auto;
    font-size: 12px;
  }

  .badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }
  .badge.success { background: #1b4332; color: #2dc653; }
  .badge.failure { background: #4a1c1c; color: #f85149; }

  .meta-item {
    color: #8b949e;
  }

  .main-content {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .panel {
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .chat-panel {
    width: 45%;
    border-right: 1px solid #30363d;
  }

  .notebook-panel {
    width: 55%;
  }

  .center-message {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
    font-size: 16px;
    color: #8b949e;
  }
  .center-message.error {
    color: #f85149;
  }
</style>
