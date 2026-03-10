<script>
  import { onMount } from 'svelte';
  import ChatPanel from './lib/ChatPanel.svelte';
  import NotebookPanel from './lib/NotebookPanel.svelte';

  let data = $state(null);
  let loading = $state(true);
  let error = $state(null);
  let availableConfigs = $state([]);
  let selectedConfig = $state('');

  // Selection state
  let selectedModel = $state('');
  let selectedLibrary = $state('');
  let selectedInstance = $state('');
  let selectedRun = $state('');
  let currentStepIndex = $state(0);
  let viewMode = $state('steps');

  // Derived: available options
  let models = $derived(
    data ? [...new Set(data.trajectories.map(t => t.model))].sort() : []
  );

  // Library with success counts (instance is successful if all runs are successful)
  let librariesWithStats = $derived.by(() => {
    if (!data || !selectedModel) return [];
    
    const filtered = data.trajectories.filter(t => t.model === selectedModel);
    const libraryMap = new Map();
    
    filtered.forEach(t => {
      if (!libraryMap.has(t.library)) {
        libraryMap.set(t.library, new Map());
      }
      const instances = libraryMap.get(t.library);
      if (!instances.has(t.instance)) {
        instances.set(t.instance, []);
      }
      instances.get(t.instance).push(t.metadata.success);
    });
    
    const result = [];
    libraryMap.forEach((instances, library) => {
      let successfulInstances = 0;
      let totalInstances = instances.size;
      
      instances.forEach((runs) => {
        // Instance is successful if all runs were successful
        if (runs.every(success => success)) {
          successfulInstances++;
        }
      });
      
      result.push({
        name: library,
        successCount: successfulInstances,
        totalCount: totalInstances
      });
    });
    
    return result.sort((a, b) => a.name.localeCompare(b.name));
  });

  let libraries = $derived(librariesWithStats.map(l => l.name));

  // Instances with success counts (per run)
  let instancesWithStats = $derived.by(() => {
    if (!data || !selectedModel || !selectedLibrary) return [];
    
    const filtered = data.trajectories.filter(t => 
      t.model === selectedModel && t.library === selectedLibrary
    );
    
    const instanceMap = new Map();
    filtered.forEach(t => {
      if (!instanceMap.has(t.instance)) {
        instanceMap.set(t.instance, []);
      }
      instanceMap.get(t.instance).push(t.metadata.success);
    });
    
    const result = [];
    instanceMap.forEach((runs, instance) => {
      const successCount = runs.filter(s => s).length;
      result.push({
        name: instance,
        successCount: successCount,
        totalCount: runs.length
      });
    });
    
    return result.sort((a, b) => {
      const numA = parseInt(a.name.replace(/\D/g, ''));
      const numB = parseInt(b.name.replace(/\D/g, ''));
      return numA - numB;
    });
  });

  let instances = $derived(instancesWithStats.map(i => i.name));

  let runs = $derived(
    data && selectedModel && selectedLibrary && selectedInstance
      ? [...new Set(data.trajectories.filter(t => 
          t.model === selectedModel && 
          t.library === selectedLibrary && 
          t.instance === selectedInstance
        ).map(t => t.run))].sort()
      : []
  );

  let currentTrajectory = $derived(
    data
      ? data.trajectories.find(t => t.model === selectedModel && t.library === selectedLibrary && t.instance === selectedInstance && t.run === selectedRun)
      : null
  );

  let currentStep = $derived(
    currentTrajectory && currentTrajectory.steps[currentStepIndex]
      ? currentTrajectory.steps[currentStepIndex]
      : null
  );

  async function loadSelectedConfig() {
    if (!selectedConfig) return;

    loading = true;
    try {
      const dataPath = `/${selectedConfig}/data.json`;
      const resp = await fetch(dataPath);
      if (!resp.ok) {
        throw new Error(`Failed to load ${dataPath}`);
      }

      const d = await resp.json();
      data = d;
      error = null;

      if (d.trajectories && d.trajectories.length > 0) {
        const first = d.trajectories[0];
        selectedModel = first.model;
        selectedLibrary = first.library;
        selectedInstance = first.instance;
        selectedRun = first.run;
        currentStepIndex = 0;
      }
    } catch (e) {
      error = e.message;
      data = null;
    } finally {
      loading = false;
    }
  }

  // Load configs and data
  onMount(async () => {
    try {
      // Load available configs
      const configResp = await fetch('/configs_index.json');
      if (configResp.ok) {
        const configData = await configResp.json();
        availableConfigs = configData.configs || [];
        
        // Select first config by default
        if (availableConfigs.length > 0) {
          selectedConfig = availableConfigs[0].name;
          await loadSelectedConfig();
        }
      }
    } catch (e) {
      console.warn('Could not load configs index:', e.message);
    }
    
    loading = false;
  });

  // Load data when config changes
  let lastLoadedConfig = $state('');
  $effect(() => {
    if (!selectedConfig || selectedConfig === lastLoadedConfig) return;
    lastLoadedConfig = selectedConfig;
    loadSelectedConfig();
  });

  // Reset downstream selections when upstream changes
  $effect(() => {
    if (selectedModel && libraries.length && !libraries.includes(selectedLibrary)) {
      selectedLibrary = libraries[0];
    }
  });
  $effect(() => {
    if (selectedLibrary && instances.length && !instances.includes(selectedInstance)) {
      selectedInstance = instances[0];
    }
  });
  $effect(() => {
    if (selectedInstance && runs.length && !runs.includes(selectedRun)) {
      selectedRun = runs[0];
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
    if (!currentTrajectory || viewMode !== 'steps') return;
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
        <span>Config</span>
        <select bind:value={selectedConfig}>
          {#each availableConfigs as config}
            <option value={config.name}>
              {config.name} ({config.trajectory_count} trajectories)
            </option>
          {/each}
        </select>
      </label>
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
          {#each librariesWithStats as lib}
            <option value={lib.name}>
              {lib.name} ({lib.successCount}/{lib.totalCount} ✓)
            </option>
          {/each}
        </select>
      </label>
      <label>
        <span>Instance</span>
        <select bind:value={selectedInstance}>
          {#each instancesWithStats as inst}
            <option value={inst.name}>
              {inst.name} ({inst.successCount}/{inst.totalCount} ✓)
            </option>
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
        <span>View</span>
        <select bind:value={viewMode}>
          <option value="steps">Steps</option>
          <option value="verify">Verify</option>
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
        {#if currentTrajectory.metadata.success && currentTrajectory.metadata.submit_reason}
          <span class="meta-item submit-reason" title={currentTrajectory.metadata.submit_reason}>
            📋 {currentTrajectory.metadata.submit_reason.length > 50 ? currentTrajectory.metadata.submit_reason.substring(0, 47) + '...' : currentTrajectory.metadata.submit_reason}
          </span>
        {/if}
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
      {#if viewMode === 'steps'}
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
            steps={currentTrajectory.steps}
            codeChanges={currentTrajectory.code_changes}
            referenceNotebook={currentTrajectory.reference_fix_notebook}
            {currentStepIndex}
            mode={viewMode}
          />
        </div>
      {:else}
        <div class="panel notebook-panel verify-full-width">
          <NotebookPanel
            cells={currentTrajectory.notebook_cells}
            step={currentStep}
            steps={currentTrajectory.steps}
            codeChanges={currentTrajectory.code_changes}
            referenceNotebook={currentTrajectory.reference_fix_notebook}
            {currentStepIndex}
            mode={viewMode}
          />
        </div>
      {/if}
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
    min-width: 120px;
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

  .meta-item.submit-reason {
    color: #58a6ff;
    font-style: italic;
    cursor: help;
    max-width: 400px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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

  .verify-full-width {
    width: 100%;
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
