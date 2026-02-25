<script>
  /** @type {{ steps: any[], currentStepIndex: number, onSelectStep: (i: number) => void }} */
  let { steps, currentStepIndex, onSelectStep } = $props();

  let stepsContainer = $state(null);

  // Auto-scroll to current step
  $effect(() => {
    if (stepsContainer && currentStepIndex >= 0) {
      const el = stepsContainer.querySelector(`[data-step-index="${currentStepIndex}"]`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  });

  function getActionIcon(actionType) {
    switch (actionType) {
      case 'run_all': return '▶▶';
      case 'run_cell': return '▶';
      case 'edit_cell': return '✏️';
      case 'get_cell': return '🔍';
      case 'get_cells': return '📋';
      case 'run_code': return '⚡';
      case 'submit': return '✅';
      default: return '•';
    }
  }

  function getActionLabel(actionType) {
    switch (actionType) {
      case 'run_all': return 'Run All';
      case 'run_cell': return 'Run Cell';
      case 'edit_cell': return 'Edit Cell';
      case 'get_cell': return 'Get Cell';
      case 'get_cells': return 'Get Cells';
      case 'get_cell_count': return 'Cell Count';
      case 'run_code': return 'Run Code';
      case 'submit': return 'Submit';
      default: return 'Action';
    }
  }

  function truncateObservation(obs) {
    if (!obs) return '';
    // Strip the XML-like wrapper
    let clean = obs.replace(/<returncode>\d+<\/returncode>\n?/g, '');
    clean = clean.replace(/<output>\n?/g, '').replace(/<\/output>/g, '');
    clean = clean.replace(/<warning>.*?<\/warning>/gs, '[truncated]');
    clean = clean.replace(/<output_head>\n?/g, '').replace(/<\/output_head>/g, '');
    clean = clean.replace(/<output_tail>\n?/g, '').replace(/<\/output_tail>/g, '');
    clean = clean.replace(/<elided_chars>.*?<\/elided_chars>/gs, '\n... [output truncated] ...\n');
    return clean.trim();
  }

  function extractExplanation(content) {
    if (!content) return '';
    // Remove the code block (action) part
    return content.replace(/```bash\n[\s\S]*?\n```/g, '').trim();
  }

  function hasError(obs) {
    return obs && (obs.includes('---ERROR---') || obs.includes('Status: error'));
  }
</script>

<div class="chat-header">
  <span class="chat-title">Agent Steps</span>
  <div class="step-nav">
    <button onclick={() => onSelectStep(Math.max(0, currentStepIndex - 1))} disabled={currentStepIndex === 0}>
      ← Prev
    </button>
    <span class="step-counter">{currentStepIndex + 1} / {steps.length}</span>
    <button onclick={() => onSelectStep(Math.min(steps.length - 1, currentStepIndex + 1))} disabled={currentStepIndex === steps.length - 1}>
      Next →
    </button>
  </div>
</div>

<div class="steps-container" bind:this={stepsContainer}>
  {#each steps as step, i}
    <div
      class="step-card"
      class:active={i === currentStepIndex}
      class:has-edit={step.action_type === 'edit_cell'}
      class:has-error={hasError(step.observation)}
      data-step-index={i}
      onclick={() => onSelectStep(i)}
      role="button"
      tabindex="0"
      onkeydown={(e) => e.key === 'Enter' && onSelectStep(i)}
    >
      <div class="step-header">
        <span class="step-number">Step {step.step}</span>
        <span class="action-badge" class:edit={step.action_type === 'edit_cell'} class:run={step.action_type === 'run_all' || step.action_type === 'run_cell'} class:inspect={step.action_type === 'get_cell' || step.action_type === 'get_cells' || step.action_type === 'run_code'} class:done={step.action_type === 'submit'}>
          {getActionIcon(step.action_type)} {getActionLabel(step.action_type)}
        </span>
      </div>

      {#if step.reasoning}
        <details class="reasoning">
          <summary>💭 Thinking</summary>
          <pre class="reasoning-text">{step.reasoning}</pre>
        </details>
      {/if}

      <div class="step-explanation">
        {extractExplanation(step.assistant)}
      </div>

      {#if step.action && step.action_type !== 'submit'}
        <div class="step-action">
          <code>{step.action}</code>
        </div>
      {/if}

      {#if step.observation}
        <details class="observation" open={i === currentStepIndex}>
          <summary class:obs-error={hasError(step.observation)}>
            Output {hasError(step.observation) ? '❌' : ''}
          </summary>
          <pre class="observation-text" class:error-text={hasError(step.observation)}>{truncateObservation(step.observation)}</pre>
        </details>
      {/if}
    </div>
  {/each}
</div>

<style>
  .chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    flex-shrink: 0;
  }

  .chat-title {
    font-weight: 600;
    font-size: 13px;
    color: #58a6ff;
  }

  .step-nav {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .step-nav button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 12px;
    cursor: pointer;
  }
  .step-nav button:hover:not(:disabled) {
    background: #30363d;
    border-color: #58a6ff;
  }
  .step-nav button:disabled {
    opacity: 0.4;
    cursor: default;
  }

  .step-counter {
    font-size: 12px;
    color: #8b949e;
    min-width: 50px;
    text-align: center;
  }

  .steps-container {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }

  .step-card {
    margin-bottom: 6px;
    padding: 10px 12px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s ease;
    border-left: 3px solid transparent;
  }
  .step-card:hover {
    background: #1c2128;
    border-color: #30363d;
  }
  .step-card.active {
    background: #1c2128;
    border-color: #58a6ff;
    border-left-color: #58a6ff;
    box-shadow: 0 0 0 1px rgba(88, 166, 255, 0.2);
  }
  .step-card.has-edit {
    border-left-color: #d29922;
  }
  .step-card.active.has-edit {
    border-left-color: #d29922;
    border-color: #d29922;
    box-shadow: 0 0 0 1px rgba(210, 153, 34, 0.2);
  }

  .step-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
  }

  .step-number {
    font-weight: 600;
    font-size: 12px;
    color: #8b949e;
  }

  .action-badge {
    font-size: 11px;
    padding: 1px 8px;
    border-radius: 10px;
    background: #21262d;
    color: #8b949e;
  }
  .action-badge.edit { background: #3d2e00; color: #d29922; }
  .action-badge.run { background: #0a2d1c; color: #3fb950; }
  .action-badge.inspect { background: #0c2d6b; color: #58a6ff; }
  .action-badge.done { background: #1b4332; color: #2dc653; }

  .reasoning {
    margin-bottom: 6px;
  }
  .reasoning summary {
    font-size: 11px;
    color: #8b949e;
    cursor: pointer;
    user-select: none;
  }
  .reasoning-text {
    font-size: 11px;
    color: #7d8590;
    white-space: pre-wrap;
    word-break: break-word;
    margin-top: 4px;
    padding: 6px 8px;
    background: #0d1117;
    border-radius: 4px;
    max-height: 150px;
    overflow-y: auto;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  .step-explanation {
    font-size: 13px;
    line-height: 1.5;
    color: #c9d1d9;
    margin-bottom: 6px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .step-action {
    margin-bottom: 6px;
  }
  .step-action code {
    display: block;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 11px;
    padding: 6px 8px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 4px;
    color: #79c0ff;
    word-break: break-all;
    white-space: pre-wrap;
  }

  .observation summary {
    font-size: 11px;
    color: #8b949e;
    cursor: pointer;
    user-select: none;
    margin-bottom: 4px;
  }
  .observation summary.obs-error {
    color: #f85149;
  }

  .observation-text {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 11px;
    padding: 6px 8px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 4px;
    color: #8b949e;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 300px;
    overflow-y: auto;
  }
  .observation-text.error-text {
    border-color: #f8514933;
    color: #f85149;
  }
</style>
