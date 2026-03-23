Recommended Prompt/Pipeline Tweaks

  1. Add a "Model Preparation Phase" (Phase 0)

  Add a new section before Phase 1 in MAIN_PROMPT.md:

  ## Phase 0: Model Preparation (for multi-file / external-dependency models)

  If the model is NOT a single self-contained .py file, first create one:

  ### Step 0.1: Identify the Computation Graph Entry Point
  - For HuggingFace models: find the `modeling_*.py` file and the main `forward()` method
  - For multi-file repos: trace imports to find all files needed for the forward pass
  - Ignore: tokenizers, processors, training utilities, data loaders

  ### Step 0.2: Create a Self-Contained Wrapper
  Create `{output_dir}/level_3_model/{model_name}.py` that:

  1. **Inlines or imports all needed code** — Copy essential classes/functions
     from dependency files directly into one file (or a local package folder)
  2. **Hardcodes the config** — Replace config-file loading with concrete values:
     ```python
     # Instead of: config = AutoConfig.from_pretrained("smolvla")
     # Do:
     HIDDEN_DIM = 768
     NUM_LAYERS = 12
     NUM_HEADS = 12
     # ... etc
  3. Wraps the model in the standard interface:
  class Model(nn.Module):
      def __init__(self):
          super().__init__()
          self.model = OriginalModelClass(hardcoded_config)

      def forward(self, *args):
          return self.model(*args)

  def get_inputs():
      # Return concrete tensors matching the model's expected inputs
      return [torch.randint(0, 32000, (1, 128)),  # input_ids
              torch.ones(1, 128, dtype=torch.long)]  # attention_mask

  def get_init_inputs():
      return []
  4. Loads pretrained weights (optional):
  def get_init_inputs():
      return []  # weights loaded in __init__ from checkpoint
  5. Verify it runs standalone: python {output_dir}/level_3_model/{model_name}.py → prints PASS

  Step 0.3: Handle External Dependencies

  - Pure PyTorch ops (einops.rearrange, etc.): Inline the equivalent torch code
  - Heavy frameworks (transformers, timm): Either:
    - (a) Copy only the needed source files locally and adjust imports, OR
    - (b) Accept the pip dependency and document it in a requirements.txt
  - Custom CUDA kernels: Replace with PyTorch-equivalent ops (mark as approximate)

  ### Step 0.4: Verify the Wrapper (GATE)

  This is the Phase 0 verification gate. You MUST NOT proceed to Phase 1 until this passes.

  #### Stage A: Standalone Execution
  Run the wrapper file. It must:
  - Execute without errors
  - Produce deterministic output shapes
  - NOT require network access or filesystem access at runtime

  #### Stage B: Numerical Equivalence Against Original

  Create `{output_dir}/steps/step_0_preparation/verify_wrapper.py` that:

  1. **Loads the original model using its native interface** (HF transformers,
     multi-file imports, config objects — whatever it needs)
  2. **Loads the self-contained wrapper** using the standard `Model`/`get_inputs()` interface
  3. **Transfers weights** from the original to the wrapper
     - For HuggingFace: `original = AutoModel.from_pretrained(...)`, then map
       state_dict keys to the wrapper's state_dict
     - For multi-file repos: load the original modules, map weights
  4. **Runs both on identical inputs** (3 trials, same seeds)
  5. **Compares outputs** with the same tolerances as verify_step.py

  ```python
  """
  Phase 0 Verification: Wrapper vs Original Model
  Confirms the self-contained wrapper reproduces the original model's output.
  """
  import torch
  import sys
  from pathlib import Path

  # ---- Load original model (using native dependencies) ----
  # This section IS allowed to use transformers, timm, etc.
  from transformers import AutoModel, AutoConfig

  config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
  original = AutoModel.from_config(config)  # random init, same architecture
  original.eval()

  # ---- Load wrapper model (self-contained) ----
  sys.path.insert(0, str(Path(__file__).parent.parent / "level_3_model"))
  from smolvlm import Model, get_inputs, get_init_inputs

  wrapper = Model(*get_init_inputs())
  wrapper.eval()

  # ---- Transfer weights: original → wrapper ----
  orig_sd = original.state_dict()
  wrap_sd = wrapper.state_dict()

  # Build mapping (auto-match by name, then by shape)
  mapped = 0
  for key in wrap_sd:
      if key in orig_sd and orig_sd[key].shape == wrap_sd[key].shape:
          wrap_sd[key] = orig_sd[key].clone()
          mapped += 1
      # Also try with prefix adjustments, e.g.:
      # "model.encoder.layers.0.weight" ↔ "encoder.layers.0.weight"

  wrapper.load_state_dict(wrap_sd)
  print(f"Mapped {mapped}/{len(wrap_sd)} parameters")

  # ---- Numerical comparison (3 trials) ----
  num_trials = 3
  max_diff_all = 0.0
  all_pass = True

  for trial in range(num_trials):
      torch.manual_seed(42 + trial)
      inputs = get_inputs()

      with torch.no_grad():
          orig_out = original(*inputs)
          wrap_out = wrapper(*inputs)

      # Handle HF model outputs (may be dataclass/tuple, not raw tensor)
      if hasattr(orig_out, "last_hidden_state"):
          orig_tensor = orig_out.last_hidden_state
      elif isinstance(orig_out, tuple):
          orig_tensor = orig_out[0]
      else:
          orig_tensor = orig_out

      if hasattr(wrap_out, "last_hidden_state"):
          wrap_tensor = wrap_out.last_hidden_state
      elif isinstance(wrap_out, tuple):
          wrap_tensor = wrap_out[0]
      else:
          wrap_tensor = wrap_out

      diff = (orig_tensor.float() - wrap_tensor.float()).abs().max().item()
      max_diff_all = max(max_diff_all, diff)
      matches = torch.allclose(orig_tensor.float(), wrap_tensor.float(),
                                rtol=1e-5, atol=1e-6)
      if not matches:
          all_pass = False
      print(f"Trial {trial}: max_diff={diff:.2e} {'PASS' if matches else 'FAIL'}")

  print(f"\n{'PASS' if all_pass else 'FAIL'} (max_diff={max_diff_all:.2e})")
  sys.exit(0 if all_pass else 1)
  ```

  #### What This Catches

  | Failure | Cause |
  |---|---|
  | Shape mismatch | Wrapper has wrong config values (hidden_dim, num_heads, etc.) |
  | Large numerical diff | Missing operation (e.g., forgot a normalization layer during extraction) |
  | Weight mapping gaps | Wrapper restructured the module tree, breaking state_dict key names |
  | Output format mismatch | Wrapper returns raw tensor but original returns a ModelOutput dataclass |

  #### If verification fails:
  - Check weight mapping — print unmapped keys from both sides
  - Check that the wrapper's forward() includes ALL operations (norms, activations, residuals)
  - Check config values match exactly
  - For HF models: ensure you handle the output format (`.last_hidden_state`, `.logits`, etc.)

  #### PASS criteria:
  - All 3 trials pass with `rtol=1e-5, atol=1e-6` (float32) or `rtol=1e-3, atol=1e-3` (bfloat16/float16)
  - Weight mapping covers >95% of parameters (some buffers like position IDs may not need mapping)

  **You MUST NOT proceed to Phase 1 until Stage B PASSES.**

  #### Key design points:

  1. **The verify script is model-specific** — unlike `verify_step.py` which is generic, the Phase 0
     verification *must* use the original model's native loading code (transformers, config objects, etc.).
     This is the only place in the pipeline where external dependencies are allowed.

  2. **It catches the hardest bugs early** — if the wrapper is wrong, every subsequent decomposition
     step will be verified against a broken baseline. This gate prevents that.

  3. **HF output format handling** — HuggingFace models return `ModelOutput` dataclasses, not raw
     tensors. The verification must extract the right field (`.logits`, `.last_hidden_state`, etc.).

  4. **Weight mapping is the critical step** — the wrapper likely restructures the module hierarchy
     (e.g., removing a `model.` prefix), so state_dict keys won't match directly. The verification
     script forces the agent to get this right.

  Once Phase 0 is complete, the wrapper file IS your input to Phase 1.

  ### 2. Modify the Agent Template for Multi-Source Input

  In `AGENT_PROMPT_TEMPLATE.md`, change the `[EDIT]` section to support richer input:

  ```markdown
  ## ──── [EDIT THIS SECTION] ────────────────────────────────────────────

  Input type: huggingface  # one of: single_file | huggingface | repo

  # For single_file:
  Model to decompose:   path/to/model.py

  # For huggingface:
  Model ID or repo:     HuggingFaceTB/SmolVLM-256M-Instruct
  Model class:          SmolVLMForConditionalGeneration
  Config overrides:     {num_hidden_layers: 2, hidden_size: 768}  # optional, for smaller test
  Entry forward args:   input_ids (int64 [1,128]), pixel_values (float32 [1,3,224,224])

  # For repo:
  Repo path:            path/to/smolvla/
  Entry file:           modeling_smolvla.py
  Entry class:          SmolVLAForConditionalGeneration
  Dependencies:         configuration_smolvla.py, processing_utils.py

  Output directory:     decomposition_workspace/output/level3/smolvla/
  ## ──────────────────────────────────────────────────────────────────────

  3. Add Prompt Guidance for Complex Forward Signatures

  In MAIN_PROMPT.md, add to the Component File Format section:

  ### Multi-Input Models

  If the original model takes multiple named inputs (e.g., input_ids, attention_mask,
  pixel_values), `get_inputs()` should return them as a list in positional order:

      def get_inputs():
          return [
              torch.randint(0, 32000, (1, 128)),      # input_ids
              torch.ones(1, 128, dtype=torch.long),    # attention_mask
              torch.randn(1, 3, 224, 224),             # pixel_values
          ]

  When decomposing, each child component receives only the inputs it needs.
  Document which parent inputs flow to which children in the architecture analysis.

  4. Add Guidance for Config-Driven Models

  Add to MAIN_PROMPT.md Phase 1:

  ### Step 1.0: Resolve Configuration (config-driven models only)

  If the model uses a config object (HuggingFace-style):
  1. Pick a CONCRETE configuration (specific model size/variant)
  2. Hardcode ALL config values as constants or __init__ parameters
  3. Do NOT use `AutoConfig`, `from_pretrained()`, or any dynamic loading
  4. Document the exact config used in decomposition_analysis.md

  Example: For SmolVLM-256M, resolve all config values and use:
      HIDDEN_SIZE = 768
      NUM_ATTENTION_HEADS = 12
      NUM_HIDDEN_LAYERS = 12
      INTERMEDIATE_SIZE = 3072

  5. Handle the verify_step.py Limitations

  The verification script needs a small tweak for models with multi-argument forward():

  In verify_step.py line 610, original(*trial_inputs) assumes get_inputs() returns positional args. This already works if get_inputs() returns
  a list. But you may want to support keyword arguments for HuggingFace models that are kwarg-heavy. Add:

  # In load_model_from_file, also look for:
  get_input_kwargs = getattr(module, "get_input_kwargs", lambda: {})

  # In the comparison loop:
  orig_out = original(*trial_inputs, **trial_kwargs)

  6. Add a "Dependency-Aware" Decomposition Strategy

  For models like smolVLA that have a vision encoder + language model + projector, add guidance:

  ### Multi-Modal / Composite Models

  For models with distinct sub-systems (e.g., vision encoder + LLM + connector):

  1. Level 3 decomposition should first separate the major sub-systems
  2. Each sub-system becomes its own Level 2 decomposition target
  3. Sub-systems may share tokenizers/processors — these are NOT part of the
     computation graph and should be excluded
  4. Weight tying between sub-systems (e.g., shared embeddings) must be
     explicitly documented and handled in weight_map.json

  ---
  Summary of Changes Needed

  | What to Change | Where | Effort |
  |---|---|---|
  | Add Phase 0 (model preparation) | MAIN_PROMPT.md | Main change — new section |
  | Multi-source [EDIT] block | AGENT_PROMPT_TEMPLATE.md | Small template change |
  | Multi-input forward guidance | MAIN_PROMPT.md Component Format | Small addition |
  | Config resolution step | MAIN_PROMPT.md Phase 1 | Small addition |
  | get_input_kwargs() support | verify_step.py | ~15 lines of code |
  | Multi-modal decomposition guidance | MAIN_PROMPT.md Phase 2 | Small addition |


Context

 The decomposition pipeline currently assumes single self-contained .py files. To support HuggingFace models, multi-file repos (like       
 smolVLA), and models with external dependencies, we need to add a "Phase 0" preparation step with a verification gate, update the agent   
 template for richer input types, add prompt guidance for config-driven and multi-input models, and extend verify_step.py to support       
 keyword arguments.

 All changes are based on the design notes in logs/2-27-dependencies.md.

 ---
 Files to Modify

 1. prompts/MAIN_PROMPT.md — 4 insertions
 2. AGENT_PROMPT_TEMPLATE.md — rewrite template for multi-source input
 3. scripts/verify_step.py — add get_input_kwargs support

 ---
 Changes

 1. prompts/MAIN_PROMPT.md

 1a. Insert Phase 0 between "Abstraction Levels" (line 28) and "Phase 1: Analysis" (line 31)

 Add new section ## Phase 0: Model Preparation (for multi-file / external-dependency models) containing:
 - Step 0.1: Identify the Computation Graph Entry Point
 - Step 0.2: Create a Self-Contained Wrapper (with code examples)
 - Step 0.3: Handle External Dependencies
 - Step 0.4: Verify the Wrapper (GATE) — Stage A (standalone execution) + Stage B (numerical equivalence with example verify_wrapper.py    
 script, failure table, pass criteria, key design points)
 - Note: "If your model is already a single self-contained .py file, skip to Phase 1."

 1b. Insert Step 1.0 at the start of Phase 1 (after line 31, before Step 1.1)

 Add ### Step 1.0: Resolve Configuration (config-driven models only) — guidance to hardcode config values.

 1c. Insert "Multi-Input Models" subsection after Component File Format (after line 312)

 Add guidance for get_inputs() returning multiple tensors and documenting input routing to children.

 1d. Insert "Multi-Modal / Composite Models" subsection in Phase 2 (after Step 2.5, before Component File Format)

 Add guidance for models with distinct sub-systems (vision encoder + LLM + connector).

 1e. Update output structure in Phase 4 to include steps/step_0_preparation/

 Add step_0_preparation/ with verify_wrapper.py and verification_result.json to the tree.

 2. AGENT_PROMPT_TEMPLATE.md

 2a. Update the [EDIT] block inside the prompt to support three input types:
 - single_file (current behavior, stays as default)
 - huggingface (model ID, class, config overrides, forward args)
 - repo (repo path, entry file, entry class, dependencies)

 2b. Add "Round 0: Model Preparation" before Round 1 in the Protocol section:
 - Only applies for huggingface/repo input types
 - Create self-contained wrapper → run verify_wrapper.py → PASS required before Round 1

 2c. Add HuggingFace and repo examples to the "Examples of the [EDIT] section"

 2d. Update Success Criteria checklist to include Phase 0 verification

 3. scripts/verify_step.py

 3a. Extend load_model_from_file (line 221-273) to also return get_input_kwargs:

 Change return type from Tuple[nn.Module, list, callable] to Tuple[nn.Module, list, callable, callable].

 Add after line 268:
 get_input_kwargs = getattr(module, "get_input_kwargs", lambda: {})

 Return becomes: return model, inputs, get_inputs, get_input_kwargs

 3b. Update all call sites of load_model_from_file in verify_step (lines 500-523):

 original, orig_inputs, get_inputs_fn, get_input_kwargs_fn = load_model_from_file(...)

 For the refactored model, capture but ignore kwargs (it uses same inputs as original):
 refactored, _, _, _ = load_model_from_file(...)

 3c. Update the trial loop (lines 601-639) to pass kwargs:

 After trial_inputs = get_inputs_fn(), add:
 trial_kwargs = get_input_kwargs_fn()

 Change lines 610 and 618 from:
 orig_out = original(*trial_inputs)
 ref_out = refactored(*trial_inputs)
 to:
 orig_out = original(*trial_inputs, **trial_kwargs)
 ref_out = refactored(*trial_inputs, **trial_kwargs)

 ---
 Verification

 After implementation:
 1. Confirm verify_step.py still works on existing single-file models (no regression):
 python scripts/verify_step.py --help
 2. Read through the modified MAIN_PROMPT.md to confirm Phase 0 flows logically into Phase 1
 3. Confirm the agent template renders correctly with the new [EDIT] examples



Here's a summary of what was changed:

 prompts/MAIN_PROMPT.md                                                                                                                      
  - Phase 0 added (between Abstraction Levels and Phase 1) — Steps 0.1-0.4 covering entry point identification, self-contained wrapper     
  creation, dependency handling, and the two-stage verification gate (standalone execution + numerical equivalence with example
  verify_wrapper.py)
  - Step 1.0 added at top of Phase 1 — config resolution guidance for HuggingFace-style models
  - Multi-Modal / Composite Models subsection added in Phase 2 — guidance for vision+LLM+connector architectures
  - Multi-Input Models subsection added after Component File Format — get_inputs() and get_input_kwargs() patterns
  - Output structure updated to include steps/step_0_preparation/

  AGENT_PROMPT_TEMPLATE.md

  - [EDIT] block now includes Input type: field (single_file | huggingface | repo)
  - Round 0 added to protocol — model preparation for non-single-file inputs
  - Success criteria updated with Phase 0 verification
  - New examples added for HuggingFace and multi-file repo input types

  scripts/verify_step.py

  - load_model_from_file now returns 4 values: (model, inputs, get_inputs_fn, get_input_kwargs_fn)
  - get_input_kwargs defaults to lambda: {} — fully backwards compatible with existing models
  - Trial loop passes **trial_kwargs to both original and refactored model calls