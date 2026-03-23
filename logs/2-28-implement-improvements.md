Changes Summary                                                                                                                                                                                                                                                                     
  1. OUTPUT_SCHEMA.md                                                                                                                      

  - Clarified unique workload naming ({operation}_{shape_signature}_{dtype}.py) with detailed examples
  - File name encodes workload identity; module uses get_init_inputs() for dimensions (not hardcoded)
  - Added kernel identity/deduplication rules: (operation, parameter_shapes, input_dtype, output_dtype)
  - Updated example output tree to use shape-based names + decomposition_log.json

  2. MAIN_PROMPT.md

  - Semantic classification (Step 1.2): Replaced mechanical op-count heuristic with semantic guidance — RMSNorm, RoPE, Softmax are
  kernel-level even with multiple primitives internally
  - Anti-cheat rule #8: No raw nn.Module construction in __init__() — all children must be imported from child files
  - Anti-cheat rule #10: Kernel files must use shape-based naming
  - Kernel deduplication: Updated with get_init_inputs() pattern, not hardcoded dims
  - Coverage report (Step 3.2): Marked as REQUIRED, not optional
  - Decision/difficulty log: New section with structured decomposition_log.json format
  - Updated constraints: "You MUST" list now includes coverage report, semantic classification, logging
  - Removed edge case guidance per your request

  3. scripts/verify_step.py

  - Added DISALLOWED_INIT_PATTERNS list for __init__() scanning
  - check_anticheat_source() now scans both forward() and __init__()
  - Violations include location field ("forward()" or "__init__()")
  - Verified: correctly catches nn.Linear/nn.Embedding in step_1's refactored.py

  4. scripts/extract_ops.py

  - Added OP_CANONICAL_MAP: normalizes names between torch.compile (ATen) and hooks (nn.Module)
  - Added DATA_PLUMBING_OPS: filters out shape/data manipulation ops from coverage comparison
  - Added is_compute_op() and normalize_op_type() helpers
  - Coverage now compares only compute ops (linear, conv2d, embedding, etc.), excluding reshapes/views/indexing/primitive arithmetic       
  - Tested: correctly identifies 7 compute op types in SmolVLA original model

  5. prompts/VERIFICATION_GUIDE.md

  - Updated L0 kernel description with semantic classification rule
  - Updated anti-cheat rules to cover both forward() and __init__()
  - Updated checklist with coverage_summary.json, semantic classification, shape naming, decomposition_log.json

  ## Mix level children:

  will there be the case that a layer level can only be decomposed to kernel + fusion? instead of all fusion level?                                                                                                                                            
● Yes, absolutely. Consider a TransformerBlock (L2):                                                                                       
  - Attention → fusion (L1): multiple distinct ops (QKV proj + matmul + softmax + matmul + out proj)
  - MLP → fusion (L1): multiple distinct ops (Linear + activation + Linear)                                                                  - RMSNorm → kernel (L0): single logical operation
  - Residual add → data plumbing (stays in refactored forward)                                                                             
  
  The current rule — "each level decomposes ONLY to the next level down" — forces you to wrap RMSNorm in a fake "fusion" just to maintain
  the hierarchy. That's exactly what happened with rms_norm.py in the SmolVLA decomposition.

  A more natural rule would be: a component can decompose into children at the next level down OR any level below, as long as:
  - No child is at the same level as the parent
  - Each child is classified by its semantic role (not forced into a level)

  So L2 → {L1, L0} is fine. L3 → {L2, L0} would also be fine (e.g., a model with a vision encoder layer + a standalone embedding kernel).  

  Want me to update the instructions to allow mixed-level children?

❯ yes I think this will be more flexible. but also I think its necesssary to be clear that we want to decompose to next level if             possible, then smaller peices
                                                                                                                                           
● Read 1 file (ctrl+o to expand)                                                                                                           
                                                                                                                                           
● Update(prompts\MAIN_PROMPT.md)                                                                                                           
  ⎿  Added 8 lines, removed 1 line
      24  MODEL ──decompose──> LAYERS ──decompose──> FUSIONS ──decompose──> KERNELS                                                        
      25  ```                                                                                                                                    26                                                                                                                                   
      27 -**Key Principle**: Each level decomposes ONLY to the next level down, never skipping levels.                                     
      27 +**Key Principle**: Prefer decomposing to the **next level down** first. Extract the largest meaningful children as the    
         +primary decomposition targets. However, if a child component is semantically at a **lower level** (e.g., a single-op      
         +kernel sitting directly inside a layer), classify it at its natural level — do NOT wrap it in a fake intermediate level.         
      28
      29 +**Allowed parent → child level relationships:**                                                                                  
      30 +- Model (L3) → Layer (L2), Fusion (L1), Kernel (L0)                                                                              
      31 +- Layer (L2) → Fusion (L1), Kernel (L0)                                                                                          
      32 +- Fusion (L1) → Kernel (L0)                                                                                                      
      33 +                                                                                                                                 
      34 +**Priority**: Always look for next-level-down children first. Only classify a child at a lower level when it genuinely           
         +belongs there (e.g., a standalone `nn.Embedding` inside a model is a kernel, not a layer).                                       
      35 +                                                                                                                                 
      36  ---
      37
      38  ## Phase 0: Model Preparation (for multi-file / external-dependency models)

● Now update the decomposition targets table to reflect this:

● Update(prompts\MAIN_PROMPT.md)                                                                                                             ⎿  Added 6 lines, removed 6 lines
      314                                                                                                                                        315  ### Step 1.3: Identify Decomposition Targets
      316                                                                                                                                  
      317 -For the CURRENT level, identify what to extract at the NEXT level down:
      317 +For the CURRENT level, identify children by their **semantic role**. Prefer next-level-down, but classify each child at  
          +its natural level:                                                                                                              
      318
      319 -| Current Level | Extract As | What to Look For |                                                                               
      320 -|--------------|------------|------------------|                                                                                
      321 -| Model (L3) | Layers (L2) | Repeated blocks, major sub-modules (encoder, decoder) |                                            
      322 -| Layer (L2) | Fusions (L1) | Attention block, MLP block, normalization groups |                                                
      323 -| Fusion (L1) | Kernels (L0) | Individual operations (linear, activation, norm) |                                               
      319 +| Current Level | Primary Children | May Also Include |                                                                         
      320 +|--------------|------------------|------------------|                                                                          
      321 +| Model (L3) | Layers (L2): repeated blocks, major sub-modules | Kernels (L0): standalone embeddings, projections |             
      322 +| Layer (L2) | Fusions (L1): attention block, MLP block | Kernels (L0): norms, activations that stand alone |                   
      323 +| Fusion (L1) | Kernels (L0): individual operations | — (L0 is the leaf level) |                                                
      324
      325  ---
      326

● Now update the decomposition order in Step 2.5:

● Read 1 file (ctrl+o to expand)                                                                                                           

● Update(prompts\MAIN_PROMPT.md)                                                                                                           
  ⎿  Added 5 lines, removed 3 lines
      461  For each child that is NOT at kernel level (L0), repeat Steps 2.1-2.4.                                                          
      462                                                                                                                                        463  **Decomposition order:**                                                                                                        
      464 -1. Model (L3) → Layers (L2) — verify step                                                                                       
      465 -2. For each Layer: Layer (L2) → Fusions (L1) — verify each step
      466 -3. For each Fusion: Fusion (L1) → Kernels (L0) — verify each step
      464 +1. Model (L3) → children (primarily L2 layers, may include L1 fusions or L0 kernels) — verify step                              
      465 +2. For each non-kernel child: decompose further — verify each step                                                              
      466 +3. Continue until all leaf nodes are at kernel level (L0)                                                                       
      467
      468 +Each decomposition step may produce a **mix of levels** among its children. For example, decomposing a Layer (L2) might         
          +yield 2 Fusions (L1) + 1 Kernel (L0). Only the non-kernel children need further decomposition.                                  
      469 +                                                                                                                                
      470  ### Multi-Modal / Composite Models
      471
      472  For models with distinct sub-systems (e.g., vision encoder + LLM + connector like SmolVLA):

● Also update the constraints section which still says "Skip abstraction levels":

● Update(prompts\MAIN_PROMPT.md)                                                                                                             ⎿  Added 1 line, removed 1 line
      782                                                                                                                                        783  ### You MUST NOT:
      784  1. Copy the entire model as a single "kernel"                                                                                   
      785 -2. Skip abstraction levels (model → kernel without layer/fusion)
      785 +2. Force components into the wrong abstraction level (e.g., wrapping a single-op kernel as a fake "fusion" just to       
          +maintain strict level ordering)                                                                                                 
      786  3. Combine unrelated operations into fake "fusions"
      787  4. Invent operations not in the original
      788  5. Generate files that don't execute

● Now update the VERIFICATION_GUIDE.md to match:  

● Update(prompts\VERIFICATION_GUIDE.md)                                                                                                      ⎿  Added 1 line, removed 1 line
      286  [ ] Kernel classification is SEMANTIC (RMSNorm, RoPE → kernel, not fusion)                                                            287  [ ] Kernel files use shape-based naming: {operation}_{shape_signature}_{dtype}.py
      288  [ ] All leaf nodes are at kernel level                                                                                          
      289 -[ ] Abstraction hierarchy is respected (L3→L2→L1→L0)
      289 +[ ] Abstraction hierarchy is respected (children are always at a lower level than parent)
      290  [ ] decomposition_tree.json is complete
      291  [ ] decomposition_log.json records key decisions and difficulties
      292  [ ] All dtypes are preserved

● And update the MAIN_PROMPT checklist:

● Update(prompts\MAIN_PROMPT.md)                                                                                                             ⎿  Added 1 line, removed 1 line
      816  - [ ] `composition_test.py` PASSES (final composed output matches original)                                                           817  - [ ] `decomposition_tree.json` is complete
      818  - [ ] Every leaf node is at kernel level (L0)                                                                                   
      819 -- [ ] Abstraction level hierarchy is respected (L3 → L2 → L1 → L0)
      819 +- [ ] Abstraction hierarchy is respected (children always at a lower level than parent; prefer next level down)
      820  - [ ] **`coverage_summary.json` exists** in `verification/` directory (from `extract_ops.py`)
      821  - [ ] Coverage analysis shows no missing operations
      822  - [ ] **Kernel files use shape-based naming**: `{operation}_{shape_signature}_{dtype}.py` with hardcoded dimensions

● Let me verify the changes are consistent:       

● Bash(cd C:/Users/liang/decomposition_workspace && grep -n "skip.*level\|ONLY.*next.*level\|never skip" prompts/MAIN_PROMPT.md                  prompts/VERIFICATION_GUIDE.md)
  ⎿  No matches found                                                                                                                         
● No leftover references to the old strict rule. Done. Here's what changed:                                                                
  
  MAIN_PROMPT.md:                                                                                                                            - Key Principle rewritten: prefer next-level-down, but classify children at their natural level — no fake wrappers
  - Allowed relationships explicitly listed: L3→{L2,L1,L0}, L2→{L1,L0}, L1→{L0}                                                            
  - Step 1.3 table: now shows "Primary Children" + "May Also Include" columns
  - Step 2.5: decomposition order updated to reflect mixed-level children
  - Constraint #2: changed from "don't skip levels" to "don't force wrong abstraction level"
  - Checklist: updated hierarchy check wording

  VERIFICATION_GUIDE.md:
  - Checklist updated to match

  The rule is now: decompose to the next level down as much as possible, then classify any remaining smaller pieces at their natural level.
   A TransformerBlock (L2) can have Attention (L1) + MLP (L1) + RMSNorm (L0) as children without wrapping RMSNorm in a pointless fusion.   

   ❯ yes. and the only reason we don't do RMSNorm in fusion is because it can't be fused with other operations right?                                                                                                                                           
● Exactly right. The classification is driven by fusibility: a fusion (L1) exists because its constituent operations can be fused into a   
  single kernel launch (e.g., Linear+SiLU, Conv+BN+ReLU). RMSNorm sits alone because there's typically a residual connection or other data
  flow break between it and adjacent ops, so it can't be fused with them.                                                                     
  Let me update the classification guidance to ground it in this reasoning:                                                                
  
● Update(prompts\MAIN_PROMPT.md)
  ⎿  Added 17 lines, removed 3 lines
      307  | 3 (Model) | Complete model with multiple layers/stages | Full Transformer, VisionEncoder, LLM |                                     308
      309  **Key distinction between kernel and fusion:**                                                                                  
      310 -- **Kernel**: Wraps ONE logical operation (even if its implementation uses multiple torch primitives). Think: "could     
          -this be a single CUDA kernel?" Examples: RMSNorm (one normalization), RoPE (one positional encoding), Softmax (one       
          -probability distribution computation).                                                                                          
      311 -- **Fusion**: Combines multiple SEMANTICALLY DISTINCT operations that could each be their own kernel. Think: "this is           
          -doing two different things." Examples: Linear + SiLU (projection + activation), Attention = MatMul + Scale + Mask +             
          -Softmax + MatMul (multiple distinct steps).                                                                                     
      310 +- **Fusion (L1)**: Groups operations that CAN be fused together — i.e., executed in a single kernel launch without              
          +intermediate memory reads/writes. The operations are adjacent in the data flow with no branching (residual connections,         
          +skip connections) between them. Examples: Linear+SiLU (projection feeds directly into activation), Conv+BN+ReLU, QKV            
          +projection+reshape.                                                                                                             
      311 +- **Kernel (L0)**: A single logical operation that either stands alone (can't be fused with its neighbors due to                
          +residual connections or data flow breaks) or is already an atomic operation. Examples: RMSNorm (typically separated from        
          + adjacent ops by a residual add), LayerNorm, Softmax, standalone Linear projections.                                            
      312
      313 -**When in doubt:** If the operation has a well-known single name in ML literature (RMSNorm, LayerNorm, Softmax, RoPE,           
          -GELU, etc.), classify it as kernel.                                                                                             
      313 +**Classification process:**                                                                                                     
      314 +1. First, identify groups of adjacent operations that can be fused (no data flow breaks between them) → these become            
          +**fusions (L1)**                                                                                                                
      315 +2. Any remaining operations that can't be grouped with neighbors → these become **kernels (L0)**                                
      316
      317 +**Example — TransformerBlock decomposition:**                                                                                   
      318 +```                                                                                                                             
      319 +residual = x                                                                                                                    
      320 +x = self.rms_norm(x)          # L0 kernel (can't fuse with residual add above)                                                  
      321 +x = self.attention(x)          # L1 fusion (QKV proj + matmul + softmax + matmul + out proj)                                    
      322 +x = x + residual               # data plumbing (residual add breaks fusion opportunity)                                         
      323 +residual = x                                                                                                                    
      324 +x = self.rms_norm_2(x)         # L0 kernel (isolated by residual adds)                                                          
      325 +x = self.mlp(x)                # L1 fusion (Linear + SiLU + Linear)                                                             
      326 +x = x + residual               # data plumbing                                                                                  
      327 +```                                                                                                                             
      328 +                                                                                                                                
      329  ### Step 1.3: Identify Decomposition Targets
      330
      331  For the CURRENT level, identify children by their **semantic role**. Prefer next-level-down, but classify each child at
           its natural level:

● Now update the classification table to reflect this fusibility-based reasoning:

● Update(prompts\MAIN_PROMPT.md)                                                                                                             ⎿  Added 3 lines, removed 3 lines
      301                                                                                                                                        302  | Level | Criteria | Examples |
      303  |-------|----------|----------|                                                                                                 
      304 -| 0 (Kernel) | A single logical operation. If PyTorch offers it as one `nn.Module` or `F.*` function, it's kernel-level  
          -. May internally use multiple primitives. | `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, `nn.Embedding`, RMSNorm, 
          -Softmax, SiLU, GELU, MatMul |                                                                                                   
      305 -| 1 (Fusion) | 2-5 semantically distinct operations commonly fused together | Linear+Activation (SwiGLU),                       
          -QKV+Reshape+Attention, Conv+BN+ReLU |                                                                                           
      306 -| 2 (Layer) | A logical building block, often repeated. Contains multiple fusions                                               
          -| TransformerBlock, ResNetBlock, EncoderLayer, DecoderLayer |                                                                   
      304 +| 0 (Kernel) | A single logical operation that stands alone — either it can't be fused with its                                 
          +neighbors (due to residual connections, data flow breaks), or it's an atomic op. May use multiple primitives internally         
          +. | `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, RMSNorm, `nn.Embedding`, Softmax, SiLU, GELU, MatMul |                            
      305 +| 1 (Fusion) | Adjacent operations that CAN be fused (no data flow breaks between them). They form a straight-line              
          +computation without residual/skip connections interrupting. | Linear+SiLU (SwiGLU gate), QKV proj+reshape+attention             
          +scores, Conv+BN+ReLU |                                                                                                          
      306 +| 2 (Layer) | A logical building block, often repeated. Contains fusions                                                        
          +and/or kernels with residual connections between them. | TransformerBlock, ResNetBlock, EncoderLayer, DecoderLayer |            
      307  | 3 (Model) | Complete model with multiple layers/stages | Full Transformer, VisionEncoder, LLM |
      308
      309  **Key distinction between kernel and fusion:**

● Also update the VERIFICATION_GUIDE.md kernel/fusion descriptions to match:

● Update(prompts\VERIFICATION_GUIDE.md)                                                                                                      ⎿  Added 5 lines, removed 4 lines
      133  - **Decomposes to**: Fusions (attention, MLP, normalization groups)                                                                   134
      135  ### Level 1: Fusion                                                                                                             
      136 -- **Contains**: 2-5 tightly coupled operations
      137 -- **Examples**: Conv+BN+ReLU, QKV projection, SwiGLU, Attention scores
      136 +- **Contains**: Adjacent operations that CAN be fused (no data flow breaks between them)                                        
      137 +- **Examples**: Conv+BN+ReLU, Linear+SiLU (SwiGLU gate), QKV proj+reshape+attention scores                                      
      138 +- **Why fused**: These operations form a straight-line computation — output of one feeds directly into the next, with no        
          + residual/skip connections interrupting                                                                                         
      139  - **Decomposes to**: Kernels (individual operations)
      140
      141  ### Level 0: Kernel
      142 -- **Contains**: Single logical operation (may use multiple primitives internally)                                               
      142 +- **Contains**: Single logical operation that stands alone (can't be fused with neighbors)                                      
      143  - **Examples**: Linear, Conv2d, ReLU, LayerNorm, RMSNorm, RoPE, MatMul, Softmax, SiLU, GELU, Embedding
      144  - **Decomposes to**: Nothing (leaf node)
      145 -- **Classification rule**: If the operation has a well-known single name in ML literature (RMSNorm, RoPE, Softmax,              
          -etc.), it's kernel-level, even if its implementation uses `pow + mean + rsqrt + mul` internally                                 
      145 +- **Why standalone**: Typically isolated by residual connections, skip connections, or data flow branches that prevent          
          +fusion with adjacent operations                                                                                                 
      146
      147  ---
      148

● The classification is now grounded in fusibility:

  - Fusion (L1): operations grouped together because they can be fused — straight-line data flow, no residual/skip connections between them  - Kernel (L0): operations that stand alone because they can't be fused with neighbors — isolated by residual adds, branching, or data
  flow breaks                                                                                                                              
  
  RMSNorm is L0 not because "it has a single name" but because in a transformer block it sits between two residual adds, so there's no
  fusion opportunity with adjacent ops.