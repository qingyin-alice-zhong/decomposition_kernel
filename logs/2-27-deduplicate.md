  Would you like me to add deduplication guidance to the prompts? The approach would be:

  - Kernel identity = (operation, input_shapes, output_shapes, dtype, parameter_shapes) — if two kernels match on all of these, they're the
   same workload
  - Keep one canonical file per unique kernel (e.g., linear_768x3072_fp32.py)
  - In decomposition_tree.json, multiple nodes reference the same kernel file
  - The refactored.py creates separate instances (different weights) but from the same class — from linear_768x3072_fp32 import Model as   
  Linear1 and from linear_768x3072_fp32 import Model as Linear2
  - Weight transfer maps different weight sets to different instances of the same kernel

  Changes
                                                                                                                                             prompts/MAIN_PROMPT.md — new "Kernel Deduplication" section (before Component File Format):
  - Defines kernel identity = (operation, input_shapes, output_shapes, dtype, parameter_shapes)                                            
  - One file per unique kernel, no _2/_3 suffixes                                                                                          
  - Shows how refactored.py imports the same file multiple times as separate instances
  - Clarifies when two kernels are NOT duplicates (different shapes, dtypes, ops)

  OUTPUT_SCHEMA.md — three updates:
  - Added "Kernel Deduplication" rule to naming conventions
  - Added unique_kernels count and instance_count per node to decomposition_tree.json schema
  - Updated example output to show deduplicated kernel listing with instance counts (10 unique files, ~250 instances)

  prompts/VERIFICATION_GUIDE.md — new edge case:
  - How to handle weight mapping for deduplicated kernels (multiple instances of same file, each with their own weights)