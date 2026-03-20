#!/usr/bin/env python3
"""
Analyze kernel benchmark results and generate detailed reports.
"""

import json
import pandas as pd
from pathlib import Path

def load_benchmark_data(json_file: str):
    """Load benchmark data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_performance_analysis(data):
    """Create detailed performance analysis."""
    results = data['results']
    
    # Convert to DataFrame for easier analysis
    rows = []
    for result in results:
        row = {
            'kernel_name': result['kernel_name'],
            'success': result['success'],
            'status': result['metrics'].get('status', 'unknown'),
            'avg_cycles': result['metrics'].get('avg_cycles'),
            'min_cycles': result['metrics'].get('min_cycles'),
            'max_cycles': result['metrics'].get('max_cycles'),
            'avg_npu_time_us': result['metrics'].get('avg_npu_time_us'),
            'min_npu_time_us': result['metrics'].get('min_npu_time_us'),
            'num_invocations': result['metrics'].get('num_invocations'),
            'execution_time_seconds': result['execution_time_seconds'],
            'error_message': result['error_message']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def categorize_kernels(df):
    """Categorize kernels by type and complexity."""
    categories = {
        'Convolution': [],
        'Pooling': [],
        'Elementwise': [],
        'Reduction': [],
        'Matrix Operations': [],
        'Type Casting': [],
        'Comparison': [],
        'Mathematical': [],
        'Other': []
    }
    
    for _, row in df.iterrows():
        name = row['kernel_name']
        if 'conv' in name:
            categories['Convolution'].append(name)
        elif 'pool' in name:
            categories['Pooling'].append(name)
        elif any(op in name for op in ['elementwise', 'vectoradd', 'vectorsub', 'vectormult']):
            categories['Elementwise'].append(name)
        elif any(op in name for op in ['reduce', 'argmax', 'argmin', 'dotproduct']):
            categories['Reduction'].append(name)
        elif any(op in name for op in ['matmul', 'matrix']):
            categories['Matrix Operations'].append(name)
        elif 'cast' in name:
            categories['Type Casting'].append(name)
        elif 'compare' in name:
            categories['Comparison'].append(name)
        elif any(op in name for op in ['sqrt', 'sin', 'tanh', 'sigmoid', 'ceil']):
            categories['Mathematical'].append(name)
        else:
            categories['Other'].append(name)
    
    return categories

def main():
    # Load data
    json_file = "/home/jl4257/llm_codegen/npueval_dataset/kernel_benchmark_results.json"
    data = load_benchmark_data(json_file)
    df = create_performance_analysis(data)
    
    print("🔍 DETAILED BENCHMARK ANALYSIS")
    print("=" * 60)
    
    # Overall summary
    summary = data['summary']
    print(f"📊 Overall Results:")
    print(f"   Total kernels: {summary['total_kernels']}")
    print(f"   ✅ Passed: {summary['passed']} ({summary['success_rate']*100:.1f}%)")
    print(f"   ❌ Failed: {summary['failed']}")
    print(f"   💥 Errors: {summary['errors']}")
    print(f"   ⏱️  Total time: {summary['total_execution_time_seconds']:.1f}s")
    print()
    
    # Successful kernels analysis
    successful_df = df[df['success'] == True].copy()
    if not successful_df.empty:
        print(f"🏆 Performance Analysis ({len(successful_df)} successful kernels):")
        print(f"   Avg cycles: {successful_df['avg_cycles'].mean():.1f}")
        print(f"   Min cycles: {successful_df['avg_cycles'].min():.1f}")
        print(f"   Max cycles: {successful_df['avg_cycles'].max():.1f}")
        print(f"   Median cycles: {successful_df['avg_cycles'].median():.1f}")
        print()
        
        # Top 15 fastest kernels
        fastest = successful_df.nsmallest(15, 'avg_cycles')
        print("🚀 TOP 15 FASTEST KERNELS:")
        for i, (_, row) in enumerate(fastest.iterrows(), 1):
            print(f"   {i:2d}. {row['kernel_name']:<35} {row['avg_cycles']:>8.1f} cycles")
        print()
        
        # Top 10 slowest kernels  
        slowest = successful_df.nlargest(10, 'avg_cycles')
        print("🐌 TOP 10 SLOWEST KERNELS:")
        for i, (_, row) in enumerate(slowest.iterrows(), 1):
            print(f"   {i:2d}. {row['kernel_name']:<35} {row['avg_cycles']:>8.1f} cycles")
        print()
    
    # Failed/Error analysis
    failed_df = df[df['success'] == False]
    if not failed_df.empty:
        print(f"❌ Failed/Error Kernels ({len(failed_df)} kernels):")
        for _, row in failed_df.iterrows():
            status_emoji = "❌" if row['status'] == 'fail' else "💥"
            print(f"   {status_emoji} {row['kernel_name']}")
            if row['error_message']:
                # Print first line of error message
                first_line = row['error_message'].split('\n')[0]
                print(f"      └─ {first_line}")
        print()
    
    # Kernel categorization
    categories = categorize_kernels(successful_df)
    print("📂 Kernel Categories (successful only):")
    for category, kernels in categories.items():
        if kernels:
            print(f"   {category}: {len(kernels)} kernels")
            if kernels:
                # Show avg cycles for this category
                cat_df = successful_df[successful_df['kernel_name'].isin(kernels)]
                if not cat_df.empty:
                    avg_cycles = cat_df['avg_cycles'].mean()
                    print(f"      └─ Avg cycles: {avg_cycles:.1f}")
    print()
    
    # Data type analysis
    print("🔢 Data Type Analysis:")
    dtype_stats = {}
    for _, row in successful_df.iterrows():
        name = row['kernel_name']
        if 'int8' in name:
            dtype = 'int8'
        elif 'int16' in name:
            dtype = 'int16'
        elif 'int32' in name:
            dtype = 'int32'
        elif 'bfloat16' in name:
            dtype = 'bfloat16'
        elif 'float32' in name:
            dtype = 'float32'
        else:
            dtype = 'mixed/other'
        
        if dtype not in dtype_stats:
            dtype_stats[dtype] = []
        dtype_stats[dtype].append(row['avg_cycles'])
    
    for dtype, cycles_list in dtype_stats.items():
        if cycles_list:
            avg_cycles = sum(cycles_list) / len(cycles_list)
            print(f"   {dtype}: {len(cycles_list)} kernels, avg {avg_cycles:.1f} cycles")
    print()
    
    # Save detailed CSV for further analysis
    csv_file = "/home/jl4257/llm_codegen/npueval_dataset/kernel_performance_details.csv"
    df.to_csv(csv_file, index=False)
    print(f"📁 Detailed results saved to: {csv_file}")
    print()
    
    print("✨ Analysis complete!")

if __name__ == "__main__":
    main()


