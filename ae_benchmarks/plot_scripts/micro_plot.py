#!/usr/bin/env python3
"""
Benchmark Plotting Tool for DiOMP and MPI Bandwidth Tests

This script reads output files from diomp_bw and mpi_bw benchmarks and creates
comparative bandwidth and latency plots saved as PNG files.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import argparse
import os

def parse_benchmark_output(output: str) -> Tuple[List[int], List[float], List[float], str]:
    """
    Parse benchmark output and extract size, bandwidth, latency data, and operation type.
    
    Args:
        output (str): Raw output from benchmark program
        
    Returns:
        Tuple of (sizes, bandwidths, latencies, operation)
    """
    sizes = []
    bandwidths = []
    latencies = []
    operation = "unknown"
    
    # Look for table rows with format: | size | bandwidth | latency | status | op |
    pattern = r'\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\w+\s*\|\s*(\w+)\s*\|'
    
    for line in output.split('\n'):
        match = re.search(pattern, line)
        if match:
            size = int(match.group(1))
            bandwidth = float(match.group(2))
            latency = float(match.group(3))
            op = match.group(4).lower()
            
            sizes.append(size)
            bandwidths.append(bandwidth)
            latencies.append(latency)
            operation = op
    
    return sizes, bandwidths, latencies, operation


def _filter_and_sort(x: List[float], y: List[float], xmin: float, xmax: float):
    """Filter points to [xmin, xmax] and sort by x ascending."""
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xmin <= xi <= xmax]
    pairs.sort(key=lambda p: p[0])
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def create_plots(benchmark_data: Dict, output_dir: str = "."):
    """
    Create bandwidth and latency comparison plots with up to 4 curves.
    
    Args:
        benchmark_data (dict): Dictionary containing all benchmark data
        output_dir (str): Directory to save plots
    """
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Define colors and markers for different combinations (maintain consistency)
    colors = {'diomp_get': 'blue', 'diomp_put': 'lightblue', 
              'mpi_get': 'red', 'mpi_put': 'orange'}
    markers = {'diomp_get': 'o', 'diomp_put': 's', 
               'mpi_get': 'o', 'mpi_put': 's'}
    labels = {'diomp_get': 'DiOMP Get', 'diomp_put': 'DiOMP Put',
              'mpi_get': 'MPI Get', 'mpi_put': 'MPI Put'}

    # --- 1) Bandwidth plot: X-axis range [1/64 MB, 128 MB] (converted to bytes for filtering) ---
    bw_xmin_bytes = (1/64) * 1024 * 1024
    bw_xmax_bytes = 128 * 1024 * 1024

    plt.figure(figsize=(12, 8))
    for key, data in benchmark_data.items():
        sizes = data.get('sizes', [])
        bws = data.get('bandwidths', [])
        if not sizes or not bws:
            continue

        # Filter & sort
        xs, ys = _filter_and_sort(sizes, bws, bw_xmin_bytes, bw_xmax_bytes)
        if not xs:
            continue

        # Plot log-log (bandwidth typically follows power law with message size)
        plt.loglog(xs, ys,
                   marker=markers[key], linestyle='-',
                   label=labels[key], linewidth=2, markersize=6, alpha=0.8)

    plt.xlabel('Message Size (bytes)')
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('Bandwidth Performance Comparison: Put/Get Operations')
    plt.xlim(bw_xmin_bytes, bw_xmax_bytes)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    bandwidth_path = os.path.join(output_dir, 'bandwidth_comparison.png')
    plt.savefig(bandwidth_path, dpi=300, bbox_inches='tight')
    print(f"Bandwidth plot saved to: {bandwidth_path}")
    plt.close()
    
    # --- 2) Latency plot: X-axis range [4, 8192] bytes; Y-axis linear, using semilogx ---
    lat_xmin = 4
    lat_xmax = 8192

    plt.figure(figsize=(12, 8))
    for key, data in benchmark_data.items():
        sizes = data.get('sizes', [])
        lats = data.get('latencies', [])
        if not sizes or not lats:
            continue

        # Filter & sort
        xs, ys = _filter_and_sort(sizes, lats, lat_xmin, lat_xmax)
        if not xs:
            continue

        # If original latency unit is seconds instead of microseconds, do unit adaptation here (<1s typical microsecond value)
        # If already in microseconds, this check won't trigger any changes
        # You can also remove this section to plot data "as is"
        if ys and 0 < max(ys) < 1:  # Guess unit is seconds
            ys = [y * 1e6 for y in ys]

        plt.semilogx(xs, ys,
                     marker=markers[key], linestyle='-',
                     label=labels[key], linewidth=2, markersize=6, alpha=0.8)

    plt.xlabel('Message Size (bytes)')
    plt.ylabel('Latency (microseconds)')
    plt.title('Latency Performance Comparison: Put/Get Operations')
    plt.xlim(lat_xmin, lat_xmax)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    latency_path = os.path.join(output_dir, 'latency_comparison.png')
    plt.savefig(latency_path, dpi=300, bbox_inches='tight')
    print(f"Latency plot saved to: {latency_path}")
    plt.close()


def plot_from_files(diomp_get_file: str = None, diomp_put_file: str = None,
                   mpi_get_file: str = None, mpi_put_file: str = None, output_dir: str = "."):
    """
    Read benchmark data from output files and create plots.
    
    Args:
        diomp_get_file (str): Path to DiOMP get benchmark output file
        diomp_put_file (str): Path to DiOMP put benchmark output file
        mpi_get_file (str): Path to MPI get benchmark output file
        mpi_put_file (str): Path to MPI put benchmark output file
        output_dir (str): Directory to save plots
    """
    benchmark_data = {
        'diomp_get': {'sizes': [], 'bandwidths': [], 'latencies': []},
        'diomp_put': {'sizes': [], 'bandwidths': [], 'latencies': []},
        'mpi_get': {'sizes': [], 'bandwidths': [], 'latencies': []},
        'mpi_put': {'sizes': [], 'bandwidths': [], 'latencies': []}
    }
    
    file_mapping = {
        'diomp_get': diomp_get_file,
        'diomp_put': diomp_put_file,
        'mpi_get': mpi_get_file,
        'mpi_put': mpi_put_file
    }
    
    for key, file_path in file_mapping.items():
        if file_path and os.path.exists(file_path):
            print(f"Processing {key}: {file_path}")
            with open(file_path, 'r') as f:
                output = f.read()
                sizes, bandwidths, latencies, operation = parse_benchmark_output(output)
                benchmark_data[key] = {'sizes': sizes, 'bandwidths': bandwidths, 'latencies': latencies}
                print(f"Loaded {key} data: {len(sizes)} data points")
        elif file_path:
            print(f"File not found: {file_path}")
        else:
            print(f"No file specified for {key}")
    
    create_plots(benchmark_data, output_dir)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Plot bandwidth and latency comparisons from DiOMP and MPI benchmark output files'
    )
    parser.add_argument('--diomp-get-file', type=str, help='Path to DiOMP get benchmark output file')
    parser.add_argument('--diomp-put-file', type=str, help='Path to DiOMP put benchmark output file')
    parser.add_argument('--mpi-get-file', type=str, help='Path to MPI get benchmark output file')
    parser.add_argument('--mpi-put-file', type=str, help='Path to MPI put benchmark output file')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if any input files are provided
    if args.diomp_get_file or args.diomp_put_file or args.mpi_get_file or args.mpi_put_file:
        print("Reading from output files...")
        plot_from_files(args.diomp_get_file, args.diomp_put_file, 
                       args.mpi_get_file, args.mpi_put_file, args.output_dir)
        print("Plotting completed!")
    else:
        print("No input files specified.")
        print("\nUsage:")
        print("  python3 micro_plot.py --diomp-get-file diomp_bw_get.txt --diomp-put-file diomp_bw_put.txt \\")
        print("                            --mpi-get-file mpi_bw_get.txt --mpi-put-file mpi_bw_put.txt")
        print("\nExample workflow:")
        print("  1. Run benchmarks manually:")
        print("     ./diomp_bw get > diomp_get.txt")
        print("     ./diomp_bw put > diomp_put.txt")
        print("     ./mpi_bw get > mpi_get.txt")
        print("     ./mpi_bw put > mpi_put.txt")
        print("  2. Generate plots:")
        print("     python3 micro_plot.py --diomp-get-file diomp_get.txt --diomp-put-file diomp_put.txt \\")
        print("                               --mpi-get-file mpi_get.txt --mpi-put-file mpi_put.txt")
        return

if __name__ == '__main__':
    main()
