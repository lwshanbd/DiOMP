import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

platforms = ["Slingshot 11\n+ A100"]

def read_timing_data_from_csv(csv_file_path):
    """
    Read timing data from CSV file
    Expected CSV format: Size(bytes),Avg Time(ms),Bandwidth(MB/s)
    Returns: tuple of (data_sizes, avg_times)
    """
    df = pd.read_csv(csv_file_path)
    data_sizes = df['Size(bytes)'].tolist()
    avg_times = df['Avg Time(ms)'].tolist()
    return data_sizes, avg_times


# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate collective communication performance heatmap')
parser.add_argument('--mpi-csv', required=True, help='Path to MPI results CSV file')
parser.add_argument('--diomp-csv', required=True, help='Path to DiOMP results CSV file')
parser.add_argument('--output', '-o', default='coll.png', help='Output plot filename (default: coll.png)')
args = parser.parse_args()

# Read data from CSV files
mpi_csv_path = args.mpi_csv
diomp_csv_path = args.diomp_csv

try:
    data_sizes_b, mpi_a100 = read_timing_data_from_csv(mpi_csv_path)
    _, diomp_a100 = read_timing_data_from_csv(diomp_csv_path)
except FileNotFoundError as e:
    print(f"CSV file not found: {e}")
    # Fallback to default data sizes if CSV files are not available
    data_sizes_b = [
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216,
        33554432, 67108864, 134217728, 268435456
    ]
    # Use default values as fallback
    mpi_a100 = [
        0.00482043, 0.00633448, 0.00846133, 0.00580878, 0.00582575, 0.0320218, 
        0.0165012, 0.0415405, 0.0192714, 0.0188931, 0.0430483, 0.0413558, 
        0.0631359, 0.0608028, 0.0759302, 0.120506, 0.192064, 0.321888, 0.586894, 
        1.09782, 2.12803, 4.37734, 8.82045, 17.5709, 36.0327, 74.2748
    ]
    diomp_a100 = [
        0.196093, 0.0504654, 0.0493097, 0.0488101, 0.0487198, 0.0483061, 
        0.0474821, 0.0471084, 0.0569142, 0.0597269, 0.0605773, 0.0647126, 
        0.0742793, 0.0850786, 0.0947047, 0.125045, 0.490531, 0.5827, 0.751994, 
        1.07246, 1.68633, 2.89703, 5.29825, 4.72923, 6.07799, 8.83213
    ]




mpi_data = np.array([
    mpi_a100,
])
diomp_data = np.array([
    diomp_a100,
])
log_ratio = np.log10(mpi_data / diomp_data)
log_ratio_df = pd.DataFrame(log_ratio, index=platforms,columns=data_sizes_b)

def format_bytes(kb):
    if kb >= 1024 * 1024:
        return f"{kb // (1024 * 1024)}MB"
    elif kb >= 1024:
        return f"{kb // 1024}KB"
    else:
        return f"{kb}B"

filtered_data_sizes = data_sizes_b[12:-2]
filtered_log_ratio_df = log_ratio_df[filtered_data_sizes]
filtered_labels = [format_bytes(kb) for kb in filtered_data_sizes]
filtered_log_ratio_df.columns = filtered_labels




plt.figure(figsize=(16, 4), dpi=300)
heatmap = sns.heatmap(filtered_log_ratio_df, cmap="coolwarm_r", center=0, linewidths=0.5, linecolor='gray',
            annot=True, fmt=".2f", cbar_kws={'label': 'log(MPI / DiOMP)'},annot_kws={"size": 16},)
colorbar = heatmap.collections[0].colorbar
colorbar.set_label("log(MPI / DiOMP)", fontsize=16)
plt.xticks(rotation=0,fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(args.output, dpi=300, bbox_inches='tight')
plt.show()


