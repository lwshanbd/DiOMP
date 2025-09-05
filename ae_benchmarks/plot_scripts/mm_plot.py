import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
# Number of threads
threads = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

# Execution times in microseconds
mpi_times = [6.89097e+07, 2.99351e+07, 1.399e+07, 7.25277e+06, 4.96014e+06, 
             4.8048e+06, 3.75112e+06, 3.82896e+06, 3.15375e+06, 3.11869e+06]
diomp_times = [6.88294e+07, 2.97954e+07, 1.36706e+07, 7.09102e+06, 4.73437e+06, 
               4.28464e+06, 3.55002e+06, 3.60411e+06, 2.92718e+06, 2.89199e+06]

# Calculate speedup (base time / time for each thread count)
mpi_speedup = [mpi_times[0] / t for t in mpi_times]
diomp_speedup = [diomp_times[0] / t for t in diomp_times]
tmp = [i/4 for i in threads]
tmp2 = [i**2 for i in range(1,11)]
# Plotting the speedup
plt.figure(figsize=(10, 6),dpi=300)

plt.plot(threads, diomp_speedup, marker='o', label='DiOMP',linewidth=2.5, markersize=16, color="#a0c4ff")
plt.plot(threads, mpi_speedup, marker='^', label='MPI', linewidth=2.5, markersize=16, color="#ffadad")


plt.xlabel("Number of GPUs", fontsize=22)
plt.ylabel("Speedup", fontsize=22)
plt.xticks(threads, fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=22)
plt.tight_layout()
plt.savefig('mm_plot.png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
