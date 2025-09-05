# DiOMP-Offloading AE for PAW-ATM

## Hardware Requirements

At least 8 Nodes with 4 NVIDIA/AMD GPUs.

## Software Requirements

MPI installed.

## Install

Installation scripts are located in the install_scripts folder. For NERSC Perlmutter and LC Tioga, you can install with one click.
For other platforms, please adjust the corresponding library paths according to your needs.
After installation, please source according to the prompts to configure environment variables.


## Micro-benchmark P2P

P2P micro-benchmarks are located in the `micro-p2p` folder. Please use Makefile for compilation.
For Perlmutter, use
`make all` for compilation.
For Tioga, use
`make -f Makefile.tioga` for compilation.
Please adjust the Makefile configuration according to the actual installation path before compilation.

This section should be run using exactly 2 nodes with 2 GPUs.
`test_perlmutter.sh` provides an example run script.

Please note that we are aware that the latest version of Cray-MPI has improved performance for very large data volumes (64-128MB) compared to the version in the paper.
Therefore, the Artifact goal should be that DiOMP outperforms MPI in most cases. The only possible exception is DiOMP Put on Perlmutter, which is a performance issue from HPE driver problems and is not within the scope of this Artifact discussion.


## Micro-benchmark Collective

Collective micro-benchmarks are located in the `micro-collective` folder. Please use Makefile for compilation.
Please adjust the Makefile configuration according to the actual installation path before compilation.
For Perlmutter, use
`make all` for compilation.
For Tioga, use
`make -f Makefile.tioga` for compilation.
Please adjust the Makefile configuration according to the actual installation path before compilation.

This section should be run using exactly 4 nodes with 16/32 GPUs.

Please note that we are aware that the latest version of Cray-MPI has improved performance compared to the version in the paper.
Therefore, the Artifact goal should be that DiOMP outperforms MPI for larger data volumes on NV platforms; no performance goals are set for AMD platforms.

## Matrix Multiplication

Matrix multiplication tests are located in the `mm` folder. Please use Makefile for compilation.
Please adjust the Makefile configuration according to the actual installation path before compilation.
For Perlmutter, use
`make all` for compilation.
For Tioga, use
`make -f Makefile.tioga` for compilation.
Please adjust the Makefile configuration according to the actual installation path before compilation.


This section should be run using 1 to 8 nodes.
**Please note that when running on Perlmutter, ensure to use the run.sh driver program.**
`run.sh` will handle device and rank binding.
For example: `srun -n 16 ./run.sh ./mpi_mm`

For Perlmutter, we also provide sbatch generation scripts for batch generation of sbatch scripts. Please adjust the Account as needed. This can also be used for other slurm platforms.
Before running, please execute `chmod +x run.sh`

The Artifact goal is that DiOMP should outperform the MPI implementation.

## Plotting
Python scripts for plotting are located in the `plot_scripts` folder.

The command to plot P2P results is:
```
python3 micro_plot.py --diomp-get-file diomp_bw_get.txt \
                      --diomp-put-file diomp_bw_put.txt \
                      --mpi-get-file mpi_bw_get.txt \
                      --mpi-put-file mpi_bw_put.txt
```
This should generate latency and bandwidth curves. For latency curves, lower is better; for bandwidth curves, higher is better.

The command to plot Collective results is:
```
python3 collective.py --mpi-csv mpi_output.csv \
                      --diomp-csv diomp_output.csv
```

For the color matrix, darker blue indicates better DiOMP performance, while lighter colors indicate worse performance. Please note that this visualization is not colorblind-friendly.


To plot MM curves, please manually input the execution times in `mm_plot.py`.

## Contact

If you encounter any issues, please contact baodi.shan@stonybrook.edu (PAW ATM is not double-blind review).


