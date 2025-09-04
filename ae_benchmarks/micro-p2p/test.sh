# Please make sure you have allocated 2 Nodes and each node has at least 1 GPU.
chmod +x run.sh
srun -n 2 ./diomp_bw get > diomp_bw_get.txt
srun -n 2 ./diomp_bw put > diomp_bw_put.txt
srun -n 2 run.sh ./mpi_bw get > mpi_bw_get.txt
srun -n 2 run.sh ./mpi_bw put > mpi_bw_put.txt

# Make sure you python3 has correctly installed the packages
# pip install matplotlib numpy pandas

python3 plot_benchmark.py --diomp-get-file diomp_bw_get.txt \
                         --diomp-put-file diomp_bw_put.txt \
                         --mpi-get-file mpi_bw_get.txt \
                         --mpi-put-file mpi_bw_put.txt