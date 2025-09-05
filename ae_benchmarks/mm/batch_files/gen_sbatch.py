import os

output_dir = os.path.join(os.getcwd(), "./")
os.makedirs(output_dir, exist_ok=True)

nodes_range = range(1, 8)

sbatch_diomp_template = """#!/bin/bash
#SBATCH -A [ACCOUNT]
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=0:5:00
#SBATCH --nodes={nodes}
#SBATCH -G {gpus}
#SBATCH --job-name=diomp_mm_{nodes}
#SBATCH --output=diomp_mm_{nodes}.log 
#SBATCH --error=diomp_mm_{nodes}.err     
#SBATCH --mail-type=begin,end,fail

module load nccl


echo "DiOMP MM"
srun -n {gpus} ../run.sh ../diomp_mm
"""

sbatch_mpi_template = """#!/bin/bash
#SBATCH -A [ACCOUNT]
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=0:5:00
#SBATCH --nodes={nodes}
#SBATCH -G {gpus}
#SBATCH --job-name=diomp_mm_{nodes}
#SBATCH --output=diomp_mm_{nodes}.log 
#SBATCH --error=diomp_mm_{nodes}.err     
#SBATCH --mail-type=begin,end,fail

module load nccl

echo "MPI MM"
srun -n {gpus} ../run.sh ../mpi_mm
"""





for nodes in nodes_range:
    gpus = nodes*4
    
    script_content = sbatch_diomp_template.format(nodes=nodes, gpus=gpus)
    script_filename = os.path.join(output_dir, f"diomp_n{nodes}_g{gpus}.sh")
    with open(script_filename, "w") as script_file:
        script_file.write(script_content)

    script_content = sbatch_mpi_template.format(nodes=nodes, gpus=gpus)
    script_filename = os.path.join(output_dir, f"mpi_n{nodes}_g{gpus}.sh")
    with open(script_filename, "w") as script_file:
        script_file.write(script_content)

    print(f"Generated sbatch scripts for {nodes} nodes and {gpus} GPUs")