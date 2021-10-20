#!/bin/bash
#SBATCH --job-name=parallel_job               	# Job name
#SBATCH --mail-type=END,FAIL         		# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodes=1                    		# Number of nodes	
#SBATCH --ntasks=1                   		# Total Number of processes on reserved nodes		
#SBATCH --cpus-per-task=20            		# Number of CPU cores per task
#SBATCH --mem=1gb                    		# Job memory request
#SBATCH --time=00:25:00              		# Time limit hrs:min:sec
#SBATCH --output=parallel_%j.log     		# Standard output and error log

pwd; hostname; date

echo "Running CRAFTML program on $SLURM_CPUS_ON_NODE CPU cores"

module add openmpi/4.0.1

#srun -n 5 ./mpi_hello_world

srun ./a.out /home/iiit/pawan/work/Eurlex/eurlex_train.txt  /home/iiit/pawan/work/Eurlex/eurlex_test.txt 2
#mpirun -np 20 ./a.out ./Eurlex/eurlex_train.txt  ./Eurlex/eurlex_test.txt 2
