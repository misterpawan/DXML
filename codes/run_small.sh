#module add openmpi/4.0.1
module load hpcx-2.7.0/hpcx
mpiCC -fopenmp DXTML_parallel_small.cpp -std=c++11

# Run the code
#mpirun -np 10 ./a.out ~/data/Bibtex/Bibtex_data.txt ~/data/Bibtex/bibtex_trSplit.txt ~/data/Bibtex/bibtex_tstSplit.txt 10
mpirun -n 2 -mca  coll_fca_enable 1 ./a.out /scratch/pawan/data/Delicious/Delicious_data.txt   /scratch/pawan/data/Delicious/delicious_trSplit.txt  /scratch/pawan/data/Delicious/delicious_tstSplit.txt 4
