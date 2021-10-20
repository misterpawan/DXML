module load openmpi/4.0.1
mpiCC -fopenmp DXML_parallel_large.cpp -std=c++11 &&
mpirun -np 4 ./a.out /home/iiit/pawan/data/Eurlex/eurlex_train.txt  /home/iiit/pawan/data/Eurlex/eurlex_test.txt 5
