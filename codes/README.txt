
We refer to the datasets as small files and large files below :
Small : Mediamill, Bibtex, Delicious
Large : EURLex-4K, Wiki10-31K, WikiLSHTC-325K, Delicious-200K, Amazon-670K, AmazonCat-13K

The .h files are additional files required for running of any of the codes for generating hash.

Below mentioned is the format and details of 3 codes included in the folder :

- CRAFTML_openmp.cpp : A single tree building and testing code, optimized by OpenMP threads .
Formatted to work for large datasets.
Compile : g++ -fopenmp CRAFTML_openmp.cpp
Run : ./a.out train_filename test_filename
e.g : ./a.out Eurlex/eurlex_train.txt Eurlex/eurlex_test.txt

- CRAFTML_parallel_large.cpp : A multiple tree building and testing code, parallelized by MPI and and further
optimized by OpenMP . Formatted to work for large datasets.
Compile : mpiCC -fopenmp CRAFTML_parallel_large.cpp -std=c++11
Run : mpirun -np num_procs ./a.out train_filename test_filename num_trees
e.g : mpirun -np 4 ./a.out Eurlex/eurlex_train.txt Eurlex/eurlex_test.txt 5 
desired to run 4 processes, each implementing 5 trees

-CRAFTML_parallel_small.cpp : A multiple tree building and testing code, parallelized by MPI and and further
optimized by OpenMP . Formatted to work for small datasets.
Compile : mpiCC -fopenmp CRAFTML_parallel_small.cpp -std=c++11
Run : mpirun -np num_procs ./a.out data_filename train_split test_split num_trees
e.g :  mpirun -np 5 ./a.out Bibtex/Bibtex_data.txt Bibtex/bibtex_trSplit.txt Bibtex/bibtex_tstSplit.txt 10
desired to run 5 processes, each implementing 10 trees
