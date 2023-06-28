import os
from mpi4py import MPI

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    host = os.uname()[1]
    print(f"hello from process {rank} on host {host}")


    data = (rank+1)**2
    data_collected = comm.gather(data, root=0)

    if rank == 0:
        sum = 0
        for i in range(len(data_collected)):
            sum = sum + data_collected[i]
        print(f"on rank {rank} I've calculated sum = {sum}")


    MPI.Finalize()

    # compare the result to:
    # sum([i**2 for i in range(N+1)])
    # where N is a number of MPI ranks you are using