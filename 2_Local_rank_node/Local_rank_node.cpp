#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size, local_rank, local_size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Get_processor_name(hostname, &name_len);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    printf("Global Rank: %d, Local Rank: %d, Node: %s\n", world_rank, local_rank, hostname);

    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}

