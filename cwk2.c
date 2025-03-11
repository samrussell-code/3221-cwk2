//
// Starting code for the MPI coursework.
//
// See lectures and/or the worksheet corresponding to this part of the module for instructions
// on how to build and launch MPI programs. A simple makefile has also been included (usage optional).
// sudo apt-get update && sudo apt-get install -y mpich


//
// Includes.
//

// Standard includes.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// The MPI library.
#include <mpi.h>

// Some extra routines for this coursework. DO NOT MODIFY OR REPLACE THESE ROUTINES,
// as this file will be replaced with a different version for assessment.
#include "cwk2_extra.h"

float calculateMeanMPI(float *globalData, int globalSize, int rank, int numProcs) {
    int localSize = globalSize / numProcs;
    float *localData = (float *)malloc(localSize * sizeof(float));

    // global data -> all procs
    MPI_Scatter(globalData, localSize, MPI_FLOAT, localData, localSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // local sum
    float localSum = 0.0f;
    for (int i = 0; i < localSize; i++) {
        localSum += localData[i];
    }

    // reduce to global sum (rank 0 receives it)
    float globalSum = 0.0f;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // mean (rank 0 computes it)
    float mean = 0.0f;
    if (rank == 0) {
        mean = globalSum / globalSize;
    }

    // broadcast mean to all processes
    MPI_Bcast(&mean, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(localData);
    return mean;
}


void distributeMean(float mean, int rank, int numProcs) {
    int level, partner;
    // second binary tree method
    // mean sent to children
    for (level = 1; level < numProcs; level *= 2) {
        // if sender
        if (rank < level && rank + level < numProcs) {
            partner = rank + level;
            MPI_Send(&mean, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);
        }
        // if receiver
        else if (rank >= level && rank < level * 2) {
            partner = rank - level;
            MPI_Recv(&mean, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

float calculateVarianceMPI(float *globalData, int globalSize, float mean, int rank, int numProcs) {
    int localSize = globalSize / numProcs;
    float *localData = (float *)malloc(localSize * sizeof(float));

    // global data -> allprocs
    MPI_Scatter(globalData, localSize, MPI_FLOAT, localData, localSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    //  sum square dif
    float localSumSqDiff = 0.0f;
    for (int i = 0; i < localSize; i++) {
        float diff = localData[i] - mean;
        localSumSqDiff += diff * diff;
    }

    // now reduce 
    float globalSumSqDiff = 0.0f;
    MPI_Reduce(&localSumSqDiff, &globalSumSqDiff, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // rank 0 does final calculation
    float variance = 0.0f;
    if (rank == 0) {
        variance = globalSumSqDiff / globalSize;
    }

    free(localData);
    return variance;
}


//
// Main.
//
int main( int argc, char **argv )
{
    int i;

    //
    // Initialisation.
    //

    // Initialise MPI and get the rank of this process, and the total number of processes.
    int rank, numProcs;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &numProcs );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank     );

    // Check that the number of processes is a power of 2, but <=256, so the data set, which is a multiple of 256 in length,
    // is also a multiple of the number of processes. If using OpenMPI, you may need to add the argument '--oversubscribe'
    // when launching the executable, to allow more processes than you have cores.
    if( (numProcs&(numProcs-1))!=0 || numProcs>256 )
    {
        // Only display the error message from one processes, but finalise and quit all of them.
        if( rank==0 ) printf( "ERROR: Launch with a number of processes that is a power of 2 (i.e. 2, 4, 8, ...) and <=256.\n" );

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Load the full data set onto rank 0.
    float *globalData = NULL;
    int globalSize = 0;
    if( rank==0 )
    {
        globalData = readDataFromFile( &globalSize );           // globalData must be free'd on rank 0 before quitting.
        if( globalData==NULL )
        {
            MPI_Finalize();                                     // Should really communicate to all other processes that they need to quit as well ...
            return EXIT_FAILURE;
        }

        printf( "Rank 0: Read in data set with %d floats.\n", globalSize );
    }

    // Calculate the number of floats per process. Note that only rank 0 has the correct value of localSize
    // at this point in the code. This will somehow need to be communicated to all other processes. Note also
    // that we can assume that globalSize is a multiple of numProcs.
    //int localSize = globalSize / numProcs;          // = 0 at this point of the code for all processes except rank 0.

    // Start the timing now, after the data has been loaded (will only output on rank 0).
    double startTime = MPI_Wtime();


    //
    // Task 1: Calculate the mean using all available processes.
    //
    float mean = 0.0f;          // Your calculated mean should be placed in this variable.
    if (rank == 0) {
        mean = calculateMeanMPI(globalData, globalSize, rank, numProcs);
    }
    distributeMean(mean, rank, numProcs);
    


    //
    // Task 2. Calculate the variance using all processes.
    //
    float variance = 0.0f;      // Your calculated variance should be placed in this variable.
    variance = calculateVarianceMPI(globalData, globalSize, mean, rank, numProcs);


    //
    // Output the results alongside a serial check.
    //
    if( rank==0 )
    {
        // Output the results of the timing now, before moving onto other calculations.
        printf( "Total time taken: %g s\n", MPI_Wtime() - startTime );

        // Your code MUST call this function after the mean and variance have been calculated using your parallel algorithms.
        // Do not modify the function itself (which is defined in 'cwk2_extra.h'), as it will be replaced with a different
        // version for the purpose of assessing. Also, don't just put the values from serial calculations here or you will lose marks.
        finalMeanAndVariance( mean, variance );

        // Check the answers against the serial calculations. This also demonstrates how to perform the calculations
        // in serial, which you may find useful. Note that small differences in floating point calculations between
        // equivalent parallel and serial codes are possible, as explained in Lecture 11.

        // Mean.
        float sum = 0.0;
        for( i=0; i<globalSize; i++ ) sum += globalData[i];
        float mean = sum / globalSize;

        // Variance.
        float sumSqrd = 0.0;
        for( i=0; i<globalSize; i++ ) sumSqrd += ( globalData[i]-mean )*( globalData[i]-mean );
        float variance = sumSqrd / globalSize;

        printf( "SERIAL CHECK: Mean=%g and Variance=%g.\n", mean, variance );

   }

    //
    // Free all resources (including any memory you have dynamically allocated), then quit.
    //
    if( rank==0 ) free( globalData );

    MPI_Finalize();

    return EXIT_SUCCESS;
}
