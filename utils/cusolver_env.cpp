#include "cusolver_env.h"

void check_error_in(cudaError_t err)
{
	if(err != cudaSuccess)
		printf("Something error ...\n");
}

/***************************************************************************//**
    Print cuSOLVER version, CUDA version, LAPACK/BLAS library version,
    available GPU devices, number of threads, date, etc.
    Used in testing.
    @ingroup magma_testing
*******************************************************************************/
void
cusolver_print_environment()
{
	int major=-1,minor=-1,patch=-1;
	cusolverGetProperty(MAJOR_VERSION, &major);
	cusolverGetProperty(MINOR_VERSION, &minor);
	cusolverGetProperty(PATCH_LEVEL, &patch);
	printf("CUSOLVER Version (Major,Minor,PatchLevel): %d.%d.%d, %lld-bit int, %lld-bit pointer. \n", 
		major, minor, patch, 
		(long long) (8*sizeof(int)),
		(long long) (8*sizeof(void*)));


//LL::    printf( "%% Compiled for CUDA architectures %s\n", IX_CUDA_ARCH );

    // CUDA, OpenCL, OpenMP, openBLAS versions all printed on same line
    int cuda_runtime=0, cuda_driver=0;
    cudaError_t err;
    err = cudaDriverGetVersion( &cuda_driver );
    check_error_in( err );
    err = cudaRuntimeGetVersion( &cuda_runtime );
    if ( err != cudaErrorNoDevice ) {
        check_error_in( err );
    }
    printf( "%% CUDA runtime %d, driver %d. ", cuda_runtime, cuda_driver );

/* OpenMP */

#if defined(_OPENMP)
    int omp_threads = 0;
    #pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
    }
    printf( "\nOpenMP threads %d. ", omp_threads );
#endif

    printf( "\n" );

    // print devices
    int ndevices = 0;
    err = cudaGetDeviceCount( &ndevices );
    if ( err != cudaErrorNoDevice ) {
        check_error_in( err );
    }

    for( int dev = 0; dev < ndevices; ++dev ) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties( &prop, dev );
        check_error_in( err );

        printf( "%% device %d: %s, %.1f MHz clock, %.1f MiB memory, capability %d.%d\n",
                dev,
                prop.name,
                prop.clockRate / 1000.,
                prop.totalGlobalMem / (1024.*1024.),
                prop.major,
                prop.minor );
    }

    //MAG_UNUSED( err );
    time_t t = time( NULL );
    printf( "%% %s", ctime( &t ));
}


//#define MAIN_

#ifdef MAIN_
int main()
{
        cusolver_print_environment();
        return 0;
}

void check_error(cudaError_t err)
{
        if(err != cudaSuccess)
                printf("Something error ...\n");
}

#endif



