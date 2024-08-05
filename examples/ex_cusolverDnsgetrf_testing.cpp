#include <stdio.h>
#include "cusolver_env.h"
#include "multi_thread_mt19937_64_rand.h"
#include <sys/time.h>

#include "lapack.h"

#include <cusolverDn.h>
#include <cublas_v2.h>

//int sgetrf_( int* M, int* N, float* h_A, int* lda, int* ipiv, int* info);


#define FMULS_GETRF(m_, n_) ( ((m_) < (n_)) \
    ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_) - 1. ) + (n_)) + (2. / 3.) * (m_)) \
    : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_) - 1. ) + (m_)) + (2. / 3.) * (n_)) )
#define FADDS_GETRF(m_, n_) ( ((m_) < (n_)) \
    ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_)      ) - (n_)) + (1. / 6.) * (m_)) \
    : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_)      ) - (m_)) + (1. / 6.) * (n_)) )

#define FLOPS_SGETRF(m_, n_) (     FMULS_GETRF((double)(m_), (double)(n_)) +       FADDS_GETRF((double)(m_), (double)(n_)) )


/***************************************************************************//**
    @return Current wall-clock time in seconds.
            Resolution is from gettimeofday.

    @ingroup solver_wtime
*******************************************************************************/
extern "C"
double sover_wtime( void )
{
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec + t.tv_usec*1e-6;
}

/***************************************************************************//**
    @return String describing CUSOLVER-OPEN errors (magma_int_t).

    @param[in]
    err     Error code.

    @ingroup cusolver_error
*******************************************************************************/
extern "C"
const char* solver_strerror( int err )
{
    // LAPACK-compliant errors
    if ( err > 0 ) {
        return "function-specific error, see documentation";
    }
    else if ( err < 0 && err > 1000 ) {
        return "invalid argument";
    }

	// solver-specific errors
    switch( err ) {
        case 0:
            return "success";

        default:
            return "unknown sover error code";
    }
}


/***************************************************************************//**
    @return Current wall-clock time in seconds.
            Resolution is from gettimeofday.

    @ingroup solver_wtime
*******************************************************************************/
extern "C"
double solver_wtime( void )
{
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec + t.tv_usec*1e-6;
}


//先写一个一摸一样的，cusolver 版本的 app
int main()
{
	cusolver_print_environment();

    double   gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    float          error;
    float *h_A;
    int     *ipiv;
    int     M, N, n2, lda, info, min_mn;
    int status = 0;

    float tol = tol = 0.000001788139343;
	//opts.tolerance * lapackf77_slamch("E");
	int check = 1;
	bool lapack = false;
	lapack = true;

	int version = 1;

    printf("%% ngpu %lld, version %lld\n", (long long) 1, (long long) version);

    if ( check == 2 ) {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    }
    else {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |PA-LU|/(N*|A|)\n");
    }
    printf("%%========================================================================\n");

	int msize[10] = { 1088, 2112, 3136, 4160, 5184, 6208, 7232, 8256, 9280, 10304};
	int nsize[10] = { 1088, 2112, 3136, 4160, 5184, 6208, 7232, 8256, 9280, 10304};

	cusolverDnHandle_t cusolverDnHan = nullptr;
	cusolverDnCreate(&cusolverDnHan);

	int bufferSize = 0;

    for( int itest = 0; itest < 10; ++itest ) {
        for( int iter = 0; iter < 1; ++iter ) {
			M = msize[itest];
			N = nsize[itest];
			min_mn = std::min(M, N);
			//printf("min_nm = %5d\n", min_mn);
			lda = M;
			n2 = lda*N;
			gflops = FLOPS_SGETRF(M, N)/1e9;
			ipiv = (int*)malloc(min_mn*sizeof(int));
			h_A = (float*)malloc(lda*N*sizeof(float*));
			if(ipiv == nullptr || h_A == nullptr){
				printf("Something error...\n");
			}
			if(lapack){
				srand_rand_float(2024, h_A, lda*N);
				cpu_time = sover_wtime();
				//lapackf77_
				sgetrf_( &M, &N, h_A, &lda, ipiv, &info );
				cpu_time = sover_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_sgetrf returned error %lld: %s.\n",
                           (long long) info, solver_strerror( info ));
                }
			}
            /* ====================================================================
               Performs operation using cusolver-open
               =================================================================== */
			srand_rand_float(2024, h_A, lda*N);
			if (version == 2) {
                // no pivoting versions, so set ipiv to identity
                for (int i=0; i < min_mn; ++i ) {
                    ipiv[i] = i+1;
                }
            }
			float *A_d = nullptr;
			cudaMalloc((void**)&A_d, lda*N*sizeof(float));
			cudaMemcpy(A_d, h_A, lda*N*sizeof(float), cudaMemcpyHostToDevice);
/*
cusolverStatus_t
cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle,
							int m,int n,float *A,int lda,int *Lwork );
*/
/*
cusolverStatus_t
cusolverDnSgetrf(cusolverDnHandle_t handle,
				int m,int n,float *A,int lda,float *Workspace,int *devIpiv,int *devInfo );
*/

			cusolverDnSgetrf_bufferSize(cusolverDnHan, M, N, A_d, lda, &bufferSize);

			float *Workspace = nullptr;
			int *info_d = nullptr;
			int *ipiv_d = nullptr;

			cudaMalloc((void**)&Workspace, bufferSize*sizeof(float));
			cudaMalloc((void**)info_d, sizeof(int));
			cudaMalloc((void**)ipiv_d, min_mn*sizeof(int));

			gpu_time = sover_wtime();
            if (version == 1) {
                cusolverDnSgetrf(cusolverDnHan, M, N, h_A, lda, Workspace, ipiv_d, info_d);
            }
            else if (version == 2) {
                cusolverDnSgetrf(cusolverDnHan, M, N, h_A, lda, Workspace, nullptr, info_d);
            }
            gpu_time = sover_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("cusolverDnSgetrf returned error %lld: %s.\n",
                       (long long) info, solver_strerror( info ));
            }

			cudaMemcpy(h_A, A_d, lda*N*sizeof(float), cudaMemcpyDeviceToHost);
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if(lapack) {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) M, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) M, (long long) N, gpu_perf, gpu_time );
            }

			if ( check ) {
//                error = get_LU_error(opts, M, N, h_A, lda, ipiv);
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf("     ---   \n");
            }



			if(A_d != nullptr){
				cudaFree(A_d);
				A_d = nullptr;
			}
			if(Workspace != nullptr){
				cudaFree(Workspace);
				Workspace = nullptr;
			}
			if(info_d != nullptr){
				cudaFree(info_d);
				info_d = nullptr;
			}

			free(ipiv);	ipiv = nullptr;
			free(h_A);	h_A = nullptr;
		}
	}
















	return 0;
}
