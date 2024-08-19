#include "cusolver_env.h"
#include "multi_thread_mt19937_64_rand.h"
#include <stdio.h>
#include <sys/time.h>

//#include "lapack.h"
//#include "cblas.h"
//#include "f77blas.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

#define SEED_A (2024)


#define FMULS_GEQRF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * (  0.5-(1./3.) * (n_) + (m_)) +    (m_) + 23. / 6.)) \
    : ((m_) * ((m_) * ( -0.5-(1./3.) * (m_) + (n_)) + 2.*(n_) + 23. / 6.)) )
#define FADDS_GEQRF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * (  0.5-(1./3.) * (n_) + (m_))           +  5. / 6.)) \
    : ((m_) * ((m_) * ( -0.5-(1./3.) * (m_) + (n_)) +    (n_) +  5. / 6.)) )

#define FLOPS_SGEQRF(m_, n_) (     FMULS_GEQRF((double)(m_), (double)(n_)) +       FADDS_GEQRF((double)(m_), (double)(n_)) )


extern "C" {
long int sgetrf_(long int *M, long int *N, float *h_A,
                        long int *lda, long int *ipiv,
                        long int *info);
float slange_(char *norm, long int *M, long int *N, float *A,
              long int *lda, float *work);
// REAL             FUNCTION SLANGE( NORM, M, N, A, LDA, WORK )
void slaswp_(long int *N, float *A, long int *lda,
             long int *k1, long int *k2, int *ipiv,
             long int *incx);
// SUBROUTINE SLASWP( N, A, LDA, K1, K2, IPIV, INCX )
void slacpy_(char *UPLO, long int *M, long int *N, float *A,
              long int *lda, float *B, long int *ldb);
// SUBROUTINE SLACPY( UPLO, M, N, A, LDA, B, LDB )
void sgemm_(char *, char *, long int *, long int *,
            long int *, float *, float *, long int *, float *,
            long int *, float *, float *, long int *);
//////////////////////////////////////////////////////////////////////////////////////////


//sorgqr    //lapack
void sorgqr_(
    long int const* m, long int const* n, long int const* k,
    float* A, long int const* lda,
    float const* tau,
    float* work, long int const* lwork,
    long int* info );
//slaset    //lapack
void slaset_(
    char const* uplo,
    long int const* m, long int const* n,
    float const* alpha,
    float const* beta,
    float* A, long int const* lda
#ifdef LAPACK_FORTRAN_STRLEN_END
    , size_t
#endif
);


typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
//ssyrk     //blas
void ssyrk_(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const long int N, const long int K,
                 const float alpha, const float *A, const long int lda,
                 const float beta, float *C, const long int ldc);

//safe_lapackf77_slansy    //blas----self
float slansy_(
    char const* norm, char const* uplo,
    long int const* n,
    float const* A, long int const* lda,
    float* work
#ifdef LAPACK_FORTRAN_STRLEN_END
    , size_t, size_t
#endif
);

//sgeqrf    //lapack
void sgeqrf_(long int const* m, long int const* n, float* A, long int const* lda,
             float* tau, float* work, long int const* lwork, long int* info );
//

}

/***************************************************************************/ /**
     @return Current wall-clock time in seconds.
             Resolution is from gettimeofday.

     @ingroup solver_wtime
 *******************************************************************************/
extern "C" double solver_wtime(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

/***************************************************************************/ /**
     @return String describing CUSOLVER-OPEN errors (magma_int_t).

     @param[in]
     err     Error code.

     @ingroup cusolver_error
 *******************************************************************************/

extern "C" const char *solver_strerror(signed long int err) {
    // LAPACK-compliant errors
    if (err > 0) {
        return "function-specific error, see documentation";
    } else if (err < 0 && err > 1000) {
        return "invalid argument";
    }

    // solver-specific errors
    switch (err) {
    case 0:
        return "success";

    default:
        return "unknown sover error code";
    }
}

/***************************************************************************/ /*
 print_matrix(int M, int N, float *A, int lda);
 **************************************************************************/

void print_matrix(signed long int M, signed long int N, float *A,
                  signed long int lda) {
    for (signed long int i = 0; i < M; i++) {
        for (signed long int j; j < N; j++) {
            printf("%7.4 ", A[i + j * lda]);
        }
        printf("\n");
    }
}

template<class T>
void print_int_vector(long int N, T *A, long int offset) {
    for (long int i = 0; i < N; i++)
        printf(" %ld", (long int) (A[i+offset]));

	printf("\n");
}



#define EL 10
typedef int info_int;
int main() {
    cusolver_print_environment();
    cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cuda_err = cudaSuccess;

    const float             d_neg_one = -1.0;
    const float             d_one     = 1.0;
    const float c_neg_one = -1.0;
    const float c_one     = 1.0;
    const float c_zero    = 0.0;

    double    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    float           Anorm, error=0, error2=0;
    float *h_A, *h_R, *tau, *h_work, tmp[1], unused[1];
    long int M, N, n2, lda, lwork, info, min_mn, nb;

    int status = 0;
    float tol = 0.00000178814;

#if 1
    signed long int msize[EL] = {1088, 2112, 3136, 4160, 5184,
                                 6208, 7232, 8256, 9280, 10304};
    signed long int nsize[EL] = {1088, 2112, 3136, 4160, 5184,
                                 6208, 7232, 8256, 9280, 10304};
#else
    //signed long int msize[EL] = {103040};
    //signed long int nsize[EL] = {103040};
     int msize[10] = {33, 64, 125, 236, 337, 458, 569, 610, 711, 1112};
     int nsize[10] = {33, 64, 125, 236, 337, 458, 569, 610, 711, 1112};
     //int nsize[10] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
#endif

    cusolverDnHandle_t cusolverDnHan = nullptr;
    cusolverDnCreate(&cusolverDnHan);

    int bufferSize = 0;
    int lapack = 1;
    int ngpu = 1;

    float* A_h = nullptr;
    float* A_d = nullptr;



    printf("%% ngpu %lld\n", (long long) ngpu);
    printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |R - Q^H*A|   |I - Q^H*Q|\n");
    printf("%%==============================================================================\n");

    for (signed long int itest = 0; itest < EL; ++itest) {
        for (signed long int iter = 0; iter < 1; ++iter) {
            M = msize[itest];
            N = nsize[itest];
            min_mn = std::min(M, N);
            lda = M;
            n2 = lda * N;
            A_h = (float*)malloc(lda*N*sizeof(float));

            gflops = FLOPS_SGEQRF( M, N ) / 1e9;
            printf("gflops = %f\n", gflops);
            float* tau_h = nullptr;
            float* tau_d = nullptr;
            info_int * info_d = nullptr;
            info_int * info_h = nullptr;

            cuda_err = cudaMalloc((void**)&info_d, 1*sizeof(info_int)); if(info_d == nullptr || cuda_err != cudaSuccess){printf("cuda_err = %d\n", (int*)cuda_err);}
            info_h = (int*)malloc(1*sizeof(info_int));
            //*info_h = 0;
            //cuda_err = cudaMemcpy(info_d, info_h, 1*sizeof(info_int), cudaMemcpyHostToDevice);if(cuda_err != cudaSuccess){printf("cuda_err = %d", (int)cuda_err);}
            if (lapack) {
                lwork = -1;
//void sgeqrf_(long int const* m, long int const* n, float* A, long int const* lda, float* tau, float* work, long int const* lwork, long int* info );
                sgeqrf_(&M, &N, unused, &lda, unused, tmp, &lwork, &info);
                lwork = tmp[0];
                //printf("LL:: lwork = %d\n", lwork = tmp[0]);
                float* work = nullptr;

                work = (float*)malloc(lda*N*sizeof(float));
                tau_h = (float*)malloc(min_mn*sizeof(float));
                srand_rand_float(2222, A_h, lda*N);
                sgeqrf_(&M, &N, A_h, &lda, tau_h, work, &lwork, &info);


                if(work != nullptr){free(work); work = nullptr;}
            }
            cuda_err = cudaMalloc((void**)&A_d, lda*N*sizeof(float));  if(cuda_err != cudaSuccess){printf("cuda_err = %d", (int)cuda_err);}
            solver_status = cusolverDnSgeqrf_bufferSize(cusolverDnHan, M, N, A_d, lda, &bufferSize);
            printf("bufferSize = %d\n", bufferSize);

            float* Workspace = nullptr;
            cuda_err = cudaMalloc((void**)&Workspace, bufferSize*sizeof(float)); if(cuda_err != cudaSuccess){printf("cuda_err = %d", (int)cuda_err);}
            cuda_err = cudaMalloc((void**)&tau_d, min_mn*sizeof(float));
            srand_rand_float(2222, A_h, lda*N);
            cuda_err = cudaMemcpy(A_d, A_h, lda*N*sizeof(float), cudaMemcpyHostToDevice);if(cuda_err != cudaSuccess){printf("cuda_err = %d", (int)cuda_err);}

            gpu_time = solver_wtime();

            solver_status = cusolverDnSgeqrf(cusolverDnHan, M, N, A_d, lda, tau_d, Workspace, bufferSize, info_d);
            if(solver_status != 0)	printf("solver_status = %d\n", (signed long int)solver_status);

            gpu_time = solver_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            //fetch info_d
            cuda_err = cudaMemcpy(info_h, info_d, 1*sizeof(info_int), cudaMemcpyDeviceToHost);if(cuda_err != cudaSuccess){printf("cuda_err = %d", (int)cuda_err);}

            if (*info_h != 0) {
                printf("cusolverDnSgetrf returned error %lld: "
                       "%s.\n",
                       (long long)*info_h, solver_strerror(info));
            }





            //solver_status = cusolverDnSgeqrf();
            if(A_d != nullptr){cudaFree(A_d);   A_d = nullptr;}
            if(Workspace != nullptr){cudaFree(Workspace);  Workspace = nullptr;}
            if(tau_d != nullptr){cudaFree(tau_d);   tau_d = nullptr;}
            if(A_h != nullptr){free(A_h); A_h = nullptr;}
            if(tau_h != nullptr){free(tau_h); tau_h = nullptr;}
            //if( != nullptr){free(); = nullptr;}
        }
    }



    cusolverDnDestroy(cusolverDnHan);
    return 0;
}



