
#include "cusolver_env.h"
#include "multi_thread_mt19937_64_rand.h"
#include <stdio.h>
#include <sys/time.h>

//#include "lapack.h"
//#include "cblas.h"
//#include "f77blas.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

/*
 * 本文的整型规则，除了例外，其余部分的整型都用 signed long int, 包括 info_lapack, piv_lapack
 * 例外：info_d, info_h，piv_d, piv_h, 这都是给
 * cuda 10.2 使用的；
 */

#define SEED_A (2024)

extern "C" {
signed long int sgetrf_(signed long int *M, signed long int *N, float *h_A,
                        signed long int *lda, signed long int *ipiv,
                        signed long int *info);
float slange_(char *norm, signed long int *M, signed long int *N, float *A,
              signed long int *lda, float *work);
// REAL             FUNCTION SLANGE( NORM, M, N, A, LDA, WORK )
void slaswp_(signed long int *N, float *A, signed long int *lda,
             signed long int *k1, signed long int *k2, int *ipiv,
             signed long int *incx);
// SUBROUTINE SLASWP( N, A, LDA, K1, K2, IPIV, INCX )
void slacpy_(char *UPLO, signed long int *M, signed long int *N, float *A,
              signed long int *lda, float *B, signed long int *ldb);
// SUBROUTINE SLACPY( UPLO, M, N, A, LDA, B, LDB )
void sgemm_(char *, char *, signed long int *, signed long int *,
            signed long int *, float *, float *, signed long int *, float *,
            signed long int *, float *, float *, signed long int *);
}

#define FMULS_GETRF(m_, n_)                                                    \
    (((m_) < (n_))                                                             \
         ? (0.5 * (m_) * ((m_) * ((n_) - (1. / 3.) * (m_)-1.) + (n_)) +        \
            (2. / 3.) * (m_))                                                  \
         : (0.5 * (n_) * ((n_) * ((m_) - (1. / 3.) * (n_)-1.) + (m_)) +        \
            (2. / 3.) * (n_)))
#define FADDS_GETRF(m_, n_)                                                    \
    (((m_) < (n_)) ? (0.5 * (m_) * ((m_) * ((n_) - (1. / 3.) * (m_)) - (n_)) + \
                      (1. / 6.) * (m_))                                        \
                   : (0.5 * (n_) * ((n_) * ((m_) - (1. / 3.) * (n_)) - (m_)) + \
                      (1. / 6.) * (n_)))

#define FLOPS_SGETRF(m_, n_)                                                   \
    (FMULS_GETRF((double)(m_), (double)(n_)) +                                 \
     FADDS_GETRF((double)(m_), (double)(n_)))

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

/***************************************************************************/ /**
 // On input, LU and ipiv is LU factorization of A. On output, LU is
 overwritten.
 // Works for any m, n.
 // Uses init_matrix() to re-generate original A as needed.
 // Returns error in factorization, |PA - LU| / (n |A|)
 // This allocates 3 more matrices to store A, L, and U.
 *******************************************************************************/

float get_LU_error(signed long int M, signed long int N, float *LU,
                   signed long int lda, int *ipiv) {
    signed long int min_mn = std::min(M, N);
    signed long int ione = 1;
    signed long int i, j;
    float alpha = 1.0f;
    float beta = 0.0f;
    float *A, *L, *U;
    float work[1], matnorm, residual;

    A = (float *)malloc(lda * N * sizeof(float));
    L = (float *)malloc(M * min_mn * sizeof(float));
    U = (float *)malloc(min_mn * N * sizeof(float));
    memset(L, 0, M * min_mn * sizeof(float));
    memset(U, 0, min_mn * N * sizeof(float));

    // set the same original matrix A
    srand_rand_float(SEED_A, A, lda * N);
    slaswp_(&N, A, &lda, &ione, &min_mn, ipiv, &ione);

    // copy LU to L and U, and set diagonal to 1// start
    slacpy_("L", &M, &min_mn, LU, &lda, L, &M);
    slacpy_("U", &min_mn, &N, LU, &lda, U, &min_mn);
    for (j = 0; j < min_mn; j++)
        L[j + j * M] = 1.0f;

    matnorm = slange_("F", &M, &N, A, &lda, work);

    //	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N,
    // min_mn,
    // alpha, L, M, U, min_mn, beta, LU, min_mn);
    sgemm_("N", "N", &M, &N, &min_mn, &alpha, L, &M, U, &min_mn, &beta, LU,
           &min_mn);
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            LU[i + j * lda] = LU[i + j * lda] - A[i + j * lda];
        }
    }

    residual = slange_("F", &M, &N, LU, &lda, work);

    if (A != nullptr) {
        free(A);
        A = nullptr;
    }
    if (L != nullptr) {
        free(L);
        L = nullptr;
    }
    if (U != nullptr) {
        free(U);
        U = nullptr;
    }

    return residual / (matnorm * N);
}

void trans_i32_2_i64(signed long int *piv_h64, int *piv_h, signed long int min_mn){
	for(long int i = 0; i< min_mn; i++){
		piv_h64[i] = piv_h[i];
	}
}

#define EL 10

int main() {
    cusolver_print_environment();
    cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cuda_err = cudaSuccess;

    double gflops, gpu_perf, gpu_time, cpu_perf = 0, cpu_time = 0;
    float error;
    float *A_h;
    int *piv_d = nullptr;
    int *piv_h = nullptr;
	long int *piv_h64 = nullptr;

    signed long int *piv_lapack = nullptr;
    signed long int M, N, n2, lda, info, min_mn;
    signed long int status = 0;

    float tol = tol = 0.000001788139343;
    // opts.tolerance * lapackf77_slamch("E");
    signed long int check = 1;
    bool lapack = false;
    lapack = true;

    signed long int version = 1;

    printf("%% ngpu %lld, version %lld\n", (long long)1, (long long)version);

    if (check == 2) {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   "
               "|Ax-b|/(N*|A|*|x|)\n");
    } else {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   "
               "|PA-LU|/(N*|A|)\n");
    }
    printf("%%================================================================="
           "=======\n");

#if 1
    signed long int msize[EL] = {1088, 2112, 3136, 4160, 5184,
                                 6208, 7232, 8256, 9280, 10304};
    signed long int nsize[EL] = {1088, 2112, 3136, 4160, 5184,
                                 6208, 7232, 8256, 9280, 10304};
#else
    //signed long int msize[EL] = {103040};
    //signed long int nsize[EL] = {103040};
     int msize[10] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
     int nsize[10] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
#endif

    cusolverDnHandle_t cusolverDnHan = nullptr;
    cusolverDnCreate(&cusolverDnHan);

    int bufferSize = 0;

    for (signed long int itest = 0; itest < EL; ++itest) {
        for (signed long int iter = 0; iter < 1; ++iter) {
            M = msize[itest];
            N = nsize[itest];
            min_mn = std::min(M, N);
            lda = M;
            n2 = lda * N;
            gflops = FLOPS_SGETRF(M, N) / 1e9;
            piv_h = (int *)malloc(min_mn * sizeof(int));
			piv_h64 = (long int*)malloc(min_mn * sizeof(long int));

            A_h = (float *)malloc(lda * N * sizeof(float *));
            if (piv_h == nullptr || A_h == nullptr || piv_h64 == nullptr) {
                printf("Something error...\n");
				return -1;
            }
            if (lapack) {
                srand_rand_float(SEED_A, A_h, lda * N);
                cpu_time = solver_wtime();
                // lapackf77_
                sgetrf_(&M, &N, A_h, &lda, piv_h64, &info);
                cpu_time = solver_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_sgetrf returned "
                           "error %lld: %s.\n",
                           (long long)info, solver_strerror(info));
                }


				//printf("\npiv_lapack = \n");				print_int_vector<int>(117, (int*)(piv_h64), 0);				printf("\n");

            }
            /* ====================================================================
               Performs operation using cusolver-open
               ===================================================================*/
            srand_rand_float(SEED_A, A_h, lda * N);
            if (version == 2) {
                // no pivoting versions, so set ipiv to identity
                for (signed long int i = 0; i < min_mn; ++i) {
                    piv_h64[i] = i + 1;
                }
            }
            float *A_d = nullptr;
            cuda_err = cudaMalloc((void **)&A_d, lda * N * sizeof(float));		if(cuda_err != cudaSuccess){printf("001err = %d", (int)cuda_err);}


            cuda_err = cudaMemcpy(A_d, A_h, lda * N * sizeof(float),
                       cudaMemcpyHostToDevice);		if(cuda_err != cudaSuccess){printf("002err = %d", (int)cuda_err);}

            solver_status = cusolverDnSgetrf_bufferSize(cusolverDnHan, M, N,
                                                        A_d, lda, &bufferSize);
            if(solver_status != 0)	printf("solver_status = %d\n", (signed long int)solver_status);
            float *Workspace = nullptr;
            int *info_d = nullptr;
            int *info_h = nullptr;

            cuda_err = cudaMalloc((void **)&Workspace, bufferSize * sizeof(float));if(cuda_err != cudaSuccess){printf("003err = %d", (int)cuda_err);}
            cuda_err = cudaMalloc((void **)&info_d, sizeof(int));if(cuda_err != cudaSuccess){printf("004err = %d", (int)cuda_err);}
            cuda_err = cudaMalloc((void **)&piv_d, min_mn * sizeof(int));if(cuda_err != cudaSuccess){printf("005err = %d", (int)cuda_err);}
            info_h = (int *)malloc(1*sizeof(int));

            gpu_time = solver_wtime();
            if (version == 1) {
                solver_status = cusolverDnSgetrf(cusolverDnHan, M, N, A_d, lda,
                                                 Workspace, piv_d, info_d);
                printf("solver_status = %d\n", (signed long int)solver_status);
            } else if (version == 2) {
                cusolverDnSgetrf(cusolverDnHan, M, N, A_d, lda, Workspace,
                                 nullptr, info_d);
            }
            cuda_err = cudaDeviceSynchronize();if(cuda_err != cudaSuccess){printf("e006rr = %d", (int)cuda_err);}
            gpu_time = solver_wtime() - gpu_time;
            //printf("gflops = %f, gpu_time = %f\n", gflops, gpu_time);
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("cusolverDnSgetrf returned error %lld: "
                       "%s.\n",
                       (long long)info, solver_strerror(info));
            }

            cuda_err = cudaMemcpy(A_h, A_d, lda * N * sizeof(float), cudaMemcpyDeviceToHost); 		if(cuda_err != cudaSuccess){printf("007err = %d", (int)cuda_err);}
            cuda_err = cudaMemcpy(piv_h, piv_d, min_mn * sizeof(float), cudaMemcpyDeviceToHost); 	if(cuda_err != cudaSuccess){printf("008err = %d", (int)cuda_err);}
            cuda_err = cudaMemcpy(info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);				if(cuda_err != cudaSuccess){printf("009err = %d", (int)cuda_err);}
            cuda_err = cudaDeviceSynchronize();														if(cuda_err != cudaSuccess){printf("010err = %d", (int)cuda_err);}
			trans_i32_2_i64(piv_h64, piv_h, min_mn);

            //printf("piv_h64 = \n");            print_int_vector<long int>(117, piv_h64, 0);
            printf("\ninfo_h = %d\n", info_h);            printf("\n");

            /* =====================================================================
               Check the factorization
               ===================================================================
             */
            if (lapack) {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f "
                       "(%7.2f)",
                       (long long)M, (long long)N, cpu_perf, cpu_time, gpu_perf,
                       gpu_time);
            } else {
                printf("%5lld %5lld     ---   (  ---  )   "
                       "%7.2f (%7.2f)",
                       (long long)M, (long long)N, gpu_perf, gpu_time);
            }

            if (check) {
                error = get_LU_error(M, N, A_h, lda, piv_h);
                printf("   %8.2e   %s\n", error,
                       (error < tol ? "ok" : "failed"));
                status += !(error < tol);
            } else {
                printf("     ---   \n");
            }

            if (A_d != nullptr) {
                cudaFree(A_d);
                A_d = nullptr;
            }
            if (Workspace != nullptr) {
                cudaFree(Workspace);
                Workspace = nullptr;
            }
			if (piv_d != nullptr) {
                cudaFree(piv_d);
                piv_d = nullptr;
            }
            if (info_d != nullptr) {
                cudaFree(info_d);
                info_d = nullptr;
            }

            if (piv_h != nullptr) {
                free(piv_h);
                piv_h = nullptr;
            }
			if (piv_h64 != nullptr) {
                free(piv_h64);
                piv_h64 = nullptr;
            }
            if (A_h != nullptr) {
                free(A_h);
                A_h = nullptr;
            }
            if (info_h != nullptr) {
                free(info_h);
                info_h = nullptr;
            }
			cuda_err = cudaDeviceSynchronize();if(cuda_err != cudaSuccess){printf("011err = %d", (int)cuda_err);}
        }
    }

    return 0;
}
