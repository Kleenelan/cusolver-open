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

#define FMULS_GEQRF(m_, n_)                                                    \
    (((m_) > (n_))                                                             \
         ? ((n_) * ((n_) * (0.5 - (1. / 3.) * (n_) + (m_)) + (m_) + 23. / 6.)) \
         : ((m_) *                                                             \
            ((m_) * (-0.5 - (1. / 3.) * (m_) + (n_)) + 2. * (n_) + 23. / 6.)))

#define FADDS_GEQRF(m_, n_)                                                    \
    (((m_) > (n_))                                                             \
         ? ((n_) * ((n_) * (0.5 - (1. / 3.) * (n_) + (m_)) + 5. / 6.))         \
         : ((m_) *                                                             \
            ((m_) * (-0.5 - (1. / 3.) * (m_) + (n_)) + (n_) + 5. / 6.)))

#define FLOPS_SGEQRF(m_, n_)                                                   \
    (FMULS_GEQRF((double)(m_), (double)(n_)) +                                 \
     FADDS_GEQRF((double)(m_), (double)(n_)))

extern "C" {
long int sgetrf_(long int *M, long int *N, float *h_A, long int *lda,
                 long int *ipiv, long int *info);
float slange_(char *norm, long int *M, long int *N, float *A, long int *lda,
              float *work);
void slaswp_(long int *N, float *A, long int *lda, long int *k1, long int *k2,
             int *ipiv, long int *incx);
void slacpy_(char *UPLO, long int *M, long int *N, float *A, long int *lda,
             float *B, long int *ldb);
void sgemm_(char *TransA, char *TransB, long int *M, long int *N, long int *K,
            const float *alpha, float *A, long int *lda, float *B,
            long int *ldb, const float *beta, float *C, long int *ldc);

void sorgqr_(long int const *m, long int const *n, long int const *k, float *A,
             long int const *lda, float const *tau, float *work,
             long int const *lwork, int *info);
void slaset_(char const *uplo, long int const *m, long int const *n,
             float const *alpha, float const *beta, float *A,
             long int const *lda
#ifdef LAPACK_FORTRAN_STRLEN_END
             ,
             size_t
#endif
);

void ssyrk_(char *Uplo, char *Trans, const long int *N, const long int *K,
            const float *alpha, const float *A, const long int *lda,
            const float *beta, float *C, const long int *ldc);
float slansy_(char const *norm, char const *uplo, long int const *n,
              float const *A, long int const *lda, float *work
#ifdef LAPACK_FORTRAN_STRLEN_END
              ,
              size_t, size_t
#endif
);
void sgeqrf_(long int const *m, long int const *n, float *A,
             long int const *lda, float *tau, float *work,
             long int const *lwork, int *info);
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
        for (signed long int j = 0; j < N; j++) {
            printf("%7.4f ", A[i + j * lda]);
        }
        printf("\n");
    }
}

template <class T> void print_int_vector(long int N, T *A, long int offset) {
    for (long int i = 0; i < N; i++)
        printf(" %ld", (long int)(A[i + offset]));

    printf("\n");
}

#define EL 10
typedef int info_int;
int main() {
    cusolver_print_environment();
    cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cuda_err = cudaSuccess;

    const float d_neg_one = -1.0;
    const float d_one = 1.0;
    const float c_neg_one = -1.0;
    const float c_one = 1.0;
    const float c_zero = 0.0;

    double gflops, gpu_perf, gpu_time, cpu_perf = 0, cpu_time = 0;
    float Anorm, error = 0, error2 = 0;
    float *h_A, *h_R, *tau, *h_work, tmp[1], unused[1];
    long int M, N, lda, lwork, min_mn, nb;
    int info;

    int status = 0;
    float tol = 0.00000178814;

#if 1
    long int msize[EL] = {1088, 2112, 3136, 4160, 5184,
                          6208, 7232, 8256, 9280, 10304};
    long int nsize[EL] = {1088, 2112, 3136, 4160, 5184,
                          6208, 7232, 8256, 9280, 10304};
#else
    // signed long int msize[EL] = {103040};
    // signed long int nsize[EL] = {103040};
    long int msize[10] = {7, 16, 125, 236, 337, 458, 569, 610, 711, 1112};
    long int nsize[10] = {7, 16, 125, 236, 337, 458, 569, 610, 711, 1112};
    // int nsize[10] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
#endif

    cusolverDnHandle_t cusolverDnHan = nullptr;
    cusolverDnCreate(&cusolverDnHan);

    int bufferSize = 0;

    int ngpu = 1;

    float *A_h = nullptr;
    float *A_d = nullptr;
    float *R_h = nullptr;

    int lapack = 1;
    int check = 1;

    info_int *info_d = nullptr;
    info_int *info_h = nullptr;

    cuda_err = cudaMalloc((void **)&info_d, 1 * sizeof(info_int));
    if (info_d == nullptr || cuda_err != cudaSuccess) {
        printf("cuda_err = %d\n", (int *)cuda_err);
    }
    info_h = (int *)malloc(1 * sizeof(info_int));

    printf("%% ngpu %lld\n", (long long)ngpu);
    printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |R - Q^H*A| "
           "  |I - Q^H*Q|\n");
    printf("%%================================================================="
           "=============\n");

    for (signed long int itest = 0; itest < EL; ++itest) {
        for (signed long int iter = 0; iter < 1; ++iter) {
            M = msize[itest];
            N = nsize[itest];
            min_mn = std::min(M, N);
            lda = M;

            A_h = (float *)malloc(lda * N * sizeof(float));
            cuda_err = cudaMalloc((void **)&A_d, lda * N * sizeof(float));
            if (cuda_err != cudaSuccess) {
                printf("cuda_err = %d", (int)cuda_err);
            }

            srand_rand_float(2222, A_h, lda * N);
            cuda_err = cudaMemcpy(A_d, A_h, lda * N * sizeof(float),
                                  cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
                printf("cuda_err = %d", (int)cuda_err);
            }

            gflops = FLOPS_SGEQRF(M, N) / 1e9;

            float *tau_h = nullptr;
            float *tau_d = nullptr;

            solver_status = cusolverDnSgeqrf_bufferSize(cusolverDnHan, M, N,
                                                        A_d, lda, &bufferSize);

            float *Workspace = nullptr;
            cuda_err =
                cudaMalloc((void **)&Workspace, bufferSize * sizeof(float));
            if (cuda_err != cudaSuccess) {
                printf("cuda_err = %d", (int)cuda_err);
            }
            cuda_err = cudaMalloc((void **)&tau_d, min_mn * sizeof(float));

            cudaDeviceSynchronize();
            gpu_time = solver_wtime();
            solver_status =
                cusolverDnSgeqrf(cusolverDnHan, M, N, A_d, lda, tau_d,
                                 Workspace, bufferSize, info_d);
            cudaDeviceSynchronize();
            gpu_time = solver_wtime() - gpu_time;
            if (solver_status != 0)
                printf("solver_status = %d\n", (signed long int)solver_status);

            gpu_perf = gflops / gpu_time;
            // fetch info_d
            cuda_err = cudaMemcpy(info_h, info_d, 1 * sizeof(info_int),
                                  cudaMemcpyDeviceToHost);

            if (cuda_err != cudaSuccess) {
                printf("cuda_err = %d", (int)cuda_err);
            }

            if (*info_h != 0) {
                printf("cusolverDnSgetrf returned error %lld: "
                       "%s.\n",
                       (long long)*info_h, solver_strerror(info));
            }

            if (check) {
                long int ldq = M;
                long int ldr = min_mn;

                float *Q = nullptr;
                float *R = nullptr;
                float *work = nullptr;
                long int nb = 32;

                int err = posix_memalign((void **)&Q, 64,
                                         ldq * min_mn * sizeof(float));
                if (err != 0) {
                    Q = NULL;
                    return -1;
                }

                err = posix_memalign((void **)&R, 64, ldr * N * sizeof(float));
                if (err != 0) {
                    R = NULL;
                    return -1;
                }

                long int llwork = N * nb;
                err =
                    posix_memalign((void **)&work, 64, llwork * sizeof(float));
                if (err != 0) {
                    work = NULL;
                    return -1;
                }

                err = posix_memalign((void **)&tau, 64, min_mn * sizeof(float));
                if (err != 0) {
                    tau = NULL;
                    return -1;
                }

                // R_h = (float *)malloc(lda*N*sizeof(float));
                err =
                    posix_memalign((void **)&R_h, 64, lda * N * sizeof(float));
                if (err != 0) {
                    R_h = NULL;
                    return -1;
                }

                cuda_err = cudaMemcpy(tau, tau_d, min_mn * sizeof(float),
                                      cudaMemcpyDeviceToHost);
                cuda_err = cudaMemcpy(R_h, A_d, lda * N * sizeof(float),
                                      cudaMemcpyDeviceToHost);

                ///////////////////////////////////////////////////////////////////////////////////////
                slacpy_("L", &M, &min_mn, R_h, &lda, Q, &ldq);
                sorgqr_(&M, &min_mn, &min_mn, Q, &ldq, tau, work, &llwork,
                        &info);
                if (info != 0)
                    printf("info = %d", info);

                // copy K by N matrix R
                slaset_("L", &min_mn, &N, &c_zero, &c_zero, R, &ldr);
                slacpy_("U", &min_mn, &N, R_h, &lda, R, &ldr);

                // error = || R - Q^H*A || / (N * ||A||)
                sgemm_("C", "N", &min_mn, &N, &M, &c_neg_one, Q, &ldq, A_h,
                       &lda, &c_one, R, &ldr);
                Anorm = slange_("1", &M, &N, A_h, &lda, work);
                error = slange_("1", &min_mn, &N, R, &ldr, work);

                if (N > 0 && Anorm > 0)
                    error /= (N * Anorm);

                // set R = I (K by K identity), then R = I - Q^H*Q
                // error = || I - Q^H*Q || / N
                slaset_("U", &min_mn, &min_mn, &c_zero, &c_one, R, &ldr);
                ssyrk_("U", "C", &min_mn, &M, &d_neg_one, Q, &ldq, &d_one, R,
                       &ldr);
                error2 = slansy_("1", "U", &min_mn, R, &ldr, work);

                if (N > 0)
                    error2 /= N;
#if 0
#endif
                ///////////////////////////////////////////////////////////////////////////////////////

                if (Q != nullptr) {
                    free(Q);
                    Q = nullptr;
                }
                if (R != nullptr) {
                    free(R);
                    R = nullptr;
                }
                if (work != nullptr) {
                    free(work);
                    work = nullptr;
                }
                if (tau != nullptr) {
                    free(tau);
                    tau = nullptr;
                }
                if (R_h != nullptr) {
                    free(R_h);
                    R_h = nullptr;
                }
                // if( != nullptr) {free(); = nullptr;}
            }

            if (lapack) {
                lwork = -1;
                // void sgeqrf_(long int const* m, long int const* n, float* A,
                // long int const* lda, float* tau, float* work, long int const*
                // lwork, long int* info );
                sgeqrf_(&M, &N, unused, &lda, unused, tmp, &lwork, &info);
                if (info != 0) {
                    printf("sgeqrf0 of lapack returned error %lld: %s.\n",
                           (long long)info, solver_strerror(info));
                }
                lwork = tmp[0]; // printf("LL:: lwork = %d\n", lwork = tmp[0]);
                float *work = nullptr;

                work = (float *)malloc(lwork * sizeof(float));
                tau_h = (float *)malloc(min_mn * sizeof(float));

                srand_rand_float(2222, A_h, lda * N);

                cpu_time = solver_wtime();
                sgeqrf_(&M, &N, A_h, &lda, tau_h, work, &lwork, &info);
                cpu_time = solver_wtime() - cpu_time;

                //                printf("R+tau lapack =\n");   print_matrix(7,
                //                7, A_h+((M-7) + (N-7)*lda), lda);

                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("sgeqrf of lapack returned error %lld: %s.\n",
                           (long long)info, solver_strerror(info));
                }

                if (work != nullptr) {
                    free(work);
                    work = nullptr;
                }
                if (tau_h != nullptr) {
                    free(tau_h);
                    tau_h = nullptr;
                }
            }

            /* =====================================================================
               Print performance and error.
               ===================================================================
             */
            printf("%5lld %5lld   ", (long long)M, (long long)N);
            if (lapack) {
                printf("%7.2f (%7.2f)", cpu_perf, cpu_time);
            } else {
                printf("  ---   (  ---  )");
            }
            printf("   %7.2f (%7.2f)   ", gpu_perf, gpu_time);
            if (check) {
                bool okay = (error < tol && error2 < tol);
                status += !okay;
                printf("%11.2e   %11.2e   %s\n", error, error2,
                       (okay ? "ok" : "failed"));
            } else {
                printf("    ---\n");
            }

            // solver_status = cusolverDnSgeqrf();
            if (A_d != nullptr) {
                cudaFree(A_d);
                A_d = nullptr;
            }
            if (Workspace != nullptr) {
                cudaFree(Workspace);
                Workspace = nullptr;
            }
            if (tau_d != nullptr) {
                cudaFree(tau_d);
                tau_d = nullptr;
            }
            if (A_h != nullptr) {
                free(A_h);
                A_h = nullptr;
            }
            if (tau_h != nullptr) {
                free(tau_h);
                tau_h = nullptr;
            }
            // if( != nullptr){free(); = nullptr;}
        }
    }

    if (info_h != nullptr) {
        free(info_h);
        info_h = nullptr;
    }
    if (info_d != nullptr) {
        cudaFree(info_d);
        info_d = nullptr;
    }

    cusolverDnDestroy(cusolverDnHan);
    return 0;
}
