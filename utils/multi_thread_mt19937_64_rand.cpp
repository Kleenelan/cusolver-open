#include "./multi_thread_mt19937_64_rand.h"
void srand_rand_float(int seed, float* A, unsigned long int len)
{
    parallel_mt19937<312>::GetInstance()->srand(2024);
    parallel_mt19937<312>::GetInstance()->rand_float(A, len);
}



#define BUILD_MAIN_

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef BUILD_MAIN

void print_matrix(unsigned long m, unsigned long n, float* A, unsigned long lda)
{
    for(unsigned long i=m-7; i<m; i++){
        for(unsigned long j=n-7; j<n; j++){
            printf("%7.4f ", A[i + j*lda]);
        }
        printf("\n");
    }
}


#include <string.h>

int main(){

    unsigned long m = 312*4*32*32*128;
    unsigned long n = 36;
    unsigned long lda = m;

    float* A = nullptr;
    printf("LL: 00\n");
    A = (float*)malloc(lda * n * sizeof(float));
    //memset(A, 0x7F, lda*n*sizeof(float));
    printf("LL:: 01\n");

    srand_rand_float(2024, A, lda*n);

    printf("LL:: 02\n");
    print_matrix(m, n, A, lda);

    free(A);
    return 0;
}

#endif

