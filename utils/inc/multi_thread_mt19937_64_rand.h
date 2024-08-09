#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>

// seed_base
// void gen_512(seed, A, start_idx, N)
// gen_512(seed_base+0, A, 0*512, N);  gen_512(seed_base + 1, A+1*512); gen_512(seed_base + 2, A+2*512);     ...    gen_512(seed_base + n, A, , N); ...
#if 1


//<312, 156, 0xB5026F5AA96619E9ULL, 0xFFFFFFFF80000000ULL, 0x7FFFFFFFULL>


//NN is tile length
//#define NN 312
//#define MM 156
//#define MATRIX_A 0xB5026F5AA96619E9ULL
//#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
//#define LM 0x7FFFFFFFULL /* Least significant 31 bits */


template <unsigned int NN, unsigned int MM, unsigned long long int MATRIX_A, unsigned long long int UM, unsigned long long int LM>
class thread_mt199937{

public:
void srand(unsigned long long seed)
{
    init_genrand64(seed);
}

/* generates a random number on [0, 2^63-1]-interval */
long long genrand64_int63(void)
{
    return (long long)(genrand64_int64() >> 1);
}

/* generates a random number on [0,1]-real-interval */
double genrand64_real1(void)
{
    return (genrand64_int64() >> 11) * (1.0/9007199254740991.0);
}

/* generates a random number on [0,1)-real-interval */
double genrand64_real2(void)
{
    return (genrand64_int64() >> 11) * (1.0/9007199254740992.0);
}

/* generates a random number on (0,1)-real-interval */
double genrand64_real3(void)
{
    return ((genrand64_int64() >> 12) + 0.5) * (1.0/4503599627370496.0);
}


private:
/* The array for the state vector */
unsigned long long mt[NN];
/* mti==NN+1 means mt[NN] is not initialized */
int mti=NN+1;

/* initializes mt[NN] with a seed */
void init_genrand64(unsigned long long seed)
{
    mt[0] = seed;
    for (mti=1; mti<NN; mti++)
        mt[mti] =  (6364136223846793005ULL * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void init_by_array64(unsigned long long init_key[],
		     unsigned long long key_length)
{
    unsigned long long i, j, k;
    init_genrand64(19650218ULL);
    i=1; j=0;
    k = (NN>key_length ? NN : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * 3935559000370003845ULL))
          + init_key[j] + j; /* non linear */
        i++; j++;
        if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=NN-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * 2862933555777941757ULL))
          - i; /* non linear */
        i++;
        if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
    }

    mt[0] = 1ULL << 63; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0, 2^64-1]-interval */
unsigned long long genrand64_int64(void)
{
    int i;
    unsigned long long x;
    //static
    unsigned long long mag01[2]={0ULL, MATRIX_A};

    if (mti >= NN) { /* generate NN words at one time */

        /* if init_genrand64() has not been called, */
        /* a default initial seed is used     */
        if (mti == NN+1)
            init_genrand64(5489ULL);

        for (i=0;i<NN-MM;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+MM] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        for (;i<NN-1;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+(MM-NN)] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        x = (mt[NN-1]&UM)|(mt[0]&LM);
        mt[NN-1] = mt[MM-1] ^ (x>>1) ^ mag01[(int)(x&1ULL)];

        mti = 0;
        //printf("LL::\n");
    }

    x = mt[mti++];

    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);

    return x;
}

};


#endif








////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
//#define NN 312
#include <mutex>

//using namespace std;
// multi thread accelerating， to be singleton
template <unsigned long NN>
class parallel_mt19937{
public:
    static parallel_mt19937* GetInstance();
    static void DestoryInstance();
//    int _base_seed;
    void srand(int seed)
    {
        _base_seed = seed;
    }
//template<>
void rand_float(float* A, unsigned long len);

private:
    int _base_seed = 0;
    static parallel_mt19937 *_p_parallel_mt19937; // pointer points self single instance
    static std::mutex _mutex; // lock for thread-safe

};
template<unsigned long NN>
parallel_mt19937<NN> *parallel_mt19937<NN>::_p_parallel_mt19937 = nullptr;
template<unsigned long NN>
std::mutex parallel_mt19937<NN>::_mutex;

template<unsigned long NN>
parallel_mt19937<NN> *parallel_mt19937<NN>::GetInstance()
{
     if (_p_parallel_mt19937 == nullptr) {
         std::lock_guard<std::mutex> lock(_mutex);
         if (_p_parallel_mt19937 == nullptr) {
             _p_parallel_mt19937 = new parallel_mt19937<NN>();
         }
      }

      return _p_parallel_mt19937;
}

template<unsigned long NN>
void parallel_mt19937<NN>::DestoryInstance()
{
        if (_p_parallel_mt19937 != nullptr) {
         std::lock_guard<std::mutex> lock(_mutex);
         if (_p_parallel_mt19937 != nullptr) {
             delete _p_parallel_mt19937;
             _p_parallel_mt19937 = nullptr;
         }
      }
}

//int pmt19937::_base_seed = 0;
template<unsigned long NN>
void parallel_mt19937<NN>::rand_float(float* A, unsigned long len)
{

    unsigned long tile_count = (len+NN-1)/NN;

    #pragma omp parallel
    {
        unsigned long block_dim = omp_get_num_threads();
        unsigned long thid = omp_get_thread_num();

        thread_mt199937<NN, NN/2, 0xB5026F5AA96619E9ULL, 0xFFFFFFFF80000000ULL, 0x7FFFFFFFULL> *t_mt_p
            = new thread_mt199937<NN, NN/2, 0xB5026F5AA96619E9ULL, 0xFFFFFFFF80000000ULL, 0x7FFFFFFFULL>();// to be singleton

        for(unsigned long tile_id=thid; tile_id<tile_count; tile_id+=block_dim)
        {
            //each tile has a specific seed: (_base_seed + tile_id) to smt19937, to keep consistence
            unsigned long tile_seed = _base_seed + tile_id;//*2046;
            t_mt_p->srand(tile_seed);

            unsigned long tile_idx_start = tile_id*NN;
            unsigned long tile_idx_END  = tile_idx_start + NN;
            unsigned long tile_idx_end = ((tile_idx_END) <= len)? (tile_idx_END-1) : len-1;

            for(unsigned long idx = tile_idx_start; idx<=tile_idx_end; idx++)
            {
                //A[idx]=float(idx);
                A[idx] = float(t_mt_p->genrand64_real2());

            }

        }
        delete t_mt_p;
    }
}

void srand_rand_float(int seed, float* A, unsigned long int len);

