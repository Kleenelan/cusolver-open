#include <cuda_runtime.h>
#include <cublas_v2.h>

int main()
{

	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;

	cublasCreate(&cublasH);

	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cublasSetStream(cublasH, stream);



	cublasDestroy(cublasH);
	cudaStreamDestroy(stream);
	cudaDeviceReset();

}







