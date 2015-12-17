// filename: gaxpy2.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "gaxpy2"
{
    __global__ void gaxpy4(const int n, const double *a, const double *b, double *c)
    {
        int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < n) {
 c[i] = (double) i;  // REMEMBER ZERO INDEXING IN C LANGUAGE!!			
}		   

    }
}