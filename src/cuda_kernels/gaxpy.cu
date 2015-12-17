// filename: gaxpy.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "gaxpy"
{
    __global__ void gaxpy(const int lengthC, const double *a, const double *b, double *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthC)
	{
        c[i] = a[0]*b[i] + c[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!	
	}
    }
}