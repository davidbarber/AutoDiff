// filename: gaxpy.cu
// a simple CUDA kernel to add two vectors

extern "C"  
{
    __global__ void gaxpy_32(const int lengthC, const float *a, const float *b, float *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthC)
	{
        c[i] = a[0]*b[i] + c[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!	
	}
    }
}