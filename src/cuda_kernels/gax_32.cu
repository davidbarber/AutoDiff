// filename: gax.cu
// a simple CUDA kernel to add two vectors

extern "C"  
{
    __global__ void gax_32(const int lengthC, const float *a, const float *b, float *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthC)
	{
        c[i] = a[0]*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
	}
    }
}