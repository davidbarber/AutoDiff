// filename: ax.cu
// a simple CUDA kernel to add two vectors

extern "C"   
{
    __global__ void ax_32(const int lengthC, const float a, const float *b, float *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthC)
	{
        c[i] = a*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
	}
    }
}