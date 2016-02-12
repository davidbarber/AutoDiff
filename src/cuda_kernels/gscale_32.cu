extern "C"  
{
    __global__ void gscale_32(const int lengthB, const float *a, float *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthB)
	{
        b[i] = a[0]*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
	}
    }
}