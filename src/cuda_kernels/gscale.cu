extern "C"  
{
    __global__ void gscale(const int lengthB, const double *a, double *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthB)
	{
        b[i] = a[0]*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
	}
    }
}