extern "C"  
{
    __global__ void gfill_32(const int n, const float *a, float *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<n)
	{
        c[i] = a[0];
	}
    }
}