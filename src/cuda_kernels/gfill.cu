extern "C"  
{
    __global__ void gfill(const int n, const double *a, double *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<n)
	{
        c[i] = a[0];
	}
    }
}