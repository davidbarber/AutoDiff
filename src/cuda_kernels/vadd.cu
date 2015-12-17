extern "C"  
{
    __global__ void vadd(const int n, const double *a, const double *b, double *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<n)
	{
        c[i] = a[i] + b[i];
	}
    }
}