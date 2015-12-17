extern "C"  
{
    __global__ void expkernel(const int lengthA, const double *a,  double *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	{
	  b[i] = exp(a[i]); 
	}
    }
}