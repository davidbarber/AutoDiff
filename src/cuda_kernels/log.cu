extern "C"  
{
    __global__ void logkernel(const int lengthA, const double *a,  double *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	{
	  b[i] = log(a[i]); 
	}
    }
}