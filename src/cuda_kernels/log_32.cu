extern "C"  
{
    __global__ void logkernel_32(const int lengthA, const float *a,  float *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	{
	  b[i] = log(a[i]); 
	}
    }
}