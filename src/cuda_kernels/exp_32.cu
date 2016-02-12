extern "C"  
{
    __global__ void expkernel_32(const int lengthA, const float *a,  float *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	{
	  b[i] = exp(a[i]); 
	}
    }
}