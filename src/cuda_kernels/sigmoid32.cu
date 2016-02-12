extern "C"   
{
    __global__ void sigmoid32(const int lengthA, const float *a,  float *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	{
	  b[i] = exp(-a[i]);  
	  b[i] = 1.0/(1.0+b[i]);
	}
    }
}