extern "C"   
{
    __global__ void sigmoid(const int lengthA, const double *a,  double *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	{
	  b[i] = exp(-a[i]);  
	  b[i] = 1.0/(1.0+b[i]);
	}
    }
}