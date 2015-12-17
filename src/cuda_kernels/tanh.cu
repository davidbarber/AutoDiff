extern "C"   
{
  __global__ void stanh(const int lengthA, const double alpha, const double *b,)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	{
	  b[i] = alpha*tanh(a[i]);
	}
    }
}