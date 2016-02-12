extern "C"   
{
  __global__ void stanh_32(const int lengthA, const float alpha, const float *a, float *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	{
	  b[i] = alpha*tanh(a[i]);
	}
    }
}