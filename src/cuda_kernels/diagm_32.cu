// B=diagm(A)

extern "C"   
{
  __global__ void diagm_kernel_32(const int lengthA, const float *a, float *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	  {
	    b[i+i*lengthA] = a[i];
	  }
    }
}