// B=diag(A)

extern "C"   
{
  __global__ void diag_kernel_32(const int lengthA, const float *a, float *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	  {
	    b[i]=a[i+i*lengthA];
	  }
    }
}