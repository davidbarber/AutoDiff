// B=diagm(A)

extern "C"   
{
  __global__ void diagm_kernel(const int lengthA, const double *a, double *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	  {
	    b[i+i*lengthA] = a[i];
	  }
    }
}