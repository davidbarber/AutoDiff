// B=diag(A)

extern "C"   
{
  __global__ void diag_kernel(const int lengthA, const double *a, double *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	  {
	    b[i]=a[i+i*lengthA];
	  }
    }
}