// filename: vmult!.cu
// a simple CUDA kernel to element multiply two vectors C=alpha*A.*B

extern "C"   // ensure function name to be exactly "vmult!"
{
  __global__ void vdivupdate(const int lengthA, const double alpha, const double *a, const double *b, double *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	  {
	    c[i] += alpha*a[i] / b[i];
	  }
    }
}