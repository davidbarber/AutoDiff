extern "C"  
{
  __global__ void vcopyshift(const int n, const int shift, const double *a, double *b)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n)
      {
	b[i+shift] = a[i];
      }
  }
}