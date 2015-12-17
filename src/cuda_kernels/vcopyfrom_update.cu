extern "C"  
{
  __global__ void vcopyfrom_update(const int n, const int shift, const double *a, double *b)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n)
      {
	b[i] += a[i+shift];
      }
  }
}