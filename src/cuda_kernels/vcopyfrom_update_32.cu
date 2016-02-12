extern "C"  
{
  __global__ void vcopyfrom_update_32(const int n, const int shift, const float *a, float *b)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n)
      {
	b[i] += a[i+shift];
      }
  }
}