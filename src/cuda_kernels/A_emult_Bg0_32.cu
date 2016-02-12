extern "C"  
{
  __global__ void A_emult_Bg0_32(const int n, const float *a, const float *b, float *c)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n)
      {
	if (b[i]>0.0)
	  {c[i] += a[i];}	
	else	 
	  {c[i] += 0.0;}
      }	
  }
}