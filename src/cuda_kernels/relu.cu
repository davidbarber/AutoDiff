extern "C"  
{
  __global__ void relu(const int n, const double *a, double *b)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n)
      {
	if (a[i]>0.0)
	  {b[i] = a[i];}	
	else	 
	  {b[i] = 0.0;}
      }	
  }
}