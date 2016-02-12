extern "C"  
{
  __global__ void alex(const int n, const double *a, double *b)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n)
      {
	if (a[i]>-0.5)
	  {b[i] = a[i];}	
	else	 
	  {b[i] = -0.5*log(-a[i])-0.5*(1-log(0.5));}
      }	
  }
}