extern "C"  
{
    __global__ void vabs(const int n, const double *a, double *b)
    {	       
      int i =	 threadIdx.x + blockIdx.x * blockDim.x;
      if (i<n) 
	{b[i]=fabs(a[i]);}
    }	  
}
