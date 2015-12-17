extern "C"  
{
    __global__ void vsign(const int n, const double *a, double *b)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<n)
	  {
	    if (a[i]<0)
	      {b[i]=-1.0;}
	    else
	      {if (a[i]>0)
		  {b[i]=1.0;}
		else
		  {b[i]=0.0;}		    
	      }
	  }
    }
}