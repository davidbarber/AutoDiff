extern "C"  
{
  __global__ void xsigny_update(const int n, const double *a, double *b, double *c)
  {	       
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n) 
      {
	if (b[i]>0) 
	  {c[i]+=a[i];}
	else
	  {if (b[i]<0) 
	      {c[i]-=a[i];}
	  }
      }
  }
}
