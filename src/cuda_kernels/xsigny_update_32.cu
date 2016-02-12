extern "C"  
{
  __global__ void xsigny_update_32(const int n, const float *a, float *b, float *c)
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
