extern "C"  
{
  __global__ void vAoverBupdate(const int lengthA, const double alpha, const double *gradc, const double *a, const double *b, double *gradn)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthA)
	  {
	    gradn[i] -= alpha*gradc[i]*a[i] / (b[i]* b[i]);
	  }
    }
}