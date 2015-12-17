extern "C"   
{
  __global__ void DmeanSquareLoss(const int lengthx, const double pref, const double *gradc, const double *x,const double *y, double *gradn )
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthx)
	  {
	    gradn[i] += pref * gradc[0] * (x[i]-y[i]);
	  }	
    }
}