extern "C"   
{
  __global__ void DmeanSquareLoss_32(const int lengthx, const float pref, const float *gradc, const float *x,const float *y, float *gradn )
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<lengthx)
	  {
	    gradn[i] += pref * gradc[0] * (x[i]-y[i]);
	  }	
    }
}