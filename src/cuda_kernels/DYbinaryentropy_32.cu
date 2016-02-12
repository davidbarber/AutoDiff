extern "C"   
{
  __global__ void DYbinaryentropy_32(const int lengthX, const float *x,  const float *y, const float *t, float *z)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	z[i] += t[0]*((y[i]-x[i])/(y[i]*(1.0-y[i])))/lengthX;
      }
    }
}