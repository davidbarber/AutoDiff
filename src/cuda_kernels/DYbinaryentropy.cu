extern "C"   
{
  __global__ void DYbinaryentropy(const int lengthX, const double *x,  const double *y, const double *t, double *z)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	z[i] += t[0]*((y[i]-x[i])/(y[i]*(1.0-y[i])))/lengthX;
      }
    }
}