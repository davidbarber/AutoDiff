extern "C"   
{
  __global__ void DYbinaryentropyXsigmoidY(const int lengthX, const double *x,  const double *y, const double *t, double *z)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	z[i] += t[0]*(1.0/(1.0+exp(-y[i]))-x[i])/lengthX;
      }
    }
}