extern "C"   
{
  __global__ void DYbinaryentropyXsigmoidY_32(const int lengthX, const float *x, const float *y, const float *t, float *z)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	z[i] += t[0]*(1.0/(1.0+exp(-y[i]))-x[i])/lengthX;
      }
    }
}