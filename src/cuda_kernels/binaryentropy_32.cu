extern "C"   
{
  __global__ void binaryentropy_32(const int lengthX, const float *x,  const float *y, float *z)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	z[i] = x[i]*log(x[i]/y[i])+ (1.0-x[i])*log((1.0-x[i])/(1.0-y[i]));
	}
    }
}