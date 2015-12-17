extern "C"   
{
  __global__ void binaryentropy(const int lengthX, const double *x,  const double *y, double *z)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	z[i] = x[i]*log(x[i]/y[i])+ (1.0-x[i])*log((1.0-x[i])/(1.0-y[i]));
	}
    }
}