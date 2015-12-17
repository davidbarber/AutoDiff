extern "C"   
{
  __global__ void binaryentropyXsigmoidY(const int lengthX, const double *x,  const double *y, double *z)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	z[i]=x[i]*log(x[i])+(1.0-x[i])*log(1.0-x[i])-x[i]*y[i]+log(1.0+exp(y[i]));
      }
  }
}