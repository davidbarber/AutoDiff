extern "C"   
{
  __global__ void tx1mx(const int lengthX, const double *t, const double *x,  double *z)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	z[i] += t[i]*x[i]*(1.0-x[i]);
      }
    }
}