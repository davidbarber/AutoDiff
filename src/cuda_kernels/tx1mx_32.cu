extern "C"   
{
  __global__ void tx1mx_32(const int lengthX, const float *t, const float *x,  float *z)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	z[i] += t[i]*x[i]*(1.0-x[i]);
      }
    }
}