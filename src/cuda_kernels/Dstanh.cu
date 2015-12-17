extern "C"   
{
  __global__ void Dstanh(const int lengthX, const double sf, const double *gradc, const double *fc,  double *gradn)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	gradn[i] += sf*gradc[i]*(1.0-(fc[i]/sf)*(fc[i]/sf));
      }
    }
}