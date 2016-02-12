extern "C"   
{
  __global__ void Dstanh_32(const int lengthX, const float sf, const float *gradc, const float *fc,  float *gradn)
  {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<lengthX)
      {
	gradn[i] += sf*gradc[i]*(1.0-(fc[i]/sf)*(fc[i]/sf));
      }
    }
}