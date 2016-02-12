extern "C"   
{
    __global__ void vsquare_32(const float *a, float *c)
    {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
	double v = a[i];
        c[i] = v*v;
   }
}