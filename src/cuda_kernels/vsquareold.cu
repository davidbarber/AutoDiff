// filename: vsquare.cu
// a simple CUDA kernel to element multiply vector with itself

extern "C"   // ensure function name to be exactly "vsquare"
{
    __global__ void vsquare(const double *a, double *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        c[i] = a[i] * a[i];
    }
}