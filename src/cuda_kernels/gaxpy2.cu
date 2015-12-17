// filename: gaxpy2.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "gaxpy2"
{
    __global__ void gaxpy2(const double *a, const double *b, double *c)
    {
        int i = threadIdx.x + threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
        c[i] = a[0]*b[i] + c[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
    }
}