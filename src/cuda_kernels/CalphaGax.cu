extern "C"  
{
    __global__ void CalpahGax(const double alpha, const double *a, const double *b, double *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        c[i] = alpha*a[0]*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
    }
}