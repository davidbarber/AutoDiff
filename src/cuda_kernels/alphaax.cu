extern "C"  
{
    __global__ void alphaax(const int lengthC, const double alpha, const double *a, const double *b, double *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i<lengthC)
        {	
        c[i] = alpha*a[0]*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
        }
    }
}