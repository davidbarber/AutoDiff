extern "C"  
{
    __global__ void alphaax_32(const int lengthC, const float alpha, const float *a, const float *b, float *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i<lengthC)
        {	
        c[i] = alpha*a[0]*b[i];  // REMEMBER ZERO INDEXING IN C LANGUAGE!!
        }
    }
}