extern "C"
{
__global__ void serialsum(const int n, const double *x, double *y)
{
y[0]=x[0];
for (int i = 1; i<n; i++)
{
y[0]+=x[i];
}
}
}