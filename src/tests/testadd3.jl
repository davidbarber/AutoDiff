N=2;
A=CudaArray(rand(N,N));
B=CudaArray(rand(N,N));
C=CudaArray(rand(N,N));

        device(0)

for i=1:10
tic()
#result = devices(dev->capability(dev)[1]>=2) do devlist
#    mymodule.init(devlist) do dev
        vadd(A,B,C)
        #to_host(C)-to_host(B)-to_host(A)
#    end
end
toc()
end
