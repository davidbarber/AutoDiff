#function DemoLinRegCUDA4()


    # note that this linear regression code is not the fastest way to do this since it is wasteful in terms of storage. We would need to code an operation Ax+beta*b where A,x,b are CudaArrays and beta is a Float64. This would mean we can then avoid the separate A*x (which stores this on the graph) and then Ax-y. Hence we can cut the storage significantly this way.

    #include("setup.jl")

    StartCode()
    A=ADvariable()
    X=ADvariable()
    Y=ADnode()
    loss=BinaryKullbackLeiblerLoss(sigma(A*X),Y)

    net=network() # defines the graph

    # instantiate parameter nodes and inputs:
    #value=Array(Any,NodeCounter()) # function values on the nodes

    D=M=N=10
    AA=randn(D,M); XX=rand(M,N); YY=rand(D,N);

    @gpu CUDArt.init([0])
    net.value[A.index]=cArray(AA)
    net.value[X.index]=cArray(XX)
    net.value[Y.index]=cArray(YY)

    #net=compile(net;debug=true) # compile and preallocate memory
    net=compile(net) # compile and preallocate memory

    for i=1:10
    ADforward!(net;exclude=[],debug=false,AllocateMemory=false)
    ADbackward!(net;debug=false,Reset=true)
    end

    @cpu gradcheck(net;showgrad=false) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow



#end

