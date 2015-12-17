#function DemoLinRegCUDA4()


    # note that this linear regression code is not the fastest way to do this since it is wasteful in terms of storage. We would need to code an operation Ax+beta*b where A,x,b are CudaArrays and beta is a Float64. This would mean we can then avoid the separate A*x (which stores this on the graph) and then Ax-y. Hence we can cut the storage significantly this way.

    #include("setup.jl")

    #H=[4 3 3 4] # number of units in each layer
H=[2 2]
    L=length(H)

    Ntrain=4
    # node indices:
    w=Array(Any,L) # weight index
    h=Array(Any,L) # hidden layer index (note that I call the input layer h[1])

    StartCode()
    ytrain=h[1]=ADnode()
    for i=2:L-1
        w[i]=ADvariable()
        h[i]=stanh(w[i]*h[i-1])
        #h[i]=rectlin(w[i]*h[i-1])
    end
    w[L]=ADvariable()
    h[L]=sigmoid(w[L]*h[L-1])
    loss=BinaryKullbackLeiblerLoss(h[L],ytrain)

    net=EndCode() # defines the graph

    # instantiate parameter nodes and inputs:
    @gpu CUDArt.init([0])
    net.value[h[1].index]=CudaArray(rand(H[1],Ntrain))
    for i=2:L
        net.value[w[i].index]=CudaArray(.5*sign(randn(H[i],H[i-1]))/sqrt(H[i-1]))
    end

    net=compile(net;debug=true) # compile and preallocate memory

    gradcheck(net;showgrad=false) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow



#end

