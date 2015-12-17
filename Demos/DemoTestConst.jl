#function DemoLinRegCUDA4()


    # note that this linear regression code is not the fastest way to do this since it is wasteful in terms of storage. We would need to code an operation Ax+beta*b where A,x,b are CudaArrays and beta is a Float64. This would mean we can then avoid the separate A*x (which stores this on the graph) and then Ax-y. Hence we can cut the storage significantly this way.

    #include("setup.jl")


    StartCode()

    A=ADvariable()
    #loss=sumSquare(rectlin(2*A-0.5))+0.5*sum(A)
    loss=0.5*sum(A)+7*sumSquare(A)

    net=EndCode() # defines the graph

    # instantiate parameter nodes and inputs:
    #value=Array(Any,NodeCounter()) # function values on the nodes

    D=M=N=2
    AA=rand(D,M); XX=rand(M,N); YY=rand(D,N);
    BB=randn(2,1);

    @gpu CUDArt.init([0])
    @gpu net.value[A]=CudaArray(AA)
    @cpu net.value[A]=AA
#@gpu net.value[scal]=CudaArray(2*ones(1,1))
#@cpu net.value[scal]=2.0*ones(1,1)
 #   @gpu net.value[X]=CudaArray(XX)
 #  @cpu net.value[X]=XX
   #@gpu net.value[b]=CudaArray(BB)
    #net.value[lambda]=0.5

    #net.value[Z.index]=YY
    #println(net.value)

    net=compile(net;debug=true) # compile and preallocate memory
    #net=compilde(net) # compile and preallocate memory

    #for i=1:10
#ADforward!(net;exclude=[],debug=true,AllocateMemory=false)
#ADbackward!(net;debug=true)
    #end

gradcheck(net;showgrad=true) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow
 #   @cpu gradcheck(net;showgrad=true) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow
#@gpu gradcheckGPU(net;showgrad=false)


#end

