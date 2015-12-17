#function DemoLinRegCUDA4()


    # note that this linear regression code is not the fastest way to do this since it is wasteful in terms of storage. We would need to code an operation Ax+beta*b where A,x,b are CudaArrays and beta is a Float64. This would mean we can then avoid the separate A*x (which stores this on the graph) and then Ax-y. Hence we can cut the storage significantly this way.

    #include("setup.jl")
    function  LSTM(xt,htm,inputbias)
        sigmiod
    end

    StartCode()

    A=ADvariable()
    loss=sum(AtransB(A,A))
#    loss=meanSquare(A')

    net=EndCode() # defines the graph

    # instantiate parameter nodes and inputs:
    #value=Array(Any,NodeCounter()) # function values on the nodes

    AA=rand(2,2);
    BB=rand(2,2);
    CC=rand(2,2);

    @gpu CUDArt.init([0])
    @gpu net.value[A]=CudaArray(AA)
    @cpu net.value[A]=AA
 # @cpu net.value[B]=BB
 # @cpu net.value[C]=CC
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

