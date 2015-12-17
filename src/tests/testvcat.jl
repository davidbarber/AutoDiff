#function DemoLinRegCUDA4()


    # note that this linear regression code is not the fastest way to do this since it is wasteful in terms of storage. We would need to code an operation Ax+beta*b where A,x,b are CudaArrays and beta is a Float64. This would mean we can then avoid the separate A*x (which stores this on the graph) and then Ax-y. Hence we can cut the storage significantly this way.

    #include("setup.jl")

    #AtransB(A,B)=A'*B

    StartCode()

    A=ADvariable()
    B=ADvariable()
    #    loss=sum(AtransB(A,A))
    #    loss=meanSquare(A')
    loss=sumSquare(hcat([A B A]))
#    loss=sumSquare(hcat([A B A B B A]))

    net=EndCode() # defines the graph

    # instantiate parameter nodes and inputs:
    #value=Array(Any,NodeCounter()) # function values on the nodes

    net.value[A]=rand(3,1);
    net.value[B]=rand(3,1);

    net=compile(net;debug=true) # compile and preallocate memory

    @gpu CUDArt.init([0])
    @gpu net=convert(net,"GPU")

gradcheck(net;showgrad=true) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow
 #   @cpu gradcheck(net;showgrad=true) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow
#@gpu gradcheckGPU(net;showgrad=false)


#end

