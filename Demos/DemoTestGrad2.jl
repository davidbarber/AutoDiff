#function DemoLinRegCUDA4()


    # note that this linear regression code is not the fastest way to do this since it is wasteful in terms of storage. We would need to code an operation Ax+beta*b where A,x,b are CudaArrays and beta is a Float64. This would mean we can then avoid the separate A*x (which stores this on the graph) and then Ax-y. Hence we can cut the storage significantly this way.

    #include("setup.jl")

    w=Array(Any,2)
    StartCode()
    #A=ADvariable()
    X=ADvariable()
    A=ADvariable()
    b=ADvariable()
    lambda=ADnode()
    #Z=ADvariable()
    #loss=sumSq(X+X-X*X+X*X*X-X*X*X*X*X)+sum(X*Y)
    #loss=sum([Z*X X*Z Y])+sumSq([X X Y Z Z Z Z Z Z])-sum(X)
    #loss=sumSq(X+X-X*X+X*X*X-X*X*X*X*X)+sumSq(X*X)+sumSq([X X X])-sum([X X Y X Y X X X X X X X X])

    #loss=meanSquareLoss(X*X*X-X*X+X,X*X)
    #loss=meanSquareLoss(Y,X)
    #loss=BinaryKullbackLeiblerLoss(sigmoid(X),sigmoid(Y))+sumSq(X*Y)-sum([X X X X])+sigmoid(sum(X))+sum((X-Y)*(X-Y))
    #loss=BinaryKullbackLeiblerLossXsigmoidY(sigmoid(X),Y)
    #loss=BinaryKullbackLeiblerLossXsigmoidY(sigmoid(X),Y)
    #loss=meanSquare(sigmoidAXplusBias(A,sigmoidAXplusBias(A,X,b),b))
    #loss=meanSquare(rectlinAXplusBias(A,X,b)-sigmoid(A*X))
    loss=meanSquare(sigmoid(AXplusBias(A,X,b))-AXplusBias(A,X,b))+lambda*sum(b)+sum(b.*b)+sum(A./X) + (sum(A*A)-sum(A*X))./(sum(X*X)+sum(A.*A))

    net=network() # defines the graph

    # instantiate parameter nodes and inputs:
    #value=Array(Any,NodeCounter()) # function values on the nodes

    D=M=N=2
    AA=randn(D,M); XX=randn(M,N); YY=randn(D,N);
    BB=randn(2,1);

    @gpu CUDArt.init([0])
    net.value[A]=AA
    net.value[X]=XX
    net.value[b]=BB
    net.value[lambda]=0.5

    #net.value[Z.index]=YY
    #println(net.value)

    net=compile(net;debug=true) # compile and preallocate memory
    #net=compilde(net) # compile and preallocate memory

    #for i=1:10
    #    ADforward!(net;exclude=[],debug=false,AllocateMemory=false)
    #    ADbackward!(net;debug=false)
    #end

    @cpu gradcheck(net;showgrad=true) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow



#end

