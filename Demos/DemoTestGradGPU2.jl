#function DemoLinRegCUDA4()


    # note that this linear regression code is not the fastest way to do this since it is wasteful in terms of storage. We would need to code an operation Ax+beta*b where A,x,b are CudaArrays and beta is a Float64. This would mean we can then avoid the separate A*x (which stores this on the graph) and then Ax-y. Hence we can cut the storage significantly this way.

    #include("setup.jl")

    w=Array(Any,2)
    StartCode()
    A=ADvariable()
scal=ADconst(1.0)
#    X=ADvariable()
 #   A=ADvariable()
    #b=ADvariable()
    #lambda=ADnode()
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
    #loss=meanSquare(sigmoid(AXplusBias(A,X,b))-AXplusBias(A,X,b))+lambda*sum(b)+sum(b.*b)+sum(A./X) + (sum(A*A)-sum(A*X))./(sum(X*X)+sum(A.*A))


    #loss=sum(X)+sum(X)-sum(X*X) +sum(X+X)-sum(X-X*X+X*X*X).*sum(X)+sum(A./X)./sum(X./A)

#    loss=sum(sigmoid(X)+stanh(X))
#loss=sum(stanh(X)+stanh(X))+sumSq(rectlin(X))
#loss=BinaryKullbackLeiblerLoss(sigmoid(X),sigmoid(A))+sumSq([X X*X A*X X-A])
#loss=BinaryKullbackLeiblerLossXsigmoidY(X,A)

#loss=meanSquareLoss(stanhAXplusBias(A,X,b),A)
#loss=meanSquare(A)
#loss=KLsoftmax(softmax(A),X)
#loss=sum(log(A)+log(A+A)+log(A)*exp(A))

#loss=sum(scal*A+scal*A+scal*scal)
loss=sum(scal+A)

#loss=sum(A./X)
#loss=sum(X.*A)
    net=network() # defines the graph

    # instantiate parameter nodes and inputs:
    #value=Array(Any,NodeCounter()) # function values on the nodes

    D=M=N=2
    AA=rand(D,M); XX=rand(M,N); YY=rand(D,N);
    BB=randn(2,1);

    @gpu CUDArt.init([0])
    @gpu net.value[A]=CudaArray(AA)
    @cpu net.value[A]=AA
@gpu net.value[scal]=CudaArray(2*ones(1,1))
@cpu net.value[scal]=2.0*ones(1,1)
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

