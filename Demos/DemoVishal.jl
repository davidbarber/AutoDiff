#function DemoMNIST()
    # Training a deep autoencoder on MNIST
    # The method uses Nesterov's accelerated gradient, with minibatches
    # (c) David Barber, University College London 2015

    #TODO: GPU version

    if 1==1
    Ntrain=10
    BatchSize=Ntrain
    TrainingIts=10 # number of Nesterov updates
    X=10
    data=1.0*rand(X,Ntrain)
    # bound away from 0 and 1 to avoid log(0) problems:
    tol=0.000001
    data[data.>(1-tol)]=1-tol
    data[data.<tol]=tol
end
    # Construct the DAG function:
    #H=[784 1000 500 250 30 250 500 1000 784] # number of units in each layer
    H=[X 20 2 X] # number of units in each layer
    #H=[784 300 30 300 784] # number of units in each layer

    L=length(H) # number of hidden layers
    # node indices:
    w=Array(ADnode,L) # weights
    bias=Array(ADnode,L) # biases
    h=Array(ADnode,L) # hidden layer index (input layer is h[1])
dropout=Array(ADnode,L)

    StartCode()
    ytrain=h[1]=ADnode()
    for layer=2:L-1
        dropout[layer]=ADnode()
        w[layer]=ADvariable()
        bias[layer]=ADvariable()
        h[layer]=diagm(dropout[layer])*rectlinAXplusBias(w[layer],h[layer-1],bias[layer])
    end
    w[L]=ADvariable()
    bias[L]=ADvariable()
dropout[L]=ADnode()
    h[L]=diagm(dropout[L])*AXplusBias(w[L],h[L-1],bias[L])
    ypred=sigmoid(h[L]) # just use for testing, not training
    loss=BinaryKullbackLeiblerLossXsigmoidY(ytrain,h[L])
    net=EndCode()

    #instantiate root node values:
    net.value[h[1].index]=data[:,1:BatchSize];
    for i=2:L
        net.value[w[i]]=.5*sign(randn(H[i],H[i-1]))/sqrt(H[i-1])
        net.value[bias[i]]=.5*sign(randn(H[i],1))
        net.value[dropout[i]]=1.0*(rand(H[i],1).>0.5)
    end

    net=compile(net) # compile the DAG and preallocate memory

@gpu CUDArt.init([0])
@gpu net=convert(net,"GPU")
gradcheck(net)
    println("Training:")
    error=Array(Float64,0)
    ParsToUpdate=Parameters(net)
    velo=NesterovInit(net)
    minibatchstart=1 # starting datapoint for the minibatch
    for iter=1:TrainingIts
        LearningRate=0.5/(1+iter/500)
        minibatchstart,minibatch=GetBatch(minibatchstart,BatchSize,Ntrain)
#        net.value[ytrain]=data[:,minibatch] # select batch of data
# @gpu       net.value[ytrain]=CudaArray(data[:,minibatch]) # select batch of data
        ADforward!(net;exclude=[ypred]) # can ignore ypred during training
        ADbackward!(net)
        push!(error,extract(net.value[net.FunctionNode])[1])
        println("iteration $iter: training loss = $(error[iter])")
        for par in ParsToUpdate
            #GradientDescentUpdate!(net.value[par],net.gradient[par],LearningRate)
            NesterovGradientDescentUpdate!(net.value[par],net.gradient[par],velo[par],LearningRate,iter)
        end
    end

    # Get the predictions:
    ADforward!(net;exclude=[])
    plot(error); title("training loss")
    figure()
    for i=1:20 # plot the reconstructions for a few datapoints
        p=imshow([reshape(net.value[h[1]][:,i],28,28)'  reshape(net.value[ypred.index][:,i],28,28)'],interpolation="none",cmap=ColorMap("gray"))
        display(p)
        println("press key to continue")
        readline(STDIN)
    end
