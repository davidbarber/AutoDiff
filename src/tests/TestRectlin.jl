#function DemoMNIST()
    # Training a deep autoencoder on MNIST
    # The method uses Nesterov's accelerated gradient, with minibatches
    # (c) David Barber, University College London 2015

    #TODO: GPU version

    StartCode()
    bias=ADvariable()
h=ADvariable()
W=ADvariable()
        h2=rectlinAXplusBias(W,h,bias)
    loss=sumSquare(h2)
    net=EndCode()

    #instantiate root node values:
net.value[bias]=rand(2,1);
net.value[W]=rand(2,2);
net.value[h]=rand(2,3)

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
