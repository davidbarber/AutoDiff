# Training a deep autoencoder on MNIST
# The method uses Nesterov's accelerated gradient, with minibatches
# (c) David Barber, University College London 2015

using MAT

Ntrain=60000
BatchSize=100
TrainingIts=1000 # number of Nesterov updates
include("loadmnist.jl")
images,label=loadmnist()
r=randperm(size(images,2))
data=images[:,r]
# bound away from 0 and 1 to avoid log(0) problems:
tol=0.000001
data[data.>(1-tol)]=1-tol
data[data.<tol]=tol

H=[784 500 250 30 250 500 784] # number of units in each layer

L=length(H) # number of hidden layers
# node indices:
w=Array(ADnode,L) # weights
bias=Array(ADnode,L) # biases
h=Array(ADnode,L) # hidden layer index (input layer is h[1])

StartCode()
ytrain=h[1]=ADnode()
for layer=2:L-1
    w[layer]=ADvariable()
    bias[layer]=ADvariable()
    h[layer]=absAXplusBias(w[layer],h[layer-1],bias[layer])
    #h[layer]=rectlinAXplusBias(w[layer],h[layer-1],bias[layer])
    #h[layer]=abs(w[layer]*h[layer-1])
    #h[layer]=rectlin(w[layer]*h[layer-1])
end
w[L]=ADvariable()
bias[L]=ADvariable()
h[L]=AXplusBias(w[L],h[L-1],bias[L])
ypred=sigmoid(h[L]) # just use for testing, not training
loss=BinaryKullbackLeiblerLossXsigmoidY(ytrain,h[L])
net=EndCode()

#instantiate root node values:
net.value[h[1].index]=data[:,1:BatchSize];
for i=2:L
    net.value[w[i]]=.5*sign(randn(H[i],H[i-1]))/sqrt(H[i-1])
    net.value[bias[i]]=.5*sign(randn(H[i],1))
end

net=compile(net) # compile the DAG and preallocate memory
#gradcheck(net)
println("Training:")
error=Array(Float64,0)
ParsToUpdate=Parameters(net)
meangrad=zeros(TrainingIts)
meanv=zeros(TrainingIts)
velo=NesterovInit(net)
minibatch=1:BatchSize # starting datapoint for the minibatch
for iter=1:TrainingIts
    LearningRate=0.5/(1+iter/500)
    minibatch=GetNewBatch(minibatch,Ntrain)
    net.value[ytrain]=data[:,minibatch] # select batch of data
    ADforward!(net;exclude=[ypred])  # ypred node only needed for prediction, not training error
    ADbackward!(net)
    push!(error,extract(net.value[net.FunctionNode])[1])
    printover("iteration $iter: training loss = $(error[iter])")
    for par in ParsToUpdate
        #GradientDescentUpdate!(net.value[par],net.gradient[par],LearningRate)
        NesterovGradientDescentUpdate!(net.value[par],net.gradient[par],velo[par],LearningRate,iter)
    end
end

using PyPlot
figure(1)
    plot(error); title("training loss")
    figure(2)
    for i=1:20 # plot the reconstructions for a few datapoints
        p=imshow([reshape(net.value[h[1]][:,i],28,28)'  reshape(net.value[ypred.index][:,i],28,28)'],interpolation="none",cmap=ColorMap("gray"))
        display(p)
        println("press key to continue")
        readline(STDIN)
    end

