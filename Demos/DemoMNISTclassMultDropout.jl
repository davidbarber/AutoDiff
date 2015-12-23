using MAT

useproc("GPU") # GPU is much faster than CPU
#useproc("CPU")

Ntrain=60000
Ntest=10000
BatchSize=100
TrainingIts=5000 # number of training iterations
include("loadmnist.jl")
images,label=loadmnist()
label=convert(Array{Int,2},label)
r=randperm(size(images,2))
data=images[:,r]; label=label[r]
class=zeros(10,70000)+0.00001
for i=1:70000
    class[label[i]+1,i]=0.9999
end
traindata=data[:,1:Ntrain]
testdata=data[:,Ntrain+1:Ntrain+Ntest]
trainclass=class[:,1:Ntrain]
testclass=class[:,Ntrain+1:Ntrain+Ntest]

# Construct the DAG function:
H=[784 3000 200 100 50 10] # number of units in each layer

L=length(H) # number of hidden layers
# node indices:
bias=Array(ADnode,L) # weight index
w=Array(ADnode,L) # weight index
h=Array(ADnode,L) # hidden layer index (note that I call the input layer h[1])
#hh=Array(ADnode,L) # hidden layer index (note that I call the input layer h[1])
dropout=Array(ADnode,L) # hidden layer index (note that I call the input layer h[1])
StartCode()
class=ADnode() # node has no parents
x=ADnode()
dropout[1]=ADnode()
h[1]=diagm(dropout[1])*x
for i=2:L-1
    dropout[i]=ADnode()
    w[i]=ADvariable()
    bias[i]=ADvariable()
    h[i]=diagm(dropout[i])*rectlinAXplusBias(w[i],h[i-1],bias[i])
end
w[L]=ADvariable()
h[L]= w[L]*h[L-1]
loss=KLsoftmax(class,h[L]) # the loss we are minimising must be the final node in the graph.
net=EndCode()

# instantiate parameter nodes and inputs:
net.value[x]=traindata[:,1:BatchSize]
net.value[class]=trainclass[:,1:BatchSize]
for i=2:L
    net.value[w[i]]=sign(randn(H[i],H[i-1]))/sqrt(H[i-1])
end
net.value[dropout[1]]=ones(H[1])
for i=2:L-1
    net.value[bias[i]]=zeros(H[i])
    net.value[dropout[i]]=ones(H[i],1)
end

net=compile(net) # compile the DAG and preallocate memory
#gradcheck(net)

@gpu CUDArt.init([0])
@gpu net=convert(net,"GPU")

tmp=net.value[1]

ADforward!(net)

dropoutprob=zeros(L)
dropoutprob[1]=0
#for i=2:L-1
#    dropoutprob[i]=0.2
#end
dropoutprob[2]=0.5
println("Training: using $(net.gpu==true? "GPU" : "CPU") ")
error=Array(Float64,0)
ParsToUpdate=Parameters(net)
meangrad=zeros(TrainingIts)
meanv=zeros(TrainingIts)
velo=NesterovInit(net)
minibatchstart=1 # starting datapoint for the minibatch
for iter=1:TrainingIts
    LearningRate=0.1/(1+iter/1500)
    minibatchstart,minibatch=GetBatch(minibatchstart,BatchSize,Ntrain)
    net.value[x]=cArray(PROC,traindata[:,minibatch]) # select batch of data
    net.value[class]=cArray(PROC,trainclass[:,minibatch]) # select batch of data
    for i=2:L-1
        net.value[dropout[i]]=cArray(PROC,1.0*(rand(H[i],1).>dropoutprob[i]))
    end
    ADforward!(net)
    ADbackward!(net)
    push!(error,extract(net.value[net.FunctionNode])[1])
    printover("iteration $iter: training loss = $(error[iter])")
    for par in ParsToUpdate
        #GradientDescentUpdate!(net.value[par],net.gradient[par],LearningRate)
        NesterovGradientDescentUpdate!(net.value[par],net.gradient[par],velo[par],LearningRate,iter)
    end
end
#For prediction scaling the weights is equivalent to using a fixed dropout value:
for i=1:L-1
    net.value[dropout[i]]=cArray((1-dropoutprob[i])*ones(H[i],1))
end
ADforward!(net)

net=convert(net,"CPU") # convert to CPU for easy results analysis

net.value[h[L]]
classpredtrain=argcolmax(softmax(net.value[h[L]]))
classtrain=argcolmax(net.value[class])
println("\nlast batch train accuracy = $(mean(classpredtrain.==classtrain))")

println("Testing: using $(net.gpu==true? "GPU" : "CPU") ")
# evaluate the test predictions:
net.value[x]=testdata
ForwardPassList!(net,ExcludeNodes=[class])
# ADforward! by default computes values inplace and does no additional memory allocation on the graph. If we change the input value dimensions (which happens here since there are a different number of test points to train points), we need to reallocatte memory based on the shapes of the new values on the graph. This only needs to be done once.
ADforward!(net,AllocateMemory=true) # reallocate since shape of net.value[x] changed

classpredtest=argcolmax(softmax(net.value[h[L]]))
classtest=argcolmax(testclass)
println("test accuracy = $(mean(classpredtest.==classtest))")


#end
