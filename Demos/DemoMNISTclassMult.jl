using MAT

Ntrain=60000
Ntest=10000
BatchSize=100
TrainingIts=500 # number of Nesterov updates
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
H=[784 100 50 10] # number of units in each layer
L=length(H) # number of hidden layers
# node indices:
w=Array(ADnode,L) # weight index
h=Array(ADnode,L) # hidden layer index (note that I call the input layer h[1])

StartCode()
x=h[1]=ADnode()
class=ADnode() # node has no parents
for i=2:L-1
    w[i]=ADvariable()
    h[i]=sigmoid(w[i]*h[i-1])
end
w[L]=ADvariable()
h[L]=w[L]*h[L-1]
loss=KLsoftmax(class,h[L]) # the loss we are minimising must be the final node in the graph.
net=EndCode()

# instantiate parameter nodes and inputs:
net.value[x]=traindata[:,1:BatchSize]
net.value[class]=trainclass[:,1:BatchSize]
for i=2:L
    net.value[w[i]]=sign(randn(H[i],H[i-1]))/sqrt(H[i-1])
end

net=compile(net) # compile the DAG and preallocate memory

println("Training:")
error=Array(Float64,0)
ParsToUpdate=Parameters(net)
meangrad=zeros(TrainingIts)
meanv=zeros(TrainingIts)
velo=NesterovInit(net)
minibatchstart=1 # starting datapoint for the minibatch
for iter=1:TrainingIts
    LearningRate=0.5/(1+iter/500)
    minibatchstart,minibatch=GetBatch(minibatchstart,BatchSize,Ntrain)
    net.value[x]=traindata[:,minibatch] # select batch of data
    net.value[class]=trainclass[:,minibatch] # select batch of data
    ADforward!(net)
    ADbackward!(net)
    push!(error,extract(net.value[net.FunctionNode])[1])
    println("iteration $iter: training loss = $(error[iter])")
    for par in ParsToUpdate
            #GradientDescentUpdate!(net.value[par],net.gradient[par],LearningRate)
        NesterovGradientDescentUpdate!(net.value[par],net.gradient[par],velo[par],LearningRate,iter)
    end
end

classpredtrain=argcolmax(softmax(net.value[h[L]]))
classtrain=argcolmax(net.value[class])
println("last batch train accuracy = $(mean(classpredtrain.==classtrain))")


println("Training:")
# evaluate the test predictions:
net.value[x]=traindata
ForwardPassList!(net,ExcludeNodes=[class])
# ADforward! by default computes values inplace. If we change the input value dimensions (which happens here since there are a different number of test points to train points), we need to do an out-of-place forward pass in order to compute the shapes of the values on the graph. This only needs to be done once.
@gpu (net=convert(net,"CPU"); ADforward!(net,AllocateMemory=true); net=convert(net,"GPU")) # net computes the loss (which depends on the class), so ignore all nodes that depend on the class
@cpu ADforward!(net,AllocateMemory=true)
ADforward!(net) # net computes the loss (which depends on the class), so ignore all nodes that depend on the class
classpredtrain=argcolmax(softmax(net.value[h[L]]))
classtrain=argcolmax(trainclass)
println("train accuracy = $(mean(classpredtrain.==classtrain))")


println("Testing:")
# evaluate the test predictions:
net.value[x]=testdata
ForwardPassList!(net,ExcludeNodes=[class loss])
@gpu (net=convert(net,"CPU"); ADforward!(net,AllocateMemory=true); net=convert(net,"GPU")) # net computes the loss (which depends on the class), so ignore all nodes that depend on the class
@cpu ADforward!(net,AllocateMemory=true)
ADforward!(net) # net computes the loss (which depends on the class), so ignore all nodes that depend on the class
classpredtest=argcolmax(softmax(net.value[h[L]]))
classtest=argcolmax(testclass)
println("test accuracy = $(mean(classpredtest.==classtest))")

#end