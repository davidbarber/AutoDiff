# Leaky Integrate and Fire Demo -- see section 26.5.4 of the Bayesian Reasoning and Machine Learning Textbook
# David Barber UCL 2015
# Uses the dbAutoDiff Julia package

PlotResults=true
useproc("CPU") # CPU is faster than GPU for this demo

T=100 # number of timesteps
a=Array(ADnode,T)
p=Array(ADnode,T)
v=Array(ADnode,T)
loss=Array(ADnode,T)

ThetaRest=-5.0
ThetaFired=-15.0
alpha=0.95

StartCode()  # This is where you start the code for your problem
W=ADVariable() # parameter that we want to learn
a[1]=ADconst() # A variable that we will not take the derivative of
p[1]=sigmoid(a[1])
v[1]=ADconst()
for t=2:T
    v[t]=ADconst()
    a[t]=(alpha*a[t-1]+W*v[t]+ThetaRest*(1-alpha)).*(1-v[t-1])+v[t-1]*ThetaFired
    p[t]=sigmoid(a[t]) # probability that neurons fire
    loss[t]=BinaryKullbackLeiblerLoss(v[t],p[t])
end
totalloss=mean(loss[2:T])
net=EndCode() # defines the graph

# instantiate parameter nodes and inputs:
N=20 # number of neurons at each time step
net.value[W]=0.001*randn(N,N)

# training data:
net.value[a[1]]=zeros(N,1)
for t=1:T
    net.value[v[t]]=0.99*(rand(N,1).>0.85)+0.001 # bound slighty away from 0/1 so that no NAN occur
end

net=compile(net) # compile and preallocate memory
#=
@gpu CUDArt.init([0])
@gpu net=convert(net,"GPU")
#gradcheck(net) # only use a small network to check the gradient, otherwise this will take a long time


# Training:
println("Training: using $(net.gpu==true? "GPU" : "CPU") ")
nupdates=1000
parstoupdate=Parameters(net)
error=Array(Float64,0)
    velo=NesterovInit(net) # Nesterov velocity
    LearningRate=5.5
    for i=1:nupdates
    ADforward!(net)
    ADbackward!(net)
    push!(error,extract(net.value[net.FunctionNode])[1])
    printover("iteration $i: training loss = $(error[i])")
    for par in parstoupdate
        #GradientDescentUpdate!(net.value[par],net.gradient[par],LearningRate)
        NesterovGradientDescentUpdate!(net.value[par],net.gradient[par],velo[par],LearningRate,i/500)
    end
end


trainvalue=zeros(N,T)
testprob=zeros(N,T)
actpot=zeros(N,T)


@gpu net=convert(net,"CPU")

for t=1:T
    trainvalue[:,t]=net.value[v[t]]
    testprob[:,t]=net.value[p[t]]
    actpot[:,t]=net.value[a[t]]
end

if PlotResults
    figure(); imshow(trainvalue, interpolation="none"); title("training firings v_i(t)")
    figure(); imshow(testprob, interpolation="none"); title("test firing probability p_i(t)")
    figure(); imshow(actpot, interpolation="none"); title("test membrane potential a_i(t)")
    figure(); plot(error); title("training loss against iteration")
end
=#
