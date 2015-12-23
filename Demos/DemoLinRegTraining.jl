# Linear Regression with L1 penalty

StartCode()
X=ADnode()
w=ADvariable()
Y=ADnode()
loss=meanSquareLoss(elu(X*w),Y)+0.1*meanAbs(w)
net=EndCode() # defines the graph

# instantiate parameter nodes and inputs:

D=20 # dimension of weight vector
N=300 # number of datapoints
truew=randn(D,1).*(rand(D,1).>0.8)
net.value[w]=truew
net.value[X]=randn(N,D)
#net.value[Y]=cArray(randn(N,1))
net.value[Y]=net.value[X]*net.value[w] # make a realisable problem
net.value[w]=randn(D,1) # start with a random w

net=compile(net) # compile and preallocate memory

@gpu CUDArt.init([0]) # let the user do device management
@gpu net=convert(net,"GPU")
gradcheck(net)

# Training:
ParsToUpdate=Parameters(net)
nupdates=500
LearningRate=0.1
error=Array(Float64,0)
velo=NesterovInit(net) # Nesterov velocity
for i=1:nupdates
    ADforward!(net)
    ADbackward!(net)
    push!(error,extract(net.value[net.FunctionNode])[1])
    printover("iteration $i: training loss = $(error[i])")
    for par in ParsToUpdate
        #GradientDescentUpdate!(net.value[par],net.gradient[par],LearningRate)
        NesterovGradientDescentUpdate!(net.value[par],net.gradient[par],velo[par],LearningRate,i)
    end
end

@gpu CUDArt.device_synchronize()

println("\ntrue weights   ",round(100*truew')/100)
println("learned weights",round(100*extract(net.value[w])')/100)





