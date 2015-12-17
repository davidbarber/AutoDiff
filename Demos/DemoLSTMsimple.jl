#function DemoLSTMsimple(PlotResults=true)
# Simple memory problem:
# If either of the first two inputs in x[t] is 1, then output to y[t] the previous memory m[t-1] and update the memory m[t] to store x[t]
# This can be achieved using a simple LSTM style approach.

PlotResults=false

T=400
m=Array(ADnode,T)
alpha=Array(ADnode,T)
beta=Array(ADnode,T)
x=Array(ADnode,T)
y=Array(ADnode,T)
yout=Array(ADnode,T)
loss=Array(ADnode,T)

StartCode()
Win=ADvariable()
Wout=ADvariable()
Bin=ADvariable()
Bout=ADvariable()
x[1]=ADnode()
alpha[1]=sigmoid(Win*x[1])
m[1]=1*x[1]
y[1]=ADnode()
beta[1]=sigmoid(Wout*x[1])
for t=2:T
    x[t]=ADnode()
    y[t]=ADnode()
    alpha[t]=sigmoid(Win*x[t]+Bin)
    m[t]=alpha[t]*x[t]+(1-alpha[t])*m[t-1]
    beta[t]=sigmoid(Wout*x[t]+Bout)
    yout[t]=beta[t]*m[t-1]
    #loss[t]=meanSquareLoss(yout[t],y[t])
    loss[t]=BinaryKullbackLeiblerLoss(y[t],yout[t])
end
totalloss=mean(loss[2:T])
net=EndCode() # defines the graph

# instantiate parameter nodes and inputs:
X=4
# parameters:
net.value[Win]=0.01*randn(1,X)
net.value[Wout]=0.01*randn(1,X)
net.value[Bin]=0.01*randn(1,1)
net.value[Bout]=0.01*randn(1,1)

# training data:
# make close to zero or 1 (strictly 0 or 1 can give NAN in KL loss)
net.value[x[1]]=zeros(X,1)+0.001; net.value[x[1]][1]=0.999;
net.value[y[1]]=zeros(X,1)+0.001

xstore=copy(net.value[x[1]])
for t=2:T
    net.value[x[t]]=0.001+0.998*(rand(X,1).>0.5)
    net.value[x[t]][1]=0.001+0.998*(rand()>0.95)

    if sum(extract(net.value[x[t]][1]))>0.5
        net.value[y[t]]=xstore
        xstore=net.value[x[t]]
    else
        net.value[y[t]]=zeros(X,1)+0.001
    end
end

net=compile(net) # compile and preallocate memory
@gpu CUDArt.init([0])
@gpu net=convert(net,"GPU")


#gradcheck(net;showgrad=true) # use a small number of datapoints and small network to check the gradient, otherwise


# Training:
ParsToUpdate=Parameters(net)
nupdates=100
error=Array(Float64,0)
    velo=NesterovInit(net) # Nesterov velocity
    for i=1:nupdates
    LearningRate=0.5
    ADforward!(net)
    ADbackward!(net)
    push!(error,extract(net.value[net.FunctionNode])[1])
    printover("iteration $i: training loss = $(error[i])")
    for par in ParsToUpdate
        #GradientDescentUpdate!(net.value[par],net.gradient[par],LearningRate)
        NesterovGradientDescentUpdate!(net.value[par],net.gradient[par],velo[par],LearningRate,i)
    end
end
println()

if PlotResults==true
    yy=zeros(X,T); yyout=zeros(X,T); xx=zeros(X,T)
    for t=2:T
        yy[:,t]=net.value[y[t]]; yyout[:,t]=net.value[yout[t]]
        xx[:,t]=net.value[x[t]];
    end

    fig,ax=PyPlot.subplots(3,1,sharex=true)
    ax[1,1][:imshow](xx,interpolation="none");ax[1,1][:set_aspect](15)
    ax[2,1][:imshow](yy,interpolation="none"); ax[2,1][:set_aspect](15);
    ax[3,1][:imshow](yyout,interpolation="none"); ax[3,1][:set_aspect](15)
    ax[1,1][:set_title]("x: train input")
    ax[2,1][:set_title]("y: train output")
    ax[3,1][:set_title]("yout: net predicted output given x")
    fig[:canvas][:draw]()

    figure(); plot(error); title("training loss")
end
#end
