function DemoLSTMsimplefunction(PlotResults=true)

# Simple memory problem using an LSTM style memory:
# If either of the first two inputs in x[t] is 1, then output to y[t] the previous memory m[t-1] and update the memory m[t] to store x[t]
# This can be achieved using a simple LSTM style approach.

# This is a version showing how to define a function and call this within the code

function LSTM(t)
    x[t]=ADnode()
    y[t]=ADnode()
    #alpha[t]=sigmoid(Win*x[t]+Bin)
    alpha[t]=sigmoidAXplusBias(Win,x[t],Bin)
    m[t]=alpha[t]*x[t]+(1-alpha[t])*m[t-1]
    m[t]=alpha[t]*x[t]+(1-alpha[t])*m[t-1]
    #beta[t]=sigmoid(Wout*x[t]+Bout)
    beta[t]=sigmoidAXplusBias(Wout,x[t],Bout)
    yout[t]=beta[t]*m[t-1]
end

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
    LSTM(t)
    loss[t]=meanSquareLoss(yout[t],y[t])
    #loss[t]=BinaryKullbackLeiblerLoss(y[t],yout[t])
end
totalloss=mean(loss[2:T])
net=EndCode() # defines the graph

# instantiate parameter nodes and inputs:
X=3
# parameters:
net.value[Win]=0.001*randn(1,X)
net.value[Wout]=0.001*randn(1,X)
net.value[Bin]=0.01*randn(1,1)
net.value[Bout]=0.01*randn(1,1)

# training data:
# make close to zero or 1 (strictly 0 or 1 can give NAN in KL loss)
net.value[x[1]]=zeros(X,1)+0.001; net.value[x[1]][1]=0.999
net.value[y[1]]=zeros(X,1)+0.001
xstore=copy(net.value[x[1]])
for t=2:T
    net.value[x[t]]=0.001+0.998*(rand(X,1).>0.5)
    net.value[x[t]][1]=0.001+0.998*(rand()>0.95)
    if sum(net.value[x[t]][1])>0.5
        net.value[y[t]]=xstore
        xstore=net.value[x[t]]
    else
        net.value[y[t]]=zeros(X,1)+0.001
    end
end

net=compile(net) # compile and preallocate memory
#gradcheck(net;showgrad=true) # only use a small network to check the gradient, otherwise this will take a long time



# Training:
nupdates=1000
ParsToUpdate=Parameters(net)
error=Array(Float64,0)
    v=NesterovInit(net) # Nesterov velocity
    for i=1:nupdates
    LearningRate=5.5
    ADforward!(net)
    ADbackward!(net)
    push!(error,extract(net.value[net.FunctionNode])[1])
    printover("iteration $i: training loss = $(error[i])")
#        print '{0}\r'.format(x),
    for par in ParsToUpdate
        #GradientDescentUpdate!(net.value[par],net.gradient[par],LearningRate)
        NesterovGradientDescentUpdate!(net.value[par],net.gradient[par],v[par],LearningRate,i/500)
    end
end

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

end
