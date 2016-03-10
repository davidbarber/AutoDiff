function GetBatch(minibatchstart,BatchSize,Ntrain)

    if minibatchstart>Ntrain-BatchSize+1
        minibatchstart=Ntrain-BatchSize+1
    end
    minibatch=minibatchstart:minibatchstart+BatchSize-1
    minibatchstart=minibatchstart+BatchSize
    if minibatchstart>Ntrain
        minibatchstart=1
    end
    return minibatchstart,minibatch
end
export GetBatch

function GetNewBatch(OldMiniBatch,Ntrain)
    MiniBatch=rem(OldMiniBatch[end]+1:OldMiniBatch[end]+length(OldMiniBatch),Ntrain)
    MiniBatch[MiniBatch.==0]=Ntrain
    return MiniBatch
end
export GetNewBatch




# Standard Gradient Descent:

function GradientDescentUpdate!(x,grad,LearningRate)
    # xnew=x-LearningRate*grad
    axpy!(-LearningRate,grad,x)
end
export GradientDescentUpdate!


# Gradient Descent with Momentum:

function GradientDescentMomentumUpdate!(x,grad,avgrad,Momentum,LearningRate)
    # avgrad_new=alpha*avgrad_old +(1-alpha)*avgrad
    # xnew=x-LearningRate*avgrad
    # where alpha=Momentum
    scale!(Momentum,avgrad)
    axpy!(1-Momentum,grad,avgrad)
    axpy!(-LearningRate,avgrad,x)
end
export GradientDescentMomentumUpdate!


function GradientDescentMomentumInit(net)
    avgrad=Array(Any,length(net.node))
    for par in Parameters(net)
        avgrad[par]=cArray(net.gpu,zeros(size(net.value[par]))) # initial average gradient
    end
    return avgrad
end
export GradientDescentMomentumInit



# Nesterov Accelerated Gradient Descent:

function NesterovInit(net)
    velo=Array(Any,length(net.node)) # needed for Nesterov
    for par in Parameters(net)
#        velo[par]=cArray(size(net.value[par])); fill!(velo[par],0.0) # Nesterov velocity
        velo[par]=cArray(net.gpu,zeros(net.eltype,size(net.value[par]))) # Nesterov velocity
    end
    return velo
end
export NesterovInit


function NesterovGradientDescentUpdate!(thetaP,gradP,v,LearningRate,t)
    # A standard Nesterov Update (see for example appendix A.1 in mlr.org/proceedings/papers/v28/sutskever13-supp.pdf) uses
    # v_new = mu*v_old-epsilon*gradient(theta_old+mu*v_old)
    # theta_new=theta_old+v_new
    # However, we can reparameterise this using
    # thetaP = theta + v
    # and write an update for thetaP
    # v_new = mu*v_old-epsilon*gradient(thetaP_old)
    # thetaP_new = thetaP_old-mu_old*v_old+v_new*(1+mu_new)
    # Formally, we would then need to retransform back to get the value for theta.
    # However, as we converge, mu goes to 1 and the gradient goes to zero.
    # This means that thetaP tends to theta anyway as we converge to the optimum.

    mu_new=min(0.99,1-3/(t+5))
    mu_old=min(0.99,1-3/(t+4))
    scale!(mu_old,v)
    axpy!(mu_new,v,thetaP)
    axpy!(-LearningRate,gradP,v)
    axpy!(-(1.0+mu_new)*LearningRate,gradP,thetaP)

end
export NesterovGradientDescentUpdate!

