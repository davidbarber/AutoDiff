#f(p,x)=KL(p,softmax(x))


function FKLsoftmax(p::Array{Float64,2},x::Array{Float64,2})
    logZ=log(sum(exp(x),1))
    return ([(sum(p.*(log(p)-x))+sum(logZ))/length(x)],logZ)
end

function FKLsoftmax_inplace(value,auxvalue,p::Array{Float64,2},x::Array{Float64,2})
    logZ=log(sum(exp(x),1))
    fill!(value,(sum(p.*(log(p)-x))+sum(logZ))/length(x))
    copy!(auxvalue,logZ)
end


function DKLsoftmax(derivativeIDX,f_c,faux_c,grad_c,grad_n,p::Array{Float64,2},x::Array{Float64,2})
    if derivativeIDX==1
        axpy!(1.0/length(x),grad_c.*(1+log(p)-x+repmat(faux_c,size(x,1),1)),grad_n) 
    elseif derivativeIDX==2
        axpy!(1.0,grad_c.*(softmax(x)-p)./length(x),grad_n)
    end
end

if GPU
 
    function FKLsoftmax(p::CudaArray,x::CudaArray)
        tmp=CudaArray(Float64,size(x))
        exp!(x,tmp)
        onr=CudaArray(Float64,(1,size(x,1))); fill!(onr,1.0)
        logZ=onr*tmp
        log!(logZ,logZ)
        log!(p,tmp)
        axpy!(-1.0,x,tmp)
        out=CudaArray(Float64,(1,1)); fill!(out,0.0)
        axpy!(1./length(x),tmp,out)
        axpy!(1./length(x),sum(logZ),out)
        free(tmp)
        return (out,logZ)
        #return ([(sum(p.*(log(p)-x))+sum(log(Z)))/length(x)],aux)
    end
 
    function FKLsoftmax_inplace(value,auxvalue,p::CudaArray,x::CudaArray)
        tmp=CudaArray(Float64,size(x))
        exp!(x,tmp)
        onr=CudaArray(Float64,(1,size(x,1))); fill!(onr,1.0);
        logZ=CudaArray(Float64,(1,size(x,2)))
        log!(onr*tmp,logZ)
        log!(p,tmp)
        axpy!(-1.0,x,tmp)
        vmult!(1.0,p,tmp,tmp)
        fill!(value,0.0)
        axpy!(1./length(x),sum(tmp),value)
        axpy!(1./length(x),sum(logZ),value)
        copy!(auxvalue,logZ)
        free(onr); free(logZ); free(tmp)
        #return ([(sum(p.*(log(p)-x))+sum(log(Z)))/length(x)],aux)
    end
    
    
    function DKLsoftmax(derivativeIDX,f_c,faux_c,grad_c,grad_n,p::CudaArray,x::CudaArray)
        if derivativeIDX==1
            #axpy!(1.0/length(x),grad_c.*(1+log(p)-x+repmat(faux_c,size(x,1),1)),grad_n) 
            tmp=CudaArray(Float64,size(x)); fill!(tmp,1.0)
            logp=CudaArray(Float64,size(p))
            log!(p,logp)
            axpy!(1.0,logp,tmp)
            axpy!(-1.0,x,tmp)
            onc=CudaArray(Float64,size(x,1),1); fill!(onc,1.0)
            axpy!(1.0,onc*faux_c,tmp)
            alphaaxpy!(1./length(x),grad_c,tmp,grad_n)
            free(tmp); free(onc); free(logp)
        elseif derivativeIDX==2
            #axpy!(1.0,grad_c.*(softmax(x)-p)./length(x),grad_n)
            tmp=CudaArray(Float64,size(x))
            softmax!(x,tmp)
            axpy!(-1.0,p,tmp)
            alphaaxpy!(1.0/length(x),grad_c,tmp,grad_n)
            free(tmp)
        end
    end
    


end


Derivative[FKLsoftmax]=DKLsoftmax
Inplace[FKLsoftmax]=FKLsoftmax_inplace

KLsoftmax(p::ADnode,x::ADnode)=ADnode(FKLsoftmax,[p x])

export KLsoftmax
