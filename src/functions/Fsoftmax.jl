#softmax function: f(x)=exp(x)/sum(exp(x))

Fsoftmax(x::Array{Float64,2})=(exp(x)./sum(exp(x),1),[]) # TODO: better to subtract the max of x to make it numerically more stable

function Fsoftmax_inplace(handle,value,auxvalue,x::Array{Float64,2})
    copy!(value,exp(x)./sum(exp(x),1)) # TODO: better to subtract the max of x to make it numerically more stable
end
          
function Dsoftmax(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::Array{Float64,2})
    axpy!(1.0,f_c.*(grad_c-repmat(sum(f_c.*grad_c,1),size(f_c,1),1)),grad_n)
end



if PROC=="GPU"
    
    function Fsoftmax(X::CudaArray)
        expX=CudaArray(Float64,size(X))
        exp!(X,expX) 
        onr=CudaArray(Float64,(1,size(X,1))); fill!(onr,1.0);
        colsum=onr*expX
        onc=CudaArray(Float64,(size(X,1),1)); fill!(onc,1.0);
        out=CudaArray(Float64,size(X))
        vdiv!(1.0,expX,onc*colsum,out)
        free(expX); free(onr); free(colsum); free(onc)
        return (out,[]) # memory leak -- how to free out
    end

    function softmax!(X::CudaArray,out::CudaArray)
        expX=CudaArray(Float64,size(X))
        exp!(X,expX) 
        onr=CudaArray(Float64,(1,size(X,1))); fill!(onr,1.0);
        colsum=onr*expX
        onc=CudaArray(Float64,(size(X,1),1)); fill!(onc,1.0);
        vdiv!(1.0,expX,onc*colsum,out)
        free(expX); free(onr); free(colsum); free(onc)
    end
    export softmax!


    function Fsoftmax_inplace(handle,value,auxvalue,X::CudaArray)
        expX=CudaArray(Float64,size(X))
        exp!(X,expX) 
        onr=CudaArray(Float64,(1,size(X,1))); fill!(onr,1.0);
        colsum=onr*expX
        onc=CudaArray(Float64,(size(X,1),1)); fill!(onc,1.0);
        vdiv!(1.0,expX,onc*colsum,value)
        free(expX); free(onr); free(colsum); free(onc)
    end


    function Dsoftmax(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray)
        tmp=CudaArray(Float64,size(f_c))
        vmult!(1.0,f_c,grad_c,tmp)
        onr=CudaArray(Float64,(1,size(tmp,1))); fill!(onr,1.0);
        colsum=onr*tmp
        onc=CudaArray(Float64,(size(tmp,1),1)); fill!(onc,1.0);
        copy!(tmp,grad_c)
        axpy!(-1.0,onc*colsum,tmp)
        vmult!(1.0,f_c,tmp,tmp)
        axpy!(1.0,tmp,grad_n)
        free(tmp); free(onr); free(onc);
    end
    
end




Derivative[Fsoftmax]=Dsoftmax
Inplace[Fsoftmax]=Fsoftmax_inplace

ADsoftmax(n)=ADFunction(Fsoftmax,n)

softmax(n::ADnode)=ADFunction(Fsoftmax,n)

export softmax
