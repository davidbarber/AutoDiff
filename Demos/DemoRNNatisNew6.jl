#function DemoRNNatisNew()

    
    using dbAutoDiff
    using PyPlot
    using MAT
    
    include("loadsave.jl")
    include("setvalue2!.jl")

    dotraining=true
    
    initialiseparameters=true
    Ntrain=2

    LogFile="LogFile"
    include("getatisdata.jl")
    tokens,lexicon=getatisdata(Ntrain)
    Ntrain=length(tokens)
    word=lexicon

    seq=tokens

    if true  # get rid of tiny length sequences (some are length 1!)
    oldseq=deepcopy(seq)
    goodseq=BitArray(Ntrain,1)
    fill!(goodseq,false)
    counter=0
    for i=1:Ntrain
        if length(seq[i])>5
            counter+=1
            #seq[counter]=oldseq[i][1:min(10,length(seq[i]))]
            seq[counter]=oldseq[i]
        end
    end
    seq=seq[1:counter]
    end

    Ntrain=length(seq)

    #BatchSize=Ntrain
    BatchSize=10

    W=length(word) # total number of words in dictionary
    Tmax=maximum(map((x)->length(x),seq))

    X=10 # size of the word vec representation

    loss=zeros(Int,1,Tmax)
    includedloss=zeros(Int,1,Tmax)
    lossIDX=zeros(Int,1,Tmax)
    x=zeros(Int,1,Tmax)
    xvec=zeros(Int,1,Tmax)
    xh=zeros(Int,1,Tmax)
    h=zeros(Int,1,Tmax)
    hout=zeros(Int,1,Tmax)
    a1=zeros(Int,1,Tmax)
    a2=zeros(Int,1,Tmax)
    a3=zeros(Int,1,Tmax)
    a4=zeros(Int,1,Tmax)
    a5=zeros(Int,1,Tmax)
    a6=zeros(Int,1,Tmax)
    a7=zeros(Int,1,Tmax)
    a8=zeros(Int,1,Tmax)
    a9=zeros(Int,1,Tmax)
    a10=zeros(Int,1,Tmax)
    a11=zeros(Int,1,Tmax)
    a12=zeros(Int,1,Tmax)
    a13=zeros(Int,1,Tmax)
    a14=zeros(Int,1,Tmax)
    a15=zeros(Int,1,Tmax)
    a16=zeros(Int,1,Tmax)

    target=zeros(Int,1,Tmax)

    node=Array(ADnode,10000) # nodes
    c=0 # node counter
    node[invmaxlength=c+=1]=ADnode([])

    # tied weights:
    node[wordmatrix=c+=1]=ADnode([];returnderivative=true)
    node[Wstart=c+=1]=ADnode([];returnderivative=true)
    node[W1=c+=1]=ADnode([];returnderivative=true)
    node[W2=c+=1]=ADnode([];returnderivative=true)
    node[W3=c+=1]=ADnode([];returnderivative=true)
    node[W4=c+=1]=ADnode([];returnderivative=true)
    node[W5=c+=1]=ADnode([];returnderivative=true)
    node[W6=c+=1]=ADnode([];returnderivative=true)
    node[W7=c+=1]=ADnode([];returnderivative=true)
    node[W8=c+=1]=ADnode([];returnderivative=true)
    node[W9=c+=1]=ADnode([];returnderivative=true)
    node[W10=c+=1]=ADnode([];returnderivative=true)
    node[W11=c+=1]=ADnode([];returnderivative=true)
    node[W12=c+=1]=ADnode([];returnderivative=true)
    node[W13=c+=1]=ADnode([];returnderivative=true)
    node[W14=c+=1]=ADnode([];returnderivative=true)
    node[W15=c+=1]=ADnode([];returnderivative=true)
    node[W16=c+=1]=ADnode([];returnderivative=true)
    node[Wout=c+=1]=ADnode([];returnderivative=true)

    node[x[1]=c+=1]=ADnode([]) # inputs are word indices
    node[xvec[1]=c+=1]=ADnode([wordmatrix,x[1]],Fgetcol) # wordvecs
    #node[h[1]=c+=1]=ADnode([Wstart,xvec[1]],FAx)
    node[h[1]=c+=1]=ADnode([Wstart,xvec[1]],FshiftedsigmaAx)
    node[lossIDX[1]=c+=1]=ADnode([])
    node[target[1]=c+=1]=ADnode([])
    node[hout[1]=c+=1]=ADnode([Wout,h[1]],FAx)
    node[loss[1]=c+=1]=ADnode([target[1] hout[1]],FMultLogisticLoss)
    node[includedloss[1]=c+=1]=ADnode([loss[1] lossIDX[1]],Fxy)

    Ftransfer=FshiftedsigmaAx
    #Ftransfer=FrectlinAx
    for t=2:Tmax
        node[target[t]=c+=1]=ADnode([])
        node[x[t]=c+=1]=ADnode([]) # inputs are word indices
        node[xvec[t]=c+=1]=ADnode([wordmatrix,x[t]],Fgetcol) # wordvecs
        node[xh[t]=c+=1]=ADnode([xvec[t] h[t-1]],Fvcat)

        node[a1[t]=c+=1]=ADnode([W1,xh[t]],Ftransfer)
        node[a2[t]=c+=1]=ADnode([W2,xh[t]],Ftransfer)
        node[a3[t]=c+=1]=ADnode([W3,xh[t]],Ftransfer)
        node[a4[t]=c+=1]=ADnode([W4,xh[t]],Ftransfer)
        node[a5[t]=c+=1]=ADnode([W5,xh[t]],Ftransfer)
        node[a6[t]=c+=1]=ADnode([W6,xh[t]],Ftransfer)
        node[a7[t]=c+=1]=ADnode([W7,xh[t]],Ftransfer)
        node[a8[t]=c+=1]=ADnode([W8,xh[t]],Ftransfer)
        node[a9[t]=c+=1]=ADnode([W9,xh[t]],Ftransfer)
        node[a10[t]=c+=1]=ADnode([W10,xh[t]],Ftransfer)
        node[a11[t]=c+=1]=ADnode([W11,xh[t]],Ftransfer)
        node[a12[t]=c+=1]=ADnode([W12,xh[t]],Ftransfer)
        node[a13[t]=c+=1]=ADnode([W13,xh[t]],Ftransfer)
        node[a14[t]=c+=1]=ADnode([W14,xh[t]],Ftransfer)
        node[a15[t]=c+=1]=ADnode([W15,xh[t]],Ftransfer)
        node[a16[t]=c+=1]=ADnode([W16,xh[t]],Ftransfer)

        (h[t],c)=mapreduce!(Fsoftmax,Fvcat,[a1[t] a2[t] a3[t] a4[t] a5[t] a6[t] a7[t] a8[t] a9[t] a10[t] a11[t] a12[t] a13[t] a14[t] a15[t] a16[t]] ,node)

#        (h[t],c)=mapreduce!(Ftanh,Fvcat,[a1[t] a2[t] a3[t] a4[t] a5[t] a6[t] a7[t] a8[t] a9[t] a10[t] a11[t] a12[t] a13[t] a14[t] a15[t] a16[t]] ,node)

        node[hout[t]=c+=1]=ADnode([Wout,h[t]],FAx)
        node[loss[t]=c+=1]=ADnode([target[t] hout[t]],FMultLogisticLoss)
        node[lossIDX[t]=c+=1]=ADnode([]) # binary loss indicator (whether to include this timepoint -- helps deal with sequences of differing lengths
        node[includedloss[t]=c+=1]=ADnode([loss[t] lossIDX[t]],Fxy)
    end

    (unscaledtotalloss,c)=mapreduce!(Fx,Fxpy,includedloss[1:Tmax-1],node)
    node[totalloss=c+=1]=ADnode([unscaledtotalloss,invmaxlength],Fxy)
    # total number of nodes in the net
    node=node[1:c]

    H1=20; H=16*H1; p=1.1;
    if initialiseparameters
        value=Array(Any,c) # function values on the nodes
        #value[h[1]]=ones(H,1)/H1
        value[W1]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W2]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W3]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W4]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W5]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W6]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W7]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W8]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W9]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W10]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W11]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W12]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W13]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W14]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W15]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[W16]=p*sign(randn(H1,X+H))/sqrt(H1)
        value[Wstart]=p*sign(randn(H,X))/sqrt(H)

        value[Wout]=p*sign(randn(W,H))/sqrt(W)

        value[wordmatrix]=0*sign(randn(X,W))/sqrt(X)
        value[invmaxlength]=1.#1/Tmax; #(Tmax*BatchSize)
        #value[invmaxlength]=1.
    end


    
    # instantiate the inputs for this sequence:
    sc=1 # just do this for sequence 1 to see if it compiles ok:
    setvalue2!(value,seq[sc],h,x,lossIDX,target,Tmax,W,H1)
    (value,auxvalue,gradient,net)=compile(value,node) # compile and preallocate memory

if dotraining
    
    gradcheck(value,net;showgrad=false)

    # Nesterov Training:
    TrainingIts=5
    #TrainingIts=2
    parstoupdate=find(map(x->x.returnderivative,net.node)) # node indices that are parameters
    er=zeros(TrainingIts)
    println("Batch Nesterov with decaying learning rate")
    tic()
    minibatchstart=1 # starting datapoint for the minibatch
    nesterov=0.0*deepcopy(gradient)
    valueold=deepcopy(value)
    oldnesterov=deepcopy(nesterov)

    totalgradient=deepcopy(gradient) # ensure totalgradient exists outside loop
        cumtotalloss=0
    for t=1:TrainingIts
        if minibatchstart>Ntrain-BatchSize+1
            minibatchstart=Ntrain-BatchSize+1
        end
        minibatch=minibatchstart:minibatchstart+BatchSize-1
        minibatchstart=minibatchstart+BatchSize
        if minibatchstart>Ntrain
            minibatchstart=1
        end

        # Nesterov Accelerated Gradient update:
        mu=1-3/(t/10+5);
        if t>1
            epsilon=1000.10/(1+t/100)
        else
            epsilon=1000.10
        end
        #epsilon=max(0.01,epsilon)

        copy!(valueold,value)
        copyind!(oldnesterov,nesterov,parstoupdate)
        copyind!(value,value+mu*oldnesterov,parstoupdate)

        gradcounter=0
        cumtotalloss=0

        for sc=minibatch
            gradcounter+=1
            # instantiate the inputs for this sequence:
            setvalue2!(value,seq[sc],h,x,lossIDX,target,Tmax,W,H1)
            ADeval!(value,net;auxvalue=auxvalue,gradient=gradient)

            if gradcounter==1
                totalgradient=deepcopy(gradient)
                cumtotalloss=deepcopy(value[totalloss])
            else
                totalgradient+=gradient
                cumtotalloss+=value[totalloss]
            end
        end
        totalgradient=totalgradient/(Tmax*BatchSize)
        er[t]=cumtotalloss/(Tmax*BatchSize)

        #for par=parstoupdate
        #    println("$(par) : $(mean(abs(totalgradient[par])))")
        #end

        #copyind!(nesterov,mu*oldnesterov-epsilon*gradient,parstoupdate)
        copyind!(nesterov,mu*oldnesterov-epsilon*totalgradient,parstoupdate)
        copyind!(value,valueold+nesterov,parstoupdate)


        println("[$(t) seq=$sc ]Loss = $(er[t]),  learning rate=$epsilon, momentum =$(mu)")
        #if er[t]<0.0000001
          #  break
        #end

        #file=open(LogFile,"a")
        #write(file,"[$(t) seq=$sc ]Loss = $(value[totalloss]), learning rate=$epsilon\n")
        #close(file)
        #plot(er)
        if mod(t,100)==1
            @savevars("results",value);p=plot(er);display(p);
        end
    end
    @savevars("results",value);p=plot(er);display(p);

    toc()
    end

    if true
        loadvars("results")

        SentenceRep=Array(Any,Ntrain)
        for sc=1:Ntrain
            println("\n-------------------------------------------------")
            println("sequence $sc :")
            setvalue2!(value,seq[sc],h,x,lossIDX,target,Tmax,W,H1)
            ADeval!(value,net;auxvalue=auxvalue,gradient=gradient)
            counter=0
            println("training:")
            for t=1:length(seq[sc])
                counter+=1
                print("$(word[seq[sc][t]]) ")
            end
            println("\nprediction:")
            print("$(word[seq[sc][1]]) ")
            for t=1:length(seq[sc])-1
                (val,ind)=findmax(softmax(value[hout[t]]))
                print("$(word[ind]) ")
            end
            SentenceRep[sc]=value[h[length(seq[sc])]]
        end


        dist=zeros(Ntrain,Ntrain)
        for sc=1:Ntrain
            for sc2=sc+1:Ntrain
                dist[sc,sc2]=sum((SentenceRep[sc]-SentenceRep[sc2]).^2)
                dist[sc2,sc]=dist[sc,sc2]
            end
            dist[sc,sc]=realmax()
        end
        NearestSentence=zeros(Ntrain)
        for sc=1:Ntrain
            (val,ind)=findmin(dist[sc,:])
            NearestSentence[sc]=ind
        end

        bow=Array(Any,Ntrain)
        for sc=1:Ntrain
            bow[sc]=zeros(W)
            for i=1:W
                if any(seq[sc].==i)
                    bow[sc][i]=1
                end
            end
        end

        cossim(x,y)=sum(x.*y)./sqrt(sum(x.*x)*sum(y.*y))

        dist2=zeros(Ntrain,Ntrain)
        for sc=1:Ntrain
            for sc2=sc+1:Ntrain
                dist2[sc,sc2]=cossim(bow[sc],bow[sc2])
                dist2[sc2,sc]=dist2[sc,sc2]
            end
            dist2[sc,sc]=realmin()
        end
        NearestSentenceBow=zeros(Ntrain)
        for sc=1:Ntrain
            (val,ind)=findmax(dist2[sc,:])
            NearestSentenceBow[sc]=ind
        end
    end
#end
