#function DemoRNNatisNew()
    include("loadsave.jl")
    include("setvalue!.jl")
    initialiseparameters=true
    Ntrain=0

    LogFile="LogFile"
    include("getatisdata.jl")
    tokens,lexicon=getatisdata(Ntrain)
    Ntrain=length(tokens)
    word=lexicon

    #BatchSize=Ntrain
    BatchSize=10

    seq=tokens

    oldseq=deepcopy(seq)
    goodseq=BitArray(Ntrain,1)
    fill!(goodseq,false)
    counter=0
    for i=1:Ntrain
        if length(seq[i])>3
            counter+=1
            seq[counter]=oldseq[i][1:min(10,length(seq[i]))]
        end
    end
    seq=seq[1:counter]

    Ntrain=length(seq)
    W=length(word) # total number of words in dictionary
    Tmax=maximum(map((x)->length(x),seq))

    H=50 # size of the latent representation
    H1=50
    X=50 # size of the word vec representation

    #H1=11; H2=4;
    loss=zeros(1,Tmax)
    includedloss=zeros(1,Tmax)
    lossIDX=zeros(1,Tmax)
    x=zeros(1,Tmax)
    xvec=zeros(1,Tmax)
    h=zeros(1,Tmax)
    hout=zeros(1,Tmax)
    a1=zeros(1,Tmax)
    a2=zeros(1,Tmax)
    r1=zeros(1,Tmax)
    r2=zeros(1,Tmax)
    target=zeros(1,Tmax)

    node=Array(ADnode,10000) # nodes
    c=0 # node counter
    node[invmaxlength=c+=1]=ADnode([])

    # tied weights:
    node[wordmatrix=c+=1]=ADnode([];returnderivative=true)
    node[W1x=c+=1]=ADnode([];returnderivative=true)
    node[W1h=c+=1]=ADnode([];returnderivative=true)
    node[W2=c+=1]=ADnode([];returnderivative=true)
    node[W3=c+=1]=ADnode([];returnderivative=true)

    node[h[1]=c+=1]=ADnode([])
    for t=1:Tmax-1
        node[target[t+1]=c+=1]=ADnode([])
        node[x[t]=c+=1]=ADnode([]) # inputs are word indices
        node[xvec[t]=c+=1]=ADnode([wordmatrix,x[t]],Fgetcol) # wordvecs
        #node[a1[t]=c+=1]=ADnode([W1x,xvec[t]],FAx)
        #node[a2[t]=c+=1]=ADnode([W1h,h[t]],FAx)
        #node[r1[t]=c+=1]=ADnode([a1[t],a2[t]],Fxpy)
        #node[r2[t]=c+=1]=ADnode(r1[t],Fsigma)

        node[a1[t]=c+=1]=ADnode([W1x,xvec[t]],FshiftedsigmaAx)
        #node[a2[t]=c+=1]=ADnode([W1h,h[t]],FshiftedsigmaAx)

        #node[a1[t]=c+=1]=ADnode([W1x,xvec[t]],FrectlinAx)
        node[a2[t]=c+=1]=ADnode([W1h,h[t]],FrectlinAx)
        node[r1[t]=c+=1]=ADnode([a1[t],a2[t]],Fxpy)
        #node[r2[t]=c+=1]=ADnode(r1[t],Fsigma)
        #node[r2[t]=c+=1]=ADnode(r1[t],Frectlin)

        #node[h[t+1]=c+=1]=ADnode([W2,r1[t]],FshiftedsigmaAx)
        node[h[t+1]=c+=1]=ADnode([W2,r1[t]],FssigmaAx)
        #node[h[t+1]=c+=1]=ADnode([W2,r2[t]],FsigmaAx)
        node[hout[t+1]=c+=1]=ADnode([W3,h[t+1]],FAx)
        node[loss[t+1]=c+=1]=ADnode([target[t+1] hout[t+1]],FMultLogisticLoss)
        node[lossIDX[t+1]=c+=1]=ADnode([]) # binary loss indicator (whether to include this timepoint -- deals with sequences of differing lengths -- shunt all sequences so that the last state is the end of each sentence)
        node[includedloss[t+1]=c+=1]=ADnode([loss[t+1] lossIDX[t+1]],Fxy)
    end

    (unscaledtotalloss,c)=mapreduce!(Fx,Fxpy,includedloss[2:Tmax],node)
    node[totalloss=c+=1]=ADnode([unscaledtotalloss,invmaxlength],Fxy)
    # total number of nodes in the net
    node=node[1:c]

    if initialiseparameters
        value=Array(Any,c) # function values on the nodes
        value[h[1]]=zeros(H,1)
        value[W1x]=sign(randn(H1,X))/sqrt(H1)
        value[W1h]=sign(randn(H1,H))/sqrt(H1)
        value[W2]=sign(randn(H,H1))/sqrt(H)
        value[W3]=sign(randn(W,H))/sqrt(W)
        value[wordmatrix]=sign(randn(X,W))/sqrt(X)
        value[invmaxlength]=1.#1000000./(Tmax*BatchSize)
        #value[invmaxlength]=1.
    end

    # instantiate the inputs for this sequence:
    sc=1 # just do this for sequence 1 to see if it compiles ok:
    setvalue!(value,seq[sc],h,x,lossIDX,target,Tmax,W)
    (value,auxvalue,gradient,net)=compile(value,node) # compile and preallocate memory

#    gradcheck(value,net;showgrad=false)

    # Nesterov Training:
    TrainingIts=5000
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
        mu=1-3/(t/1000+5);
        if t>1
            epsilon=10.10/(1+t/1000)
        else
            epsilon=10.10
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
            setvalue!(value,seq[sc],h,x,lossIDX,target,Tmax,W)
            ADeval!(value,net;auxvalue=auxvalue,gradient=gradient)

            if gradcounter==1
                totalgradient=deepcopy(gradient)
                cumtotalloss=deepcopy(totalloss)
            else
                totalgradient+=gradient
                cumtotalloss+=value[totalloss]
            end
        end
        totalgradient=totalgradient/BatchSize
        er[t]=cumtotalloss/BatchSize

        for par=parstoupdate
            println("$(par) : $(mean(abs(totalgradient[par])))")
        end

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
        if mod(t,1000)==1
            @savevars("results",value)
        end
    end


    toc()

    if true

        SentenceRep=Array(Any,Ntrain)
        for sc=1:Ntrain
            println("\nsequence $sc :")
            setvalue!(value,seq[sc],h,x,lossIDX,target,Tmax,W)
            ADeval!(value,net;auxvalue=auxvalue,gradient=gradient)
            counter=0
            println("training:")
            for t=1:length(seq[sc])
                counter+=1
                print("$(word[seq[sc][t]]) ")
            end
            println("\nprediction:")
            for t=Tmax-length(seq[sc])+2:Tmax
                (val,ind)=findmax(softmax(value[hout[t]]))
                print("$(word[ind]) ")
            end
            SentenceRep[sc]=value[h[end]]
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
