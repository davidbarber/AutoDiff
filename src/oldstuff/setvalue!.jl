  function setvalue!(value,thisseq,h,x,lossIDX,target,Tmax,W,H1)
      H=length(value[h[1]])
                 for tind=1:Tmax
               value[h[tind]]=ones(H,1)/H1
           end

            T=length(thisseq)
            for tind=1:Tmax-length(thisseq)
                value[x[tind]]=0
            end
            counter=0

            for tind=2:Tmax
                value[lossIDX[tind]]=0 # initialise all to zero
                value[target[tind]]=BitArray(W,1)
                #fill!(value[target[tind]],false)
            end

            counter=0
            for tind=Tmax-length(thisseq)+2:Tmax
                counter+=1
                fill!(value[target[tind]],false)
                value[target[tind]][thisseq[counter+1]]=true
                value[lossIDX[tind]]=1
                value[x[tind-1]]=thisseq[counter]
            end
      end
