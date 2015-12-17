function nearestword(word,wvec,lexicon)

    wind=find(lexicon.==word)
    dist=zeros(length(lexicon))
    thiswvec=wvec[wind]
    for i=1:length(lexicon)
        dist[i]=sum( (wvec[i]-thiswvec[1]).^2)
    end
    dist[wind]=realmax()
    (val,ind)=findmin(dist)
    return lexicon[ind]
end


function nearestword(word,wmatrix::Array{Float64,2},lexicon)
    wind=find(lexicon.==word)
    dist=zeros(length(lexicon))
    thiswvec=wmatrix[:,wind]
    for i=1:length(lexicon)
        dist[i]=sum( (wmatrix[:,i]-thiswvec).^2)
    end
    dist[wind]=realmax()
    (val,ind)=findmin(dist)
    return lexicon[ind]
end
