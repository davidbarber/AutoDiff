function loadmnist()
    # (c) David Barber, University College London 2015    
    alldata=matread("mnist-original.mat")
    images=(float(alldata["data"]))./255
    return images, alldata["label"]
end
