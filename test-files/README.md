Configuration file structure:
1.- Network architecture (layer-0 is the input layer and layer-(N-1) is the output one)
    .- #layers(N) #neurons-layer-0 #neurons-layer-1 ... #neurons-layer-(N-1)
    .- learning-rate (float)
    .- batch-size (int)
    .- #training-epochs (int)

2.- Data set description (number of neurons in layer-0 must be the image size [Image-dimension-x * Image-dimension-y] and number of neurons in layer-(N-1) must be number of output classes. Training-file-name = Set-file-name.tra and Validation-file-name=Set-file-name.cv)
    .- #training-images 
    .- #validation-images
    .- Image-dimension-x Image-dimension-y
    .- Set-file-name
   