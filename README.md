# hand-pose-estimation
![Hand-pose-estimation](https://user-images.githubusercontent.com/52950649/138914461-3fc0fd26-c126-4d5c-9a96-dbcde704c229.png)


 This is a repository for the demo of CPM2DPose network. It infers 21 key points of hand image and visualizes the inferred skeleton. It is based on the CPM network and some layers such as batch normalization and drop-out layer is added which improves the accuracy. There're two version of the file. 'CV_Final.py' trains the network with a single GPU setting while 'CV_Final-DDP-AMP-test.py' trains the network with multiple GPU via PyTorch DDP module. The 'Automatic Mixed Precision' is also used in later version to accelerate the training.
