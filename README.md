mono-yolo-depthestimation !!!

YOLOv5: I'm using online pretrained model (torch.hub.load from GitHub).

Monodepth2: I'm using an offline saved disparity map (.npy file), which means the model inference was done earlier, not in this script.

Then atlast running the combined script (run_combined.py) to predict the depth estimation.
