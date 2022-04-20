# CSC413FinalProject
The dataset is retrived from https://www.kaggle.com/c/plant-seedlings-classification

`SplitTest` create new folders for spliting the test set. `Train.py` is for training. `VisualizeCAM` generates CAM plots. `LayerCAM` is from the implementation of LayerCAM paper: https://github.com/PengtaoJiang/LayerCAM-jittor. I modified the `__init__.py` in `utils` and `layercam` in `cam` to make it capable of working with inception-v3 and retriving the desired layer. 