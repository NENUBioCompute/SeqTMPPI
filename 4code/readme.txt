tips:
- files from _1* to _5* records part of our methods steps
- tool/_8DIPPredict_support.py provides a guid to construct a balanced dataset of TMP-nonTMP for evaluating the model with whole matrix
- _5DIPPredict.py provides a guid to predicting with our well trained model

the dataset was processed on cpu
the model was trained on gpu

requirement:
pip install -r requirements_cpu.txt
pip install -r requirements_gpu.txt
