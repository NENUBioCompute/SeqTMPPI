_8DIPPredict_1.py

make dir  file/8DIPPredict/data_all/Ecoli/
make dir  file/8DIPPredict/data_all/Ecoli/dirRelated
save related to file/8DIPPredict/data_all/Ecoli/dirRelated
save 880 protein fasta file/8DIPPredict/data_all/Ecoli/dirRelated/2pair.fasta
save 369 tmp  file/8DIPPredict/data_all/Ecoli/dirRelated/2tmp.fasta
save 511 nontmp  file/8DIPPredict/data_all/Ecoli/dirRelated/2nontmp.fasta
make dir  file/8DIPPredict/data_all/Mus/
make dir  file/8DIPPredict/data_all/Mus/dirRelated
save related to file/8DIPPredict/data_all/Mus/dirRelated
save 552 protein fasta file/8DIPPredict/data_all/Mus/dirRelated/2pair.fasta
save 265 tmp  file/8DIPPredict/data_all/Mus/dirRelated/2tmp.fasta
save 287 nontmp  file/8DIPPredict/data_all/Mus/dirRelated/2nontmp.fasta
make dir  file/8DIPPredict/data_all/Human/
make dir  file/8DIPPredict/data_all/Human/dirRelated
save related to file/8DIPPredict/data_all/Human/dirRelated
save 1029 protein fasta file/8DIPPredict/data_all/Human/dirRelated/2pair.fasta
save 426 tmp  file/8DIPPredict/data_all/Human/dirRelated/2tmp.fasta
save 603 nontmp  file/8DIPPredict/data_all/Human/dirRelated/2nontmp.fasta
make dir  file/8DIPPredict/data_all/SC/
make dir  file/8DIPPredict/data_all/SC/dirRelated
save related to file/8DIPPredict/data_all/SC/dirRelated
save 1959 protein fasta file/8DIPPredict/data_all/SC/dirRelated/2pair.fasta
save 866 tmp  file/8DIPPredict/data_all/SC/dirRelated/2tmp.fasta
save 1093 nontmp  file/8DIPPredict/data_all/SC/dirRelated/2nontmp.fasta
make dir  file/8DIPPredict/data_all/HP/
make dir  file/8DIPPredict/data_all/HP/dirRelated
save related to file/8DIPPredict/data_all/HP/dirRelated
save 38 protein fasta file/8DIPPredict/data_all/HP/dirRelated/2pair.fasta
save 18 tmp  file/8DIPPredict/data_all/HP/dirRelated/2tmp.fasta
save 20 nontmp  file/8DIPPredict/data_all/HP/dirRelated/2nontmp.fasta
stop 2021-03-21 21:07:29
time 1.3807487487792969


ssh://19jjhnenu@210.47.18.43:22/usr/bin/python -u /home/19jjhnenu/Code/.pycharm_helpers/pydev/pydevconsole.py --mode=server
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/19jjhnenu/Code/SeqTMPPI20201226', '/home/19jjhnenu/Code/SeqTMPPI20201226'])
PyDev console: starting.
Python 3.6.13 |Anaconda, Inc.| (default, Jun  4 2021, 14:25:59)
[GCC 7.5.0] on linux
runfile('/home/19jjhnenu/Code/SeqTMPPI20201226/v2_train.py', wdir='/home/19jjhnenu/Code/SeqTMPPI20201226')
Using TensorFlow backend.
start 2021-06-30 15:47:19
training model
cross train
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/0
predict  file/8DIPPredict/predict/Ecoli/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/0
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/0/4/_my_model.h5
2021-06-30 15:47:43.870684: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2021-06-30 15:47:44.231229: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:04:00.0
2021-06-30 15:47:44.236160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 1 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:05:00.0
2021-06-30 15:47:44.250504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 2 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:08:00.0
2021-06-30 15:47:44.274089: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 3 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:09:00.0
2021-06-30 15:47:44.286792: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 4 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:83:00.0
2021-06-30 15:47:44.317730: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 5 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:84:00.0
2021-06-30 15:47:44.323222: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 6 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:87:00.0
2021-06-30 15:47:44.329374: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 7 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:88:00.0
2021-06-30 15:47:44.330788: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2021-06-30 15:47:44.353853: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2021-06-30 15:47:44.368394: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2021-06-30 15:47:44.371204: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2021-06-30 15:47:44.381209: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2021-06-30 15:47:44.384826: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2021-06-30 15:47:44.405463: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2021-06-30 15:47:44.607273: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0, 1, 2, 3, 4, 5, 6, 7
2021-06-30 15:47:44.611351: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2021-06-30 15:47:44.655580: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2100115000 Hz
2021-06-30 15:47:44.675621: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56417d891f60 executing computations on platform Host. Devices:
2021-06-30 15:47:44.676187: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
2021-06-30 15:47:54.267667: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:04:00.0
2021-06-30 15:47:54.270797: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 1 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:05:00.0
2021-06-30 15:47:54.273932: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 2 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:08:00.0
2021-06-30 15:47:54.276938: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 3 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:09:00.0
2021-06-30 15:47:54.287258: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 4 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:83:00.0
2021-06-30 15:47:54.300002: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 5 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:84:00.0
2021-06-30 15:47:54.302714: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 6 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:87:00.0
2021-06-30 15:47:54.317447: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 7 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:88:00.0
2021-06-30 15:47:54.320750: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2021-06-30 15:47:54.321027: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2021-06-30 15:47:54.321101: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2021-06-30 15:47:54.321163: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2021-06-30 15:47:54.321236: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2021-06-30 15:47:54.321298: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2021-06-30 15:47:54.321357: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2021-06-30 15:47:54.385041: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0, 1, 2, 3, 4, 5, 6, 7
2021-06-30 15:47:54.386573: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2021-06-30 15:47:54.465887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-06-30 15:47:54.470098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 1 2 3 4 5 6 7
2021-06-30 15:47:54.473388: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N Y Y Y N N N N
2021-06-30 15:47:54.475445: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 1:   Y N Y Y N N N N
2021-06-30 15:47:54.477348: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 2:   Y Y N Y N N N N
2021-06-30 15:47:54.478046: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 3:   Y Y Y N N N N N
2021-06-30 15:47:54.479215: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 4:   N N N N N Y Y Y
2021-06-30 15:47:54.480106: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 5:   N N N N Y N Y Y
2021-06-30 15:47:54.480874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 6:   N N N N Y Y N Y
2021-06-30 15:47:54.481477: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 7:   N N N N Y Y Y N
2021-06-30 15:47:54.570685: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10481 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:04:00.0, compute capability: 6.1)
2021-06-30 15:47:54.597155: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10481 MB memory) -> physical GPU (device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:05:00.0, compute capability: 6.1)
2021-06-30 15:47:54.605265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 10481 MB memory) -> physical GPU (device: 2, name: GeForce GTX 1080 Ti, pci bus id: 0000:08:00.0, compute capability: 6.1)
2021-06-30 15:47:54.613129: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:3 with 10481 MB memory) -> physical GPU (device: 3, name: GeForce GTX 1080 Ti, pci bus id: 0000:09:00.0, compute capability: 6.1)
2021-06-30 15:47:54.622901: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:4 with 10481 MB memory) -> physical GPU (device: 4, name: GeForce GTX 1080 Ti, pci bus id: 0000:83:00.0, compute capability: 6.1)
2021-06-30 15:47:54.631306: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:5 with 10481 MB memory) -> physical GPU (device: 5, name: GeForce GTX 1080 Ti, pci bus id: 0000:84:00.0, compute capability: 6.1)
2021-06-30 15:47:54.638935: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:6 with 1800 MB memory) -> physical GPU (device: 6, name: GeForce GTX 1080 Ti, pci bus id: 0000:87:00.0, compute capability: 6.1)
2021-06-30 15:47:54.649787: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:7 with 10219 MB memory) -> physical GPU (device: 7, name: GeForce GTX 1080 Ti, pci bus id: 0000:88:00.0, compute capability: 6.1)
2021-06-30 15:47:54.737146: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56418124bef0 executing computations on platform CUDA. Devices:
2021-06-30 15:47:54.741798: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): GeForce GTX 1080 Ti, Compute Capability 6.1
2021-06-30 15:47:54.750640: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (1): GeForce GTX 1080 Ti, Compute Capability 6.1
2021-06-30 15:47:54.751074: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (2): GeForce GTX 1080 Ti, Compute Capability 6.1
2021-06-30 15:47:54.751393: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (3): GeForce GTX 1080 Ti, Compute Capability 6.1
2021-06-30 15:47:54.751690: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (4): GeForce GTX 1080 Ti, Compute Capability 6.1
2021-06-30 15:47:54.751861: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (5): GeForce GTX 1080 Ti, Compute Capability 6.1
2021-06-30 15:47:54.752178: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (6): GeForce GTX 1080 Ti, Compute Capability 6.1
2021-06-30 15:47:54.752585: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (7): GeForce GTX 1080 Ti, Compute Capability 6.1
2021-06-30 15:48:12.965966: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2021-06-30 15:48:15.918254: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2021-06-30 15:48:25.291221: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Not found: ./bin/ptxas not found
Relying on driver to perform ptx compilation. This message will be only logged once.
2000/2000 [==============================] - 34s 17ms/step
Loss:1.144413,ACC:0.536000
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/0
predict  file/8DIPPredict/predict/Human/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/0
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/0/4/_my_model.h5
2004/2004 [==============================] - 24s 12ms/step
Loss:1.457272,ACC:0.512974
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/0
predict  file/8DIPPredict/predict/Mus/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/0
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/0/4/_my_model.h5
916/916 [==============================] - 12s 13ms/step
Loss:1.365923,ACC:0.507642
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/0
predict  file/8DIPPredict/predict/SC/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/0
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/0/4/_my_model.h5
4726/4726 [==============================] - 58s 12ms/step
Loss:0.861028,ACC:0.582522
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/1
predict  file/8DIPPredict/predict/Ecoli/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/1
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/1/4/_my_model.h5
2000/2000 [==============================] - 23s 11ms/step
Loss:1.260938,ACC:0.525500
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/1
predict  file/8DIPPredict/predict/Human/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/1
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/1/4/_my_model.h5
2004/2004 [==============================] - 23s 11ms/step
Loss:1.359686,ACC:0.510978
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/1
predict  file/8DIPPredict/predict/Mus/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/1
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/1/4/_my_model.h5
916/916 [==============================] - 11s 12ms/step
Loss:1.248406,ACC:0.522926
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/1
predict  file/8DIPPredict/predict/SC/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/1
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/1/4/_my_model.h5
4726/4726 [==============================] - 67s 14ms/step
Loss:0.844533,ACC:0.578079
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/2
predict  file/8DIPPredict/predict/Ecoli/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/2
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/2/4/_my_model.h5
2000/2000 [==============================] - 29s 14ms/step
Loss:1.150688,ACC:0.528000
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/2
predict  file/8DIPPredict/predict/Human/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/2
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/2/4/_my_model.h5
2004/2004 [==============================] - 29s 14ms/step
Loss:1.490821,ACC:0.506487
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/2
predict  file/8DIPPredict/predict/Mus/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/2
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/2/4/_my_model.h5
916/916 [==============================] - 14s 15ms/step
Loss:1.399673,ACC:0.510917
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/2
predict  file/8DIPPredict/predict/SC/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/2
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/2/4/_my_model.h5
4726/4726 [==============================] - 47s 10ms/step
Loss:0.917640,ACC:0.559670
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/3
predict  file/8DIPPredict/predict/Ecoli/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/3
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/3/4/_my_model.h5
2000/2000 [==============================] - 19s 9ms/step
Loss:1.104716,ACC:0.533500
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/3
predict  file/8DIPPredict/predict/Human/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/3
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/3/4/_my_model.h5
2004/2004 [==============================] - 18s 9ms/step
Loss:1.596447,ACC:0.504990
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/3
predict  file/8DIPPredict/predict/Mus/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/3
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/3/4/_my_model.h5
916/916 [==============================] - 6s 6ms/step
Loss:1.495470,ACC:0.510917
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/3
predict  file/8DIPPredict/predict/SC/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/3
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/3/4/_my_model.h5
4726/4726 [==============================] - 33s 7ms/step
Loss:0.917850,ACC:0.563267
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/4
predict  file/8DIPPredict/predict/Ecoli/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Ecoli_+-/4
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/4/3/_my_model.h5
2000/2000 [==============================] - 13s 6ms/step
Loss:1.224196,ACC:0.523500
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/4
predict  file/8DIPPredict/predict/Human/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Human_+-/4
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/4/3/_my_model.h5
2004/2004 [==============================] - 13s 6ms/step
Loss:1.458684,ACC:0.511976
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/4
predict  file/8DIPPredict/predict/Mus/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_Mus_+-/4
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/4/3/_my_model.h5
916/916 [==============================] - 6s 6ms/step
Loss:1.349307,ACC:0.516376
make dir  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/4
predict  file/8DIPPredict/predict/SC/0/all.txt ...
save result in  /home/19jjhnenu/Data/Phsi_Blos/result/5CV_1_DIP_SC_+-/4
load model... /home/19jjhnenu/Data/Phsi_Blos/result/benchmark/4/3/_my_model.h5
4726/4726 [==============================] - 30s 6ms/step
Loss:0.898729,ACC:0.566864
testing the model
stop 2021-06-30 16:15:27
time 1688.0957515239716

