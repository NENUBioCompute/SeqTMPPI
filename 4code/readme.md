
# quickly start with this
~~~
# run with default model and dataset 
python run.py

# run with selected model and dataset
python run.py --model ../2model/0/_my_model.h5 --fasta sample/pair.fasta --pair sample/pair.tsv --output_path result/


~~~

# environment
the model was trained on gpu

# requirement
- python 3.8.3 
- Keras==2.4.3
- tensorflow==2.4.0

pip install -r requirements_gpu.txt




