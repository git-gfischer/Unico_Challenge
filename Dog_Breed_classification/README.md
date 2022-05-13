# Unico Challenge  - Dog Bread Recognition 

## Description

Deep Learning application to recognize dogs breeds. <br/>

# Depedencies
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

# Preprocessing
Before training the model the dataset must be organized in training, validation and test data. For some reason there are some black images in the dataset. In order to prevent those image to disturb the training process, a black image filter was implemented. Run the following command to organize and filter the black images from the dataset.
```
cd Dataset_tools
python3 Organize.py --dataset [PATH] --black_img_filter
```

# Training
In order to train the model, all important parameters can be found at ```config``` folder. To train the model execute the following command line. The trained model weights will be saved at the ```experiments``` folder. The dataset used was provided by the Unico research team. The trained weights will be stored at ```experiments/[EX_NAME]/weights```.

```
python3 Train.py --cfg[PATH][OPTIONAL] --logs_path[PATH][OPTINAL] --exp_name[NAME][OPTIONAL]
```

# Training Loss and Acc graphs
In every training experiment, a report file is created inside the ```experiments``` folder. This file contains information about the Loss and accuracy from the training in each epoch. In order to see theses values in graph form you can run the following command:
```
python3 graphs.py --report [PATH]
```

# Inference
To inference the model on a single image, execute the following command line
```
python3 inference.py --img[PATH] --model[PATH]
```

# Benchmark
To measure the model accuracy and robustness in a test dataset, execute the following command line. This process will generate a csv file at ```benchmark_results ``` folder. The ```--cm ``` flag will generate the confusion matrix and will store along with the csv file.
```
python3 Benchmark.py --mode path --dataset [PATH] --csv_out [NAME.csv] --model [PATH][OPTIONAL] --csv_path [PATH][OPTIONAL] --cfg [PATH][OPTIONAL] --cm [OPTIONAL]
```
To read the generated csv file and calculate the metrics (and spare process time) run:
```
  python3 Benchmark.py --mode csv --dataset [PATH.csv]
```

# Results
Here the following models were tested using the 2 different dataset (from the ``` Benchmark.py ``` command):

| Model Name | ACC(%) | Precision(%) | Recall(%) | F1(%) | #Samples | #Support |
|  :-----------:    |     :---:    |     :---------:      |     :------------:     |     :---------:    | :-----:  | :-----:  |
| **VGG19_T1_e13_RFW_retina** | 84 | 82/95/77/88 | 98/77/89/73 | 90/85/82/80 | 40599 | 10414/9685/10193/10307 |


# References
1. https://www.analyticsvidhya.com/blog/2022/03/dog-breed-classification-using-a-stacked-pre-trained-model/
2. https://www.kaggle.com/code/masrur007/resnet50-and-cnn-from-scratch-dog-breed-classify
3. https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5