# Training
In order to train the model, all important parameters can be found at ```config``` folder. To train the model execute the following command line. The trained model weights will be saved at the ```experiments``` folder. The dataset used was provided by the Unico research team. The trained weights will be stored at ```experiments/[EX_NAME]/weights```.

```
python3 Train.py --exp_name [NAME] --cfg [PATH][OPTIONAL] --logs_path [PATH][OPTINAL] 
```

# Training Loss and Acc graphs
In every training experiment, a report file is created inside the ```experiments``` folder. This file contains information about the Loss and accuracy from the training in each epoch. In order to see theses values in graph form you can run the following command:
```
python3 graphs.py --report [PATH]
```

# Inference
To inference the model on a single image, execute the following command line
```
python3 inference.py --img[PATH] --model[PATH][OPTIONAL]
```

# Enrollment 
To add new unseen labeled dogs run the following command. It will add the images to the ``` dogs_dataset/train_organized```. After the enrollment the model must be retrained with old and new labels.
```
  python3 Enrollment.py --new_samples [PATH] --database [PATH]
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

# Results dog breed classification
Three deep learning architeture were analyzed for this application: efficientNet_B2, resnet18 and resnet50. The architeture were tested using a separated test data from the original dataset, resulting in 12 trained models. All the results are stored at ```benchmark_results``` along with confusion matrices data (resulted from the ``` Benchmark.py ``` command). The model with the best accuracy was chosen to the web application and the following analyses with new label enrollment and unkown detection. The model with the best accuracy according to the experiments was ```resnet50_ext_SGD_001_LR5```.

| Model Name | ACC(%) | #Samples | Inference_Time(s) |
|  :-----------:    | :-----:  | :-----:  | :-----:  |
| **resnet18_ext_SGD_001_cte** | 72 | 1079 |  0.0037 |
| **resnet18_ext_adam_0001_cte** | 75 | 1079 |  0.0039 |
| **resnet18_ext_SGD_001_LR5** | 78 | 1079 |  0.0038 |
| **resnet18_ext_adam_0001_LR5** | 79 | 1079 | 0.0037 |
| **resnet50_ext_SGD_001_cte** | 80 | 1079 | 0.0083 |
| **resnet50_ext_adam_0001_cte** | 82 | 1079 | 0.0079 |
| **resnet50_adam_0001_LR5** | 85 | 1079 | 0.0082 |
| **resnet50_ext_SGD_001_LR5** | **85** | **1079** | **0.0084** |
| **EffNetB2_SGD_001_cte** | 76 | 1079 |  0.137 | 
| **EffNetB2_adam_0001_cte** | 67 | 1079 | 0.0143 |
| **EffNetB2_SGD_001_LR5** | 81 | 1079 | 0.0135 |
| **EffNetB2_adam_0001_LR5** | 73 | 1079 | 0.0140 |

# Result enrollment new labels
The enrollment analysis was done in 2 ways. First the model with the best accuracy from previous results was retrained with the original 100 labels + 20 new labels. The model was then tested with a test dataset with 120 labels. Second the same model architeture was retrained only with the new labels. The results can be seen in the table below.
| Model Name | ACC(%) | #Samples | Inference_Time(s) |
|  :-----------:    | :-----:  | :-----:  | :-----:  |
| **resnet50_ext_SGD_001_LR5_120** | 81 | 2109 | 0.0036 |
| **resnet50_ext_SGD_001_LR5_20** | 95 | 1030 | 0.0036 |


# Results unknown detection
In order to detect unlabeled images, a simple threshold in relation with the predicted probability was implemented. Threshold values were tested experimentaly to achieve better accuracy. The results shows that this approach does not perform well since it is splitting the data in known and unknown labels. Maybe another approach would be to add a new label called "unknown" with different dog images and train the model with it. The results were calculated using the command:

```
  python3 Benchmark_unlabeled.py --tr [Float] --dataset [PATH] --unknown [PATH]
```

| Threshold | ACC(%) | #Samples |
|  :-----------:    | :-----:  | :-----:  |
|  0.1   | 28 | 2109 |
|  0.2   | 28 | 2109 |
|  0.3   | 31 | 2109 |
|  0.4   | 34 | 2109 |
|  0.5   | 37 | 2109 |
|  0.6   | 40 | 2109 |
|  0.7   | 44 | 2109 |
|  0.8   | 47 | 2109 |
|  0.9   | 50 | 2109 |

# References
1. https://www.analyticsvidhya.com/blog/2022/03/dog-breed-classification-using-a-stacked-pre-trained-model/
2. https://www.kaggle.com/code/masrur007/resnet50-and-cnn-from-scratch-dog-breed-classify
3. https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
4. https://machinelearningmastery.com/update-neural-network-models-with-more-data/
