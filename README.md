# Facial-Expression-Recognition-using-CNN

## Description: 
- The study of facial expression recognition using convolution neural networks (CNN) is being proposed, and the aim is to classify the expression of different faces. The model based on LeNet-5 and the algorithm of CNNs is used to train with the FER2013 dataset. In the LeNet-5 model, we are using the two convolutional layers with the ReLu, filters, and the two max-pooling layers and the fully connected layer with activation functions i.e., soft max for a probability distribution.

## Dataset Description:
- The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).


## Installation:
1. Clone the repository to your local machine:
  ```bash
  git clone https://github.com/VaibhavGovindwar/Facial-Expression-Recognition-using-CNN.git
```
 ```cd Facial-Expression-Recognition-using-CNN```

2. Install the required dependencies using pip:
  ```bash
  pip install -r requirements.txt
```

## Download and load the Dataset file in your environment:
  ```bash 
  Dataset.csv
```

## Project Structure:
- To run the project, execute the scripts in the following order:
- The project is divided into multiple Python files that need to be run sequentially to set up the entire pipeline. Below is the order in which the scripts should be executed:

1. Import_Libraries.py
- First, ensure that all necessary libraries are imported.
  ```bash
  Import_Libraries.py
  ```

2. Load_dataSets.py
- Load the dataset for facial expression recognition.
  ```bash
  Load_dataSets.py
  ```

3. Analizing_DataSet.py
- Analyzes the loaded dataset, providing insights into the data.
  ```bash
  Analizing_DataSet.py
  ```

4. Visualize_the_DataSet_Graph.py
- Visualizes the dataset and statistical information through graphs.
  ```bash
  Visualize_the_DataSet_Graph.py
  ```

5. Visualize_Emotions.py
- Plots the distribution of different emotions in the dataset.
  ```bash
  Visualize_Emotions.py
  ```

6. Pre-processing_Data.py
- Splitting dataset into three parts: train, validation, test
- Convert strings to lists of integers
- Reshape to 48x48 and normalize the grayscale image with 255.0
  ```bash
  Pre-processing_Data.py
  ```

7. LeNet-5_model.py
- Defines the architecture of the LeNet-5 model for facial expression recognition.
 ```bash
  LeNet-5_model.py
```

8. Compiling_model.py
- Compiles the model, specifying the optimizer, loss function, and metrics.
  ```bash
  Compiling_model.py
  ```
  
9. Performance.py
- Evaluate the performance of the model using test data and displays the results.
  ```bash
  Performance.py
  ```
  
10. Visualize_Training_&_Validation.py
- Visualizes the training and validation accuracy/loss over epochs.
 ```bash
  Visualize_Training_&_Validation.py
```
  
11. Confusion_Matrix.py
- Generates a confusion matrix to evaluate the performance of the model on different classes.
  ```bash
  Confusion_Matrix.py
  ```
  
12. Classification.py
- Performs the classification of new data using the trained model.
  ```bash
  Classification.py
  ```
  
13. Save_model.py
- Saves the trained model to a file for future use.
  ```bash
  Save_model.py
  ```

## Description of LeNet-5 Model:
- LeNet5 is a small network, it contains the basic modules of deep learning: convolutional layer, pooling layer, and fully connected layer.

1. Conv2D(6,(5,5)) -> Activation(Relu) -> Input_shape(48,48,1)

2. Maxpooling(2,2)

3. Conv2D(16,(5,5)) -> Activation(Relu)

4. MaxPooling(2,2)

- Flatten()

5. Dense(120) -> Activation(Relu)

6. Dense(84) -> Activation(Relu)

7. Dense(7)-> Activation(soft-Max)

## Compiling the model:
- Compile the model using the Adam optimizer with a learning rate of 0.001. Categorical cross-entropy is chosen as the loss function, which is commonly used for multi-class classification problems. Accuracy is specified as the metric to monitor during training.

- 'train_x' and 'train_y': are the input features and target labels for training, respectively.
- 'epochs': specifies the number of epochs for training.
- 'batch_size': determines the number of samples per gradient update.
- 'validation_data': is a tuple (test_x, test_y) providing the validation data.
- 'verbose': controls the verbosity of the training output. Setting verbose=1 means you'll see progress bars for each epoch.

- The training history is stored in the history object, which can be used to visualize the training process.

## Analysis using Confusion Matrix:
- The confusion matrix is calculated based on the predictions made by the model on the testing dataset. It compares the predicted labels against the true labels for each sample in the testing dataset. The rows of the matrix represent the true classes, while the columns represent the predicted classes. Each cell in the matrix shows the number (or proportion) of samples that were classified into a particular combination of true and predicted classes.

- By analyzing the confusion matrix, you can gain insights into which classes the model is performing well on and which classes it's struggling with. This information can be valuable for fine-tuning the model or identifying areas for improvement.

- In summary, there's no strict requirement for the sum of all classes in the confusion matrix to be exactly 100 or any other specific value. It depends on factors such as the dataset distribution, model performance, and whether the confusion matrix is normalized.

## Conclusion:
- The LeNet-5 Model, initially designed for image recognition tasks such as Digit Recognition and Face Recognition, was employed for facial expression recognition using the FER2013 dataset. Despite achieving a high training accuracy of 95.39%, the model's performance on the testing dataset was disappointing, yielding only a 49.47% accuracy. This significant drop in accuracy suggests that the LeNet-5 architecture may not be well-suited for the complexities of the FER2013 dataset, indicating limitations in its ability to generalize to unseen facial expressions. Therefore, further exploration with alternative models or modifications to the LeNet-5 architecture may be necessary to improve performance on facial expression recognition tasks.

## Future Enhancements:

- Implement real-time emotion recognition through webcam input.
- Use transfer learning to further optimize the model using pre-trained models.

## License
This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

## Acknowledgments:
The complete code and output of this project 👇
```bash
https://www.kaggle.com/code/vaibhavgovindwar/fer-cnn-lenet-5
```
