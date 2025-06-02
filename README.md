# Building a Review classification model using LSTM with use of PyTorch 

## Project Description 
The purpose of this project is to build a review classification model, that does sentiment analysis of the review content from an application which is scaled in the rage of 1 to 5. The deep learning model was built using LSTM (Long Short Term Memory) which is a variant of RNN ( Recurrent Neural Network). The model is trained on the review data and their properties to predict the sentiment, using PyTorch module.This project outlines an automated model pipeline which can be used to evaluate and validate the model's performance.

## Background 
Due to the vainishing gradient problem during the backpropogation, the traditional RNNs become ineffective in capturing long-term dependencies in long textual sequences.LSTMS have a gating mechanism which uses three gates called input, forget and output to control the flow of information between layers.Forget gate determines which information from previous cell state to discard, input gate determines the new information added to the current cell state and ouput gate process previous hidden states and current cell state to determie which information to output in the current cell state. These gates in the LSTM architecture helps maintain long term dependencies and adress the vanishing gradient problem and work effectively in sequential text sentimental analysis.

## Objective 
The primary goal of this project is to build an automated pipeline that preprocess textual data, builds a predictive model using pyTorch module, evaluate and validate the performance of the model. This workflow allows for batch processing of extensive text datasets using PyTorch's module, and it evaluates the training and validation loss of the developed model to help identify areas for improvement before deploying it in a production environment.

## Data 
The  raw information collected from users as  shown in the [review dataset](Data/review_data.csv)  is processed to extract the necessary columns for model building in this project, specifically `content` and `score`, which respectively represent the review text and the rating given by the user based on their review. The `score` which is the target column in this project is a categorical variable that contains values between 0-5, reflecting the sentiment expressed in the review.

## Model pipeline 

1. Preprocessing the dataset:
- Clean the text data by removing stopwords and stemming text to ensure an appropriate format.
- Resampling the target variable if an imbalanced data distribution is found.
- Encode categorical labels to numerical values which is in the compatible format for machine learning algorithms.
2. Tokenization
  - Generate word tokens.
  - Generate word index 


### Tokenization 
While deep learning model is built using pyTorch module due to its flexible model-building capabilities, the Tensorflow is utilized for word tokenization of the textual data as its static graphs optimize the tokenization process



everage Best Tools: You can utilize TensorFlow’s robust tokenization tools and PyTorch’s flexible model-building capabilities, combining the strengths of both frameworks.

Flexibility: PyTorch’s dynamic computation graphs make it easier to experiment and debug during model development, while TensorFlow’s static graphs optimize the tokenization process.

Efficiency: TensorFlow’s efficient tokenization and preprocessing tools can handle large datasets effectively, and PyTorch’s intuitive API simplifies the model training process.

Scalability: TensorFlow’s tokenization tools are designed to scale well, and PyTorch’s model-building capabilities can be easily adapted for various architectures and tasks.

Community Support: Both TensorFlow and PyTorch have large, active communities, providing extensive resources, tutorials, and support for implementing this hybrid approach.

Production Readiness: TensorFlow’s serving capabilities ensure that the tokenization process is optimized for production, while PyTorch’s TorchServe can efficiently deploy models.

















































