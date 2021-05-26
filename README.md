 # Dialogue Act Tagging

Dialogue act (DA) tagging is an important step in the process of developing dialog systems. DA tagging is a problem usually solved by supervised machine learning approaches that all require large amounts of hand labeled data. A wide range of techniques have been investigated for DA tagging. In this project, I explore two approaches to DA classification. Here I am using the Switchboard Dialog Act Corpus for training. Corpus can be downloaded from http://compprag.christopherpotts.net/swda.html.

There are 43 tags in this dataset. Some of the tags are Yes-No-Question('qy'), Statement-non-opinion('sd') and Statement-opinion('sv'). Tags information can be found here http://compprag.christopherpotts.net/swda.html#tags. 

Model 1 -

The first approach we'll try is to treat DA tagging as a standard multi-class text classification task.
Each utterance will be treated independently as a text to be classified with its DA tag label. This model has an architecture of:

    Embedding
    BLSTM
    Fully Connected Layer
    Softmax Activation

The model architecture is as follows: Embedding Layer (to generate word embeddings) Next layer Bidirectional LSTM. Feed forward layer with number of neurons = number of tags. Softmax activation to get the probabilities
The overall accuracy I got was 69%, an effective accuracy for this task.

Model 2 - Balanced Network

One thing we can do to try to improve performance is therefore to balance the data more sensibly. As the dataset is highly imbalanced, we can simply weight the loss function in training, to weight up the minority classes proportionally to their underrepresentation.
The overall accuracy decreased drastically to 34% for model 2.

Other ways to handle imbalanced classes

    Under-sampling: Under-sampling can be used to decrease the instances of majority classes untill it is comparable 
    with the minority class. But as this method removes the data from dataset, some usseful information may be lost.

    Over-sampling: Over-sampling can be used to increase the isntances of minority classes on the training set by duplication. 
    The advantage here is that in over-sampling there is no loss of information, whereas there is a chance that model becomes prone to overfitting.

Using Context for Dialog Act Classification

The second approach we will try is a hierarchical approach to DA tagging. We expect there is valuable sequential information among the DA tags. So in this section we apply a BiLSTM on top of the utterance CNN representation. The CNN model learns textual information in each utterance for DA classification, acting like the text classifier from Model 1 above. Then we use a bidirectional-LSTM (BLSTM) above that to learn how to use the context before and after the current utterance to improve the output.


This model has an architecture of:

    Word Embedding
    CNN
    Bidirectional LSTM
    Fully-Connected output
For the CNN model I got the accuracy of over 71%.

