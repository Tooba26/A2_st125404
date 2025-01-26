# A2_Language_Model
# Task 1: Dataset Acquisition
1) The dataset chosen is a bookcorpus retrieved from huggingface.  Combined with the diff results, the manual inspection confirmed that each filename represents one unique book, thus BookCorpus contained at most 7,185 unique books. Each book in BookCorpus simply includes the full text from the ebook (often including preamble, copyright text, etc.).

https://huggingface.co/datasets/bookcorpus/bookcorpus 

Furthermore, I also tried to use another dataset which was Switchboard-1 Release 2. The Switchboard-1 Telephone Speech Corpus (LDC97S62) consists of approximately 260 hours of speech and was originally collected by Texas Instruments in 1990-1, under DARPA sponsorship. Switchboard is a collection of about 2,400 two-sided telephone conversations among 543 speakers (302 male, 241 female) from all areas of the United States. 
 
https://catalog.ldc.upenn.edu/LDC97S62

# Task 2: Model Training
1) **Steps taken to preprocess the text data.**
**Bookcorpus**
- The data consisted of only train dataset. After retrieving dataset, they were divided into training, testing and validation dataset.
- As the dataset consisted of large set of data but due to device limitation i took only some samples from it for training.

**Switchboard-1 Release 2**
- The dataset was stored in number of folders.
- For data preprocessing, first of all, all the data was combined in a single Dataframe from all the folders.
- Then the dataframe was converted to a plain dictionary with only text. - The data was split into train, test and validation set and stored in a dictionary. 
- It was then tokenized, numericalized and then trained.

2) **model architecture and the training process.**
**Model Architecture**

The LSTMLanguageModel is an RNN-based architecture for language modeling, primarily leveraging LSTMs (Long Short-Term Memory networks) to learn sequential patterns and long-term dependencies in text. The model begins with an embedding layer, which maps discrete token indices into dense continuous representations of size emb_dim. These embeddings are fed into a stack of LSTM layers (defined by num_layers), which process sequences step-by-step, maintaining hidden and cell states of size hid_dim to capture temporal dependencies. A dropout layer is applied to the embeddings and LSTM outputs to prevent overfitting. The final LSTM outputs are passed through a fully connected layer (fc) that projects them into a vector of size equal to the vocabulary (vocab_size), generating logits for token prediction. Custom initialization is used for embeddings, fully connected layers, and LSTM weights to ensure stable and effective training. Hidden and cell states are reinitialized or detached as needed to handle training efficiency and prevent gradient backpropagation across sequences. 

**Training**
Dataset Preparation:

The input data is tokenized and organized into batches of size [batch_size, seq_len]. The get_batch function retrieves input (src) and target sequences (target), where the target is the input sequence shifted by one time step.
Cross-Entropy Loss is used, which measures the difference between the predicted probability distribution and the true target distribution.
Adam Optimizer is used to update the model parameters.
A ReduceLROnPlateau scheduler reduces the learning rate when the validation loss plateaus to improve convergence.
Gradients are clipped to a maximum value (clip) to prevent exploding gradients, which are common in RNNs.
Training Loop:
Hidden State Initialization:
The hidden states of the LSTM are initialized to zeros at the beginning of each epoch.

For each batch of input (src) and target (target):
The hidden state is detached to prevent backpropagation through time beyond the current batch.
The forward pass computes predictions using the LSTM and linear layers.
The predictions are reshaped to match the dimensions required by the loss function.
Backpropagation:

The loss is computed, and gradients are calculated via loss.backward().
Gradients are clipped using torch.nn.utils.clip_grad_norm_.
Parameter Update:

The optimizer updates the model parameters using the computed gradients.

The loss is accumulated over all batches in the epoch to compute the average training loss.

The model is set to evaluation mode (model.eval()), and gradient computation is disabled (torch.no_grad()).

For each batch of validation data:
Predictions are generated and reshaped, and the loss is computed.

The average validation loss is computed over all batches.
Perplexity is used to evaluate the model's performance. It is defined as 
Hyperparameters
Embedding Dimension (emb_dim): 1024
Hidden Dimension (hid_dim): 1024
Number of Layers (num_layers): 2
Dropout Rate: 0.65
Sequence Length (seq_len): 50

The models were trained on **50 epochs.**
