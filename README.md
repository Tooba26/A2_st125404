# A2_Language_Model
Due to large size of models, I have uploaded on the drive: https://drive.google.com/drive/folders/1t-9Ar3suiY6NtadSKTwdhkt88394smzY?usp=sharing
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
The hidden states of the LSTM are initialized to zeros at the beginning of each epoch.

For each batch of input (src) and target (target), The hidden state is detached to prevent backpropagation through time beyond the current batch.
The forward pass computes predictions using the LSTM and linear layers.
The predictions are reshaped to match the dimensions required by the loss function.
The loss is computed, and gradients are calculated via loss.backward().
Gradients are clipped using torch.nn.utils.clip_grad_norm_.
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

# Task 3
The web app is developed on Streamlit and styled using CSS.
Demo of Web App: https://drive.google.com/file/d/1VjvmQUvBDStcCQxuTwkQvXAvyvKmDpyc/view?usp=drive_link
1. Application Navigation (app.py)
- The app.py file initializes a Streamlit session state to manage navigation across multiple pages:

- Home Page: Displays navigation buttons.
- Book Page: For predictive text generation using one language model.
- Tel Page: For predictive text generation using another language model.
- Buttons (Go to Book Page and Go to Tel Page) allow the user to navigate between pages by setting the session state (st.session_state.page).

Based on the page value in the session state:

book.py and tel.py scripts are executed dynamically using exec.

2. Book Page (pages/book.py)
The Book Page performs predictive text generation using a pre-trained LSTM-based language model.
- Loads a pre-trained LSTM model (best-val-lstm_lm.pt) using PyTorch.
- The model is set to evaluation mode and moved to the appropriate device (CPU or GPU).
- Loads a vocabulary (vocab.pkl) for token-to-index and index-to-token conversions.
- Uses the tokenizer (basic_english) to tokenize the input text.
- The generate function takes a prompt, maximum sequence length, and temperature as inputs.
- Converts the prompt into token indices.
- Feeds tokens into the LSTM model in a loop to predict the next token until the maximum sequence length or <eos> token is reached.
- Probabilities of the next word are adjusted using the temperature parameter to control diversity:
- Higher temperature → More diverse output (less certainty).
- Lower temperature → Less diverse (more focused output).
- Converts token indices back into words to form the final generated text.

Users enter a starting prompt and adjust parameters (maximum text length, temperature) using input fields and sliders.
Clicking the Generate button triggers the text generation function and displays the output.

3. Tel Page (pages/tel.py)
The Tel Page is structured similarly to the Book Page but uses a different LSTM model (best-val-lstm_phone_seq.pt) and vocabulary.

Differences:
- Vocabulary size is different (10531 vs. 10727).

4. Language Model Interface

- Both pages use a custom LSTM-based language model, defined in model.py. The process involves:
- Loading Pre-Trained Models

- LSTM models with specific configurations (embedding size, hidden size, layers, dropout) are instantiated and loaded with pre-trained weights.
Hidden State Initialization:
- The init_hidden method of the model initializes the hidden and cell states of the LSTM for text generation.
- Input text is tokenized into indices using the vocabulary.
- Model outputs probabilities for the next token in the sequence.
- Multinomial sampling is used to pick the next token based on probabilities, controlled by temperature.
-The Streamlit UI enables real-time interaction, letting users customize inputs and immediately see the generated output.

