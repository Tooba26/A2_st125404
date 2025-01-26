import streamlit as st
import torch
from torch.nn.functional import softmax
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import pickle
from model import LSTMLanguageModel  # Make sure this matches your file

st.markdown(
    """
    <style>
    .stApp {
        background: -webkit-linear-gradient(to right, #24243e, #302b63, #0f0c29);  /* Chrome 10-25, Safari 5.1-6 */
        background: linear-gradient(to right, #24243e, #302b63, #0f0c29); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
        color: white;
    }

    [data-testid="stSidebar"] {
        background-color: #1f1f3f; 
        color: #ffffff; 
    }
    </style>
    """,
unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Navigation")  # This will always appear first
    st.markdown("Choose an option from above option ⬆️")
    
# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMLanguageModel(vocab_size=10531, emb_dim=1024, hid_dim=1024, num_layers=2, dropout_rate=0.65)
model.load_state_dict(torch.load('saved_model/best-val-lstm_phone_seq.pt', map_location=device))
model.to(device)
model.eval()

# Load the vocabulary
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

tokenizer = get_tokenizer("basic_english")

def generate(prompt, max_seq_len, temperature):
    tokens = tokenizer(prompt)
    indices = [vocab[token] for token in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for _ in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break
            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[idx] for idx in indices]
    return " ".join(tokens)

# Streamlit UI
st.title("📞 Switchboard-1 Release 2")

prompt = st.text_input("✏️ Enter the starting text (e.g., 'I am'):", value="")

with st.expander("⚙️ Optional: Customize Settings"):
    max_seq_len = st.slider("Max Length of Generated Text", min_value=10, max_value=100, value=30)
    temperature = st.slider("Diversity (Temperature)", min_value=0.5, max_value=1.5, step=0.1, value=1.0)

max_seq_len  = max_seq_len if 'max_seq_len' in locals() else 30
temperature = temperature if 'temperature' in locals() else 1.0

if st.button("Generate"):
    if prompt.strip() == "":
        st.warning("Please enter some starting text.")
    else:
        generated_text = generate(prompt, max_seq_len, temperature)
        st.success("Generated Text:")
        st.write(generated_text)



