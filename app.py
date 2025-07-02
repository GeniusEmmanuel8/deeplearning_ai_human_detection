"""
AI vs Human Text Detection App
Author: Emmanuel Ojo
Description: Streamlit app to classify text as AI- or human-written using deep learning models (CNN, LSTM, RNN).
"""
from io import StringIO
import streamlit as st
import torch
import pickle
import os
import numpy as np
import PyPDF2
import docx

from models.cnn import CNNTextClassifier
from models.lstm import LSTMTextClassifier
from models.rnn import RNNTextClassifier

def default_clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def load_preprocessing_objects():
    """Load all preprocessing objects and parameters from the models directory."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    def load_pkl(name):
        with open(os.path.join(models_dir, name), "rb") as f:
            return pickle.load(f)
    objects = {}
    try:
        objects['tokenizer'] = load_pkl("tokenizer.pkl")
    except Exception:
        objects['tokenizer'] = None
    try:
        objects['clean_text_func'] = load_pkl("clean_text_func.pkl")
    except Exception:
        objects['clean_text_func'] = None
    if objects['clean_text_func'] is None:
        objects['clean_text_func'] = default_clean_text
    try:
        objects['lemmatize_text_func'] = load_pkl("lemmatize_text_func.pkl")
    except Exception:
        objects['lemmatize_text_func'] = None
    try:
        objects['lemmatizer_obj'] = load_pkl("lemmatizer_obj.pkl")
    except Exception:
        objects['lemmatizer_obj'] = None
    try:
        objects['remove_stopwords_func'] = load_pkl("remove_stopwords_func.pkl")
    except Exception:
        objects['remove_stopwords_func'] = None
    try:
        objects['stop_words_set'] = load_pkl("stop_words_set.pkl")
    except Exception:
        objects['stop_words_set'] = None
    try:
        objects['MAX_LEN'] = load_pkl("MAX_LEN.pkl")
    except Exception:
        objects['MAX_LEN'] = 256
    try:
        objects['VOCAB_SIZE'] = load_pkl("VOCAB_SIZE.pkl")
    except Exception:
        objects['VOCAB_SIZE'] = 20000
    try:
        objects['NUM_CLASSES'] = load_pkl("NUM_CLASSES.pkl")
    except Exception:
        objects['NUM_CLASSES'] = 2
    return objects

def preprocess_text(text):
    """Apply all preprocessing steps to the input text and return a tensor."""
    objs = load_preprocessing_objects()
    if objs['clean_text_func'] is not None and callable(objs['clean_text_func']):
        text = objs['clean_text_func'](text)
    if objs['remove_stopwords_func'] is not None and callable(objs['remove_stopwords_func']) and objs['stop_words_set']:
        text = objs['remove_stopwords_func'](text, objs['stop_words_set'])
    if objs['lemmatize_text_func'] is not None and callable(objs['lemmatize_text_func']) and objs['lemmatizer_obj']:
        text = objs['lemmatize_text_func'](text, objs['lemmatizer_obj'])
    if objs['tokenizer'] is not None and hasattr(objs['tokenizer'], 'texts_to_sequences'):
        seq = objs['tokenizer'].texts_to_sequences([text])
    else:
        seq = [[0]]
    max_len = objs['MAX_LEN'] if objs['MAX_LEN'] else 256
    if len(seq[0]) < max_len:
        seq[0] = seq[0] + [0] * (max_len - len(seq[0]))
    else:
        seq[0] = seq[0][:max_len]
    arr = np.array(seq)
    tensor = torch.tensor(arr, dtype=torch.long)
    return tensor

def load_cnn_model():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    with open(os.path.join(models_dir, "cnn_hparams.pkl"), "rb") as f:
        hparams = pickle.load(f)
    with open(os.path.join(models_dir, "VOCAB_SIZE.pkl"), "rb") as f:
        VOCAB_SIZE = pickle.load(f)
    with open(os.path.join(models_dir, "NUM_CLASSES.pkl"), "rb") as f:
        NUM_CLASSES = pickle.load(f)
    model = CNNTextClassifier(VOCAB_SIZE, **hparams, num_classes=NUM_CLASSES)
    weight_path = os.path.join(models_dir, "cnn_state_dict.pth")
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_lstm_model():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    with open(os.path.join(models_dir, "lstm_hparams.pkl"), "rb") as f:
        hparams = pickle.load(f)
    with open(os.path.join(models_dir, "VOCAB_SIZE.pkl"), "rb") as f:
        VOCAB_SIZE = pickle.load(f)
    with open(os.path.join(models_dir, "NUM_CLASSES.pkl"), "rb") as f:
        NUM_CLASSES = pickle.load(f)
    model = LSTMTextClassifier(VOCAB_SIZE, **hparams, num_classes=NUM_CLASSES)
    weight_path = os.path.join(models_dir, "lstm_state_dict.pth")
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_rnn_model():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    with open(os.path.join(models_dir, "rnn_hparams.pkl"), "rb") as f:
        hparams = pickle.load(f)
    with open(os.path.join(models_dir, "VOCAB_SIZE.pkl"), "rb") as f:
        VOCAB_SIZE = pickle.load(f)
    with open(os.path.join(models_dir, "NUM_CLASSES.pkl"), "rb") as f:
        NUM_CLASSES = pickle.load(f)
    model = RNNTextClassifier(VOCAB_SIZE, **hparams, num_classes=NUM_CLASSES)
    weight_path = os.path.join(models_dir, "rnn_state_dict.pth")
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_cnn(text):
    model = load_cnn_model()
    input_tensor = preprocess_text(text)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
        return "AI" if pred.item() == 1 else "Human", prob.squeeze().tolist()

def predict_lstm(text):
    model = load_lstm_model()
    input_tensor = preprocess_text(text)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
        return "AI" if pred.item() == 1 else "Human", prob.squeeze().tolist()

def predict_rnn(text):
    model = load_rnn_model()
    input_tensor = preprocess_text(text)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
        return "AI" if pred.item() == 1 else "Human", prob.squeeze().tolist()

def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            return text
        except Exception:
            return ""
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        try:
            doc = docx.Document(uploaded_file)
            text = " ".join([para.text for para in doc.paragraphs])
            return text
        except Exception:
            return ""
    elif file_type.startswith("text/"):
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            return stringio.read()
        except Exception:
            return ""
    else:
        return ""

# Streamlit UI
st.set_page_config(page_title="AI vs Human Text Detection", layout="wide")
st.title("AI vs Human Text Detection")
st.markdown("""
Instructions:
- Type or paste your text in the box below.
- Optionally, upload a PDF, Word, or text file. The extracted text will be appended to your input.
- Select a model and click Analyze Text to get a prediction.
""")
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ("CNN", "LSTM", "RNN")
)

st.subheader("Text Input")
input_text = st.text_area("Paste or type your text here:", height=200)

uploaded_file = st.file_uploader("Optionally, upload a PDF, Word, or Text file", type=["pdf", "docx", "txt"])

file_text = ""
if uploaded_file is not None:
    file_text = extract_text_from_file(uploaded_file)
    if file_text:
        st.info("Text extracted from uploaded file will be appended to your input.")
        input_text = (input_text + "\n" + file_text).strip() if input_text else file_text
    else:
        st.warning("Could not extract text from the uploaded file.")

if st.button("Analyze Text"):
    st.write("## Prediction Results")
    if not input_text.strip():
        st.warning("Please provide text input for analysis.")
    else:
        if model_choice == "CNN":
            pred, prob = predict_cnn(input_text)
            st.write(f"**Prediction:** {pred}")
            st.write(f"**Confidence:** {prob}")
        elif model_choice == "LSTM":
            pred, prob = predict_lstm(input_text)
            st.write(f"**Prediction:** {pred}")
            st.write(f"**Confidence:** {prob}")
        elif model_choice == "RNN":
            pred, prob = predict_rnn(input_text)
            st.write(f"**Prediction:** {pred}")
            st.write(f"**Confidence:** {prob}")
        else:
            st.info("Model not implemented.")
