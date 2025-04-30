import os
import torch
import fitz
import unicodedata
from docx import Document
from dotenv import load_dotenv
from collections import Counter
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
from openai import OpenAI

LABELS_4 = ["None", "Mild", "High", "Max"]
LABELS_3 = ["None", "High", "Max"]
SCORE_MAP_4 = {"None": 1, "Mild": 3, "High": 6, "Max": 9}
SCORE_MAP_3 = {"None": 1, "High": 6, "Max": 9}

PRIMARY_MODEL_PATH = "./xlmr-4class-toxic-finetuned-epoch10"
RECHECK_HIGH_MAX_PATH = "./xlmr-4class-toxic-model-324"
RECHECK_MILD_PATH = "./xlmr-epoch10-mild-high-finetuned"
THREE_CLASS_MODEL_PATH = "./xlmr-3class-toxic-finetuned"
GPT_MODEL_ID_A = "gpt-4"

CONF_THRESH = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
primary_tokenizer = XLMRobertaTokenizer.from_pretrained(PRIMARY_MODEL_PATH)
primary_model = XLMRobertaForSequenceClassification.from_pretrained(PRIMARY_MODEL_PATH).to(DEVICE).eval()
highmax_tokenizer = XLMRobertaTokenizer.from_pretrained(RECHECK_HIGH_MAX_PATH)
highmax_model = XLMRobertaForSequenceClassification.from_pretrained(RECHECK_HIGH_MAX_PATH).to(DEVICE).eval()
mildhigh_tokenizer = XLMRobertaTokenizer.from_pretrained(RECHECK_MILD_PATH)
mildhigh_model = XLMRobertaForSequenceClassification.from_pretrained(RECHECK_MILD_PATH).to(DEVICE).eval()
three_tokenizer = XLMRobertaTokenizer.from_pretrained(THREE_CLASS_MODEL_PATH)
three_model = XLMRobertaForSequenceClassification.from_pretrained(THREE_CLASS_MODEL_PATH).to(DEVICE).eval()

load_dotenv()
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

def clean_text(text):
    return unicodedata.normalize("NFKC", text).replace("\xa0", " ").replace("\u200b", "").strip()

def extract_paragraphs(file):
    if file.name.endswith(".docx"):
        doc = Document(file)
        return [clean_text(p.text) for p in doc.paragraphs if len(p.text.strip()) >= 15]
    elif file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        return [clean_text(p) for p in text.split("\n\n") if len(p.strip()) >= 15]
    return []

def predict_with_conf(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        return int(probs.argmax()), probs

def detect_direction_a(paragraphs):
    results = []
    for para in paragraphs:
        pred_main, _ = predict_with_conf(primary_model, primary_tokenizer, para)
        final = pred_main
        if pred_main in [2, 3]:
            pred_review, prob_review = predict_with_conf(highmax_model, highmax_tokenizer, para)
            delta = float(prob_review[3] - prob_review[2])
            final = 3 if pred_review == 3 and delta > CONF_THRESH else 2
        elif pred_main == 1:
            pred_mild, prob_mild = predict_with_conf(mildhigh_model, mildhigh_tokenizer, para)
            if pred_mild == 1 and prob_mild[1] - prob_mild[0] > CONF_THRESH:
                final = 2
        results.append(final)
    return results

def detect_direction_b(paragraphs):
    results = []
    for para in paragraphs:
        pred, _ = predict_with_conf(three_model, three_tokenizer, para)
        results.append(pred)
    return results

def stars(score):
    full = int(score // 2)
    return "⭐" * full + "✩" * (5 - full)

def score_to_stars(score):
    val = int(round(score / 2))
    return "⭐" * val + "✩" * (5 - val)

def gpt_4_analysis(text, model_label):
    prompt = f"""You are a multilingual content moderation assistant. A machine learning model classified the following text as **{model_label}**. 

Your job is to independently assess the severity of four specific toxicity dimensions.  
Use the following scale for each (0–10):

- 0: Not present at all  
- 3–4: Mildly present  
- 5–7: Noticeably present  
- 8–10: Strongly and explicitly present  

Do not base your judgment solely on the model label. Analyze the text carefully and assign each score independently. Avoid giving all 10s unless fully justified.

Please rate and explain:
- Bias
- Xenophobia
- Misinformation
- Cultural Superiority

Text:
{text}
"""
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL_ID_A,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT-4 Error] {e}"

# Streamlit UI
st.set_page_config(page_title="Toxicity Detection A+B", layout="wide")
st.title("🌍 Multilingual Toxicity Detection System")

mode = st.radio("Select Detection Mode:", ["Direction A: Document Analysis (4-class)", "Direction B: Human-labeled Data (3-class)"])
uploaded_file = st.file_uploader("Upload a PDF or DOCX file below for toxicity analysis.", type=["pdf", "docx"])
user_input = st.text_area("Or paste some text here:")

if st.button("Analyze"):
    paragraphs = []
    if uploaded_file:
        paragraphs = extract_paragraphs(uploaded_file)
        text = "\n".join(paragraphs)
    elif user_input.strip():
        text = user_input.strip()
        paragraphs = [clean_text(text)]
    else:
        st.warning("Please upload a file or paste some text.")
        st.stop()

    if mode.startswith("Direction A"):
        preds = detect_direction_a(paragraphs)
        label = LABELS_4[Counter(preds).most_common(1)[0][0]]
        raw_score = SCORE_MAP_4[label]
    else:
        preds = detect_direction_b(paragraphs)
        label = LABELS_3[Counter(preds).most_common(1)[0][0]]
        raw_score = SCORE_MAP_3[label]

    scaled_score = round((raw_score / 9) * 10, 1)
    st.success(f"🧠 Final Document Toxicity Level: **{label}**  {stars(scaled_score)}")
    st.markdown(f"**Toxicity Score:** {scaled_score} / 10")

    gpt_out = gpt_4_analysis(text, label)
    st.markdown("### 🧠 GPT Analysis with Star Ratings")
    for line in gpt_out.splitlines():
        if ":" in line:
            prefix, body = line.split(":", 1)
            try:
                score = int("".join(filter(str.isdigit, body.strip().split(" ")[0])))
                st.markdown(f"**{prefix.strip()}**: {score_to_stars(score)} ({score})")
            except:
                st.markdown(f"**{prefix.strip()}**: {body.strip()}")
        else:
            st.markdown(line)
            # streamlit run "Apr18Prototype.py"