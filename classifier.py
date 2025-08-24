import json
import re

import nltk
# pip install sentence-transformers hdbscan transformers
import pandas as pd
from ollama import Client
from PIL import Image
# from sentence_transformers import SentenceTransformer
# import hdbscan
# from collections import Counter
from transformers import pipeline

client = Client()


def extract_prediction_from_text(text):
    """
    Cerca di estrarre prediction anche da testo non JSON usando regex.
    Riconosce sia inglese che italiano.
    Restituisce 'real face', 'generated', o None se non trova nulla.
    """
    text_lower = text.lower()

    # Liste di parole chiave in inglese e italiano
    real_keywords = [
        # inglese
        "real", "real face", "no generated", "not see any artifacts", "no artifacts",
        # italiano
        "reale", "volto reale", "non generato", "non vedo artefatti", "nessun artefatto", "non ci sono artefatti",
        "non indica la presenza di artefatti"
    ]

    fake_keywords = [
        # inglese
        "generated", "fake", "generated face", "no real face", "there are some artifacts"
        # italiano
        "generato", "falso", "volto generato", "nessun volto reale", "ci sono artefatti"
    ]

    # Prova a matchare "real"
    for kw in real_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text_lower):
            return "real face"

    # Prova a matchare "fake"
    for kw in fake_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text_lower):
            return "generated"

    return None


def analyze_image(img_path, lab, prompt, modelName, fewShot, few_shot_messages, systemPrompt, counters, showImages):
    result_entry = {
        "image_path": str(img_path),
        "ground_truth": "real" if lab == 1 else "fake",
        "prediction": None,
        "explanation": None,
        "error": None,
    }

    try:
        # Costruzione dei messaggi
        messages = []

        # Aggiungi system prompt come primo messaggio se serve
        if systemPrompt:
            messages.append({
                "role": "system",
                "content": systemPrompt
            })

        # Aggiungi few-shot messages se richiesto
        if fewShot and few_shot_messages:
            messages.extend(few_shot_messages)  # applicabile al momento su ollama

        # ONESHOT applicabile su  hugging face
        # if oneshot:
        #     msg=prompt.createOneShot(exampleImage, isFake, imagePath ,prompt, isItalian) # paramentri che prendiamo in ingresso
        #     messages.extend(msg)
        # else:
        #     #  continuare il normale procedimento
        # Aggiungi messaggio dell'utente con immagine e prompt
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [str(img_path)]
        })

        # Chiamata al modello
        response = client.chat(
            model=modelName,
            messages=messages
        )

        text = response['message']['content'].strip()

        print(f"\nRaw Response: {text}\n")

        # Pulizia markdown
        text_clean = re.sub(r"^```(?:json)?\s*([\s\S]*?)\s*```$", r"\1", text.strip(), flags=re.MULTILINE)

        text_clean = repair_json(text_clean)  # aggiusta il testo per il parsing

        try:
            parsed = json.loads(text_clean)
            result = parsed.get("result")
            explanation = parsed.get("explanation", None)

            # Gestisci lista/stringa se serve
            if isinstance(result, list) and result:
                result = result[0]

            prediction = str(result).strip().lower()
            result_entry["prediction"] = prediction
            result_entry["explanation"] = explanation

        except Exception as e:
            prediction = extract_prediction_from_text(text)
            result_entry["prediction"] = prediction
            result_entry["explanation"] = text  # Salva la raw response qui
            result_entry["error"] = f"Parsing failed, fallback prediction: {prediction}"
            print(f"Parsing failed, fallback prediction: {prediction}")


        if showImages:
            Image.open(img_path).show()

        # Decisione e valutazione
        if lab == 1:  # Real
            if prediction in {"real face", "real", "[real face]", "agreed", "[real]", "[no]", "no"}:
                counters["tn"] += 1
                print(" TN (real correctly identified)")
            elif prediction in {"generated", "[generated]", "didn't agree", "generated face", "[generated face]", "yes",
                                "[yes]"}:
                counters["fp"] += 1
                print(" FP (real misclassified as fake)")
            else:
                counters["rejection_real"] += 1
                print(" Rejection on real image")
        else:  # Fake
            if prediction in {"generated", "[generated]", "didn't agree", "generated face", "[generated face]", "yes",
                              "[yes]"}:
                counters["tp"] += 1
                print(" TP (fake correctly identified)")
            elif prediction in {"real face", "real", "[real face]", "agreed", "[real]", "[no]", "no"}:
                counters["fn"] += 1
                print(" FN (fake misclassified as real)")
            else:
                counters["rejection_fake"] += 1
                print(" Rejection on fake image")

    except Exception as e:
        print(f" Error on {img_path}: {e}")
        counters["er"] += 1
        result_entry["error"] = f"Runtime error: {e}"

    return result_entry, counters


def repair_json(text):
    # Bilancia parentesi graffe
    open_braces = text.count('{')
    close_braces = text.count('}')
    missing = open_braces - close_braces
    if missing > 0:
        text += '}' * missing

    # Tenta di chiudere virgolette aperte
    quote_count = text.count('"')
    if quote_count % 2 != 0:
        text += '"'

    # Rimuovi caratteri dopo ultima graffa chiusa (potrebbe esserci garbage)
    last_close = text.rfind('}')
    if last_close != -1:
        text = text[:last_close + 1]

    return text


# TODO informarsi attentamente per la tesi scritta
def repair_dates(results):
    df = pd.DataFrame(results)
    df['explanation'] = df['explanation'].fillna('').str.strip()

    if df.empty:
        print("Nessun dato trovato per il tipo selezionato.")
        exit()
    return df


# --- Funzione di chunking testo ---
nltk.download('punkt')


def chunk_text_by_sentence(text, max_len=1000):
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def hierarchical_summarize_with_prompt(explanations, max_chunk_len=1000):
    summarizer = pipeline("summarization", model="google/flan-t5-small")

    # Primo livello
    full_text = " ".join(explanations)
    chunks = chunk_text_by_sentence(full_text, max_chunk_len)
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Riassumendo chunk {i + 1}/{len(chunks)}...")
        prompt = (
                "Analizza i seguenti casi e individua le cause principali dell'incertezza del modello, "
                "descrivendo i problemi visivi o ambiguità ricorrenti:\n" + chunk
        )
        summary = summarizer(prompt, max_new_tokens=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    # Secondo livello (finché resta più di un riassunto)
    while len(summaries) > 1:
        combined_text = " ".join(summaries)
        chunks = chunk_text_by_sentence(combined_text, max_chunk_len)
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Riassumendo livello superiore, chunk {i + 1}/{len(chunks)}...")
            prompt = (
                    "Sulla base di questi riassunti, fornisci una sintesi unica dei pattern e cause più frequenti "
                    "che portano il modello a classificare come 'incerto':\n" + chunk
            )
            summary = summarizer(prompt, max_new_tokens=200, min_length=40, do_sample=False)[0]['summary_text']
            summaries.append(summary)

    return summaries[0]

# def summarizeALL(explanations):
#     summarizer = pipeline("summarization", model="google/flan-t5-small")
#     full_text = " ".join(explanations)
#     prompt = (f"Elenca e descrivi i motivi più frequenti per cui il modello ha classificato l'immagine come "
#               "'incerto', focalizzandoti su pattern comuni o problemi visivi presenti: " + full_text)
#     summary = summarizer(prompt, max_new_tokens=150, min_length=20, do_sample=False)[0]['summary_text']
#     return summary

# FixMe si può usare per il summerize and explaination gemma3:1b
