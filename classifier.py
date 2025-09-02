import json
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
from ollama import Client
from PIL import Image
import torch
from sklearn.cluster import KMeans
import numpy as np

client = Client()


def extract_prediction_from_text(text):
    """
    Estrae la predizione da testo non JSON usando keyword sia in inglese che italiano.
    Restituisce 'real face', 'generated', o None se non trova nulla.
    Utilizza le stesse keyword della versione originale ma fa un conteggio ponderato.
    """
    text_lower = text.lower()

    # Keyword originali
    real_keywords = [
        "real", "real face", "no generated", "not see any artifacts", "no artifacts",
        "reale", "volto reale", "non generato", "non vedo artefatti", "nessun artefatto",
        "non ci sono artefatti", "non indica la presenza di artefatti"
    ]

    fake_keywords = [
        "generated", "fake", "generated face", "no real face", "there are some artifacts",
        "generato", "falso", "volto generato", "nessun volto reale", "ci sono artefatti"
    ]

    # Conteggio match
    real_score = sum(1 for kw in real_keywords if re.search(rf"\b{re.escape(kw)}\b", text_lower))
    fake_score = sum(1 for kw in fake_keywords if re.search(rf"\b{re.escape(kw)}\b", text_lower))

    if real_score > fake_score:
        return "real face"
    elif fake_score > real_score:
        return "generated"
    else:
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


# SOLUZIONE USANDO GEMMA3
# --- Funzione di chunking testo ---
# def chunk_list(lst, chunk_size):
#     return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
#
#
# def summarize_uncertain_patterns_large(client, explanations, chunk_size=50, model="gemma3:1b"):
#     chunks = chunk_list(explanations, chunk_size)
#     chunk_summaries = []
#
#     for idx, chunk in enumerate(chunks):
#         chunk_text = "\n".join(chunk)
#         prompt = (
#                 "Analizza queste spiegazioni di risposte incerte e genera un elenco puntato "
#                 "dei pattern ricorrenti che portano il modello a classificare le immagini come '[uncertain]'. "
#                 "Indica chiaramente i problemi visivi o ambiguità comuni in ogni punto:\n\n"
#                 + chunk_text +
#                 "\n\nFormato desiderato:\n- Pattern 1: descrizione\n- Pattern 2: descrizione\n..."
#         )
#         response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
#         chunk_summary = response.get("content", "").strip()
#         chunk_summaries.append(chunk_summary)
#
#     if len(chunk_summaries) == 1:
#         return chunk_summaries[0]
#
#     combined_text = "\n".join(chunk_summaries)
#     final_prompt = (
#             "Sulla base dei seguenti riassunti di pattern ricorrenti nelle risposte incerte, "
#             "fornisci un unico elenco puntato sintetico dei pattern principali:\n\n"
#             + combined_text +
#             "\n\nFormato desiderato:\n- Pattern 1: descrizione\n- Pattern 2: descrizione\n..."
#     )
#     final_response = client.chat(model=model, messages=[{"role": "user", "content": final_prompt}])
#     final_summary = final_response.get("content", "").strip()
#
#     return final_summary
#
#
# def count_patterns_from_bullets(summary_text):
#     lines = summary_text.splitlines()
#     patterns = []
#     for line in lines:
#         line = line.strip()
#         if line.startswith("-"):
#             pattern = line.lstrip("- ").split(":")[0].strip()
#             patterns.append(pattern)
#     return Counter(patterns)


# SOLUZIONE USANDO GOOGLE/FLAN-T5-SMALL
# --- Funzione per estrarre embedding con flan-t5-small ---
def get_embeddings(texts, model_name="google/flan-t5-small", device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # media dei token embedding come rappresentazione
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)
