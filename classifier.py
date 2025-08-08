import json
import re
from ollama import Client
from PIL import Image

client = Client()


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
            messages.extend(few_shot_messages)

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

        text_clean = repair_json(text_clean) # aggiusta il testo per il parsing

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
            counters["er"] += 1
            result_entry["error"] = f"Unexpected error during parsing: {e}"
            print(f"Unexpected error during parsing: {e}")
            return result_entry, counters

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
