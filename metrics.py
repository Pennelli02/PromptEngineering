import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import glob
import numpy as np


def initMetrics():
    counters = {
        "tp": 0, "tn": 0, "fp": 0, "fn": 0, "er": 0,
        "rejection_real": 0, "rejection_fake": 0
    }
    return counters


def analyzeMetrics(counters, images_with_labels, prompt, systemPrompt, oneShot, oneShotMessage, real_images,
                   fake_images):
    total_classified = counters["tp"] + counters["tn"] + counters["fp"] + counters["fn"]
    accuracy = (counters["tp"] + counters["tn"]) / total_classified if total_classified else 0
    precision = counters["tp"] / (counters["tp"] + counters["fp"]) if (counters["tp"] + counters["fp"]) else 0
    recall = counters["tp"] / (counters["tp"] + counters["fn"]) if (counters["tp"] + counters["fn"]) else 0
    total_real = counters["tp"] + counters["fn"] + counters["rejection_real"]
    total_fake = counters["tn"] + counters["fp"] + counters["rejection_fake"]

    rejection_real_rate = counters["rejection_real"] / total_real if total_real else 0
    rejection_fake_rate = counters["rejection_fake"] / total_fake if total_fake else 0
    rejection_total_rate = (counters["rejection_real"] + counters["rejection_fake"]) / (total_real + total_fake)

    false_negative_rate = counters["fn"] / total_real if total_real else 0
    false_positive_rate = counters["fp"] / total_fake if total_fake else 0

    print("\n====== FINAL REPORT ======")
    print(f"Total processed: {len(images_with_labels)}")
    print(f"TP: {counters['tp']} | TN: {counters['tn']} | FP: {counters['fp']} | FN: {counters['fn']}")
    print(f"Rejections on real: {counters['rejection_real']} | Rejections on fake: {counters['rejection_fake']}")
    print(f"Text parsing errors: {counters['er']} ({(counters['er'] / len(images_with_labels)) * 100:.2f}%)\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"False Negative Rate (real->fake): {false_negative_rate * 100:.2f}%")
    print(f"False Positive Rate (fake->real): {false_positive_rate * 100:.2f}%")
    print(f"Rejection Rate on real images: {rejection_real_rate * 100:.2f}%")
    print(f"Rejection Rate on fake images: {rejection_fake_rate * 100:.2f}%")

    results = {
        "total_processed": len(images_with_labels),
        "total_real": real_images,
        "total_fake": fake_images,
        "TP": counters["tp"],
        "TN": counters["tn"],
        "FP": counters["fp"],
        "FN": counters["fn"],
        "rejection_real": counters["rejection_real"],
        "rejection_fake": counters["rejection_fake"],
        "text_parsing_errors": counters["er"],
        "text_parsing_error_rate": (counters["er"] / len(images_with_labels)) if len(images_with_labels) else 0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "rejection_real_rate": rejection_real_rate,
        "rejection_fake_rate": rejection_fake_rate,
        "rejection_total_rate": rejection_total_rate,
        "system_prompt": systemPrompt,
        "user_prompt": prompt,
        "oneShot": oneShot,
    }
    if oneShot:
        results["oneShotMessage"] = oneShotMessage
    return results


def saveAllJson(metrics, responses, PromptITA, modelName, i):
    outputData = {
        "metrics": metrics,
        "responses": responses
    }
    Path("resultsJSON").mkdir(exist_ok=True)

    # Imposta lingua
    language_tag = "ITA" if PromptITA else "ENG"

    # Timestamp per identificare diversi tentativi
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Pulisci MODEL_NAME da caratteri non ammessi nei nomi file
    safe_model_name = modelName.replace(":", "_").replace("/", "_")

    # Costruisci filename
    filename = f"resultsJSON/real-vs-fake_{safe_model_name}_PromptType-{i}_{language_tag}_{timestamp}_result.json"

    # Salva JSON
    with open(filename, "w") as f:
        json.dump(outputData, f, indent=4)

    print(f"Results saved to {filename}.")


# funzione che prende i valori di una cartella e ne fa la media
def createJSONMeanStats(folder_path, oneshot=False):
    # Inizializza accumulatore
    aggregated = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "false_negative_rate": [],
        "false_positive_rate": [],
        "rejection_real_rate": [],
        "rejection_fake_rate": []
    }

    # Prendi tutti i file json nella cartella
    fileList = glob.glob(os.path.join(folder_path, "*.json"))

    if not fileList:
        print("⚠️ Nessun file JSON trovato nella cartella indicata.")
        return

    # Prendo il primo file per ricavare nome base
    first_file = os.path.basename(fileList[0])

    # Rimpiazza la parte "_YYYYMMDD-HHMMSS_result.json" con "_mean-result.json"
    output_filename = re.sub(r"_\d{8}-\d{6}_result\.json$", "_mean-result.json", first_file)

    for file in fileList:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data["metrics"]

        # Accumula i valori
        for key in aggregated.keys():
            if key in metrics:
                aggregated[key].append(metrics[key])

    # Calcola la media
    mean_results = {
        "num_files": len(fileList)
    }
    for key, values in aggregated.items():
        mean_results[f"{key}_mean"] = sum(values) / len(values) if values else 0

    # Path completo di output
    output_path = os.path.join(folder_path, output_filename)

    # Salva su file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mean_results, f, indent=4, ensure_ascii=False)

    print(f" File salvato in: {output_path}")

    if not oneshot:
        # --------------------------
        # Copia nella cartella "promptSection"
        # --------------------------
        norm_path = os.path.normpath(folder_path)  # normalizza separatori
        parts = norm_path.split(os.sep)
        # Esempio: JsonMeanStats/Sure/gemma3/prompt-0-Eng
        # -> sure_type = Sure, model = gemma3, prompt_folder = prompt-0-Eng
        sure_type = parts[1]       # Sure / Uncertain
        model = parts[2]           # gemma3
        prompt_folder = parts[-1]  # prompt-0-Eng

        dest_dir = os.path.join("promptSection", sure_type.lower(), prompt_folder.replace("prompt", "Prompt"))
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, output_filename)
        shutil.copy(output_path, dest_path)

        print(f" Copiato anche in: {dest_path}")


if __name__ == "__main__":
    base_path = "JsonMeanStats/Sure/llava"

    for i in range(7):  # indici da 0 a 6
        for lang in ["Eng", "Ita"]:
            folder = os.path.join(base_path, f"prompt-{i}-{lang}")
            print(f"➡️ Elaboro: {folder}")
            createJSONMeanStats(folder)

