import os
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
def createJSONMeanStats(dirList):
    # Dati aggregati: {prompt_type: {"accuracy": [], "precision": [], "recall": []}}
    stats = defaultdict(lambda: {"accuracy": [], "precision": [], "recall": []})
    for dir in dirList:
        fileList = glob.glob(os.path.join(dir, "*.json"))
        for file in fileList:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            metrics = data["metrics"]

            baseName = os.path.splitext(os.path.basename(file))[0]
            parts = baseName.split("_")

            if len(parts) > 2:
                modello = f"{parts[1]}-{parts[2]}"
            elif len(parts) > 1:
                modello = parts[1]
            else:
                modello = "ModelloSconosciuto"

            promptType = f"{parts[3]}-{parts[4]}"

            stats[promptType]["accuracy"].append(metrics["accuracy"])
            stats[promptType]["precision"].append(metrics["precision"])
            stats[promptType]["recall"].append(metrics["recall"])

    # Calcola le medie
    results = []
    for promptType, values in stats.items():
        results.append({
            "prompt": promptType,
            "accuracy_mean": sum(values["accuracy"]) / len(values["accuracy"]) if values["accuracy"] else 0,
            "precision_mean": sum(values["precision"]) / len(values["precision"]) if values["precision"] else 0,
            "recall_mean": sum(values["recall"]) / len(values["recall"]) if values["recall"] else 0,
        })

    output_filename = f"resultsJSON/mean_stats_{modello}_{len(dirList)}.json"

    # Salva su file
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"file salvato in: {output_filename}")
