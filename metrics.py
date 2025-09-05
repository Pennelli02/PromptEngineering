import os
import re
import shutil
from datetime import datetime
import json
from pathlib import Path
import glob


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

    # ================= One-class accuracy =================
    one_class_accuracy_real = counters["tn"] / (counters["tn"] + counters["fp"]) if (
            counters["tn"] + counters["fp"]) else 0
    one_class_accuracy_fake = counters["tp"] / (counters["tp"] + counters["fn"]) if (
            counters["tp"] + counters["fn"]) else 0
    # =====================================================

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
    print(f"One-class Accuracy (real images): {one_class_accuracy_real:.4f}")
    print(f"One-class Accuracy (fake images): {one_class_accuracy_fake:.4f}")

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
        "one_class_accuracy_real": one_class_accuracy_real,
        "one_class_accuracy_fake": one_class_accuracy_fake,
        "system_prompt": systemPrompt,
        "user_prompt": prompt,
        "oneShot": oneShot,
    }
    if oneShot:
        results["oneShotMessage"] = oneShotMessage
    return results


# funzione di update per inserire in tutti one-class-accuracy (non serve per fortuna)

def add_one_class_accuracy(json_file_path):
    """
    Legge un file JSON con struttura { "metrics": { ... } },
    calcola la one-class accuracy e aggiorna il JSON con i nuovi campi.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    metrics = data.get("metrics", None)
    if metrics is None:
        print(f"Errore: 'metrics' non trovato in {json_file_path}")
        return

    # Calcolo one-class accuracy
    TP = metrics.get("TP", 0)
    TN = metrics.get("TN", 0)
    FP = metrics.get("FP", 0)
    FN = metrics.get("FN", 0)

    one_class_accuracy_real = TP / (TP + FN) if (TP + FN) else 0
    one_class_accuracy_fake = TN / (TN + FP) if (TN + FP) else 0

    # Aggiorna il JSON
    metrics["one_class_accuracy_real"] = one_class_accuracy_real
    metrics["one_class_accuracy_fake"] = one_class_accuracy_fake

    # Salva di nuovo il JSON
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Aggiornato {json_file_path} con one-class accuracy.")


def process_json_folder(folder_path):
    """
    Processa tutti i file JSON in una cartella e aggiunge la one-class accuracy.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            add_one_class_accuracy(os.path.join(folder_path, file_name))


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
def createJSONMeanStats(folder_path):
    # Inizializza accumulatore
    aggregated = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "rejection_real_rate": [],
        "rejection_fake_rate": [],
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": [],
        "one_class_accuracy_real": [],
        "one_class_accuracy_fake": []
    }

    # Prendi tutti i file json nella cartella
    fileList = glob.glob(os.path.join(folder_path, "*.json"))

    if not fileList:
        print(" Nessun file JSON trovato nella cartella indicata.")
        return

    # Prendo il primo file per ricavare nome base
    first_file = os.path.basename(fileList[0])
    output_filename = re.sub(r"_\d{8}-\d{6}_result\.json$", "_mean-result.json", first_file)

    for file in fileList:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data["metrics"]

        # Calcolo one-class accuracy se mancante
        if "one_class_accuracy_real" not in metrics:
            total_real = metrics.get("TN", 0) + metrics.get("FP", 0)
            metrics["one_class_accuracy_real"] = metrics.get("TN", 0) / total_real if total_real else 0

        if "one_class_accuracy_fake" not in metrics:
            total_fake = metrics.get("TP", 0) + metrics.get("FN", 0)
            metrics["one_class_accuracy_fake"] = metrics.get("TP", 0) / total_fake if total_fake else 0

        # Accumula i valori
        for key in aggregated.keys():
            if key in metrics:
                aggregated[key].append(metrics[key])

    # Calcola la media per metriche continue
    mean_results = {"num_files": len(fileList)}
    for key in ["accuracy", "precision", "recall", "rejection_real_rate", "rejection_fake_rate",
                "one_class_accuracy_real", "one_class_accuracy_fake"]:
        values = aggregated[key]
        mean_results[f"{key}_mean"] = sum(values) / len(values) if values else 0

    # Somma totale per TP, TN, FP, FN
    for key in ["TP", "TN", "FP", "FN"]:
        mean_results[f"{key}_total"] = sum(aggregated[key]) if aggregated[key] else 0

    # Calcola F1 e F2-score
    precision = mean_results.get("precision_mean", 0)
    recall = mean_results.get("recall_mean", 0)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        beta = 2
        f2 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    else:
        f1, f2 = 0, 0

    mean_results["F1_score"] = f1
    mean_results["F2_score"] = f2

    # Salvataggio
    output_path = os.path.join(folder_path, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mean_results, f, indent=4, ensure_ascii=False)
    print(f"File salvato in: {output_path}")
    norm_path = os.path.normpath(folder_path)
    parts = norm_path.split(os.sep)
    sure_type = parts[1]
    prompt_folder = parts[-1]
    dest_dir = os.path.join("promptSection", sure_type.lower(), prompt_folder.replace("prompt", "Prompt"))
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, output_filename)
    shutil.copy(output_path, dest_path)
    print(f"Copiato anche in: {dest_path}")


if __name__ == "__main__":
    base_path = "JsonMeanStats/Uncertain/qwen7b"

    # for i in range(7):  # indici da 0 a 6
    #     for lang in ["Eng", "Ita"]:
    #         folder = os.path.join(base_path, f"prompt-{i}-{lang}")
    #         print(f" Elaboro: {folder}")
    #         createJSONMeanStats(folder)
    createJSONMeanStats("JsonMeanStats/OneShot/llava/real_example")