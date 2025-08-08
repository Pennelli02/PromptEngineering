from datetime import datetime
import json
from pathlib import Path


def initMetrics():
    counters = {
        "tp": 0, "tn": 0, "fp": 0, "fn": 0, "er": 0,
        "rejection_real": 0, "rejection_fake": 0
    }
    return counters


def analizeMetrics(counters, images_with_labels, prompt, systemPrompt, oneShot, oneShotMessage, real_images,
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
