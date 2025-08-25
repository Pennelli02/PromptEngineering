# parte dedicata ad un'eventuale stampa dei risultati serve per fare i grafici
# c'è bisogno di valutare maggiormente gli esperimenti tenere conto maggiormente delle statistiche.
# efficacia ITA
# efficacia ENG
# efficacia generale prompt --> (grafico a barre)
# tenere conto delle rejection + (magari) modello testuale di summerize
import json
import glob
import os
import re
from collections import Counter

from sklearn.cluster import KMeans
from transformers import pipeline

import classifier
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

nltk.download('stopwords')


# Fixme Suddividere il grafico di llava-7b in 2/3 grafici separati in base al tipo di prompt: veri, falsi,
#  neutri. Per esempio, io ho clusterizzato i prompt in questo modo: neutri:1,2,4 - veri:5,6 - falsi:3,7.
def plotStatsPrompt(dirName, Uncertain=False, OneShot=False):
    cartella = dirName  # metti qui la tua cartella
    fileList = glob.glob(os.path.join(cartella, "*.json"))

    risultati = []
    for file_path in fileList:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data["metrics"]

        baseName = os.path.splitext(os.path.basename(file_path))[0]
        parts = baseName.split("_")

        # Prendi il modello come seconda parte (indice 1)
        if len(parts) > 2:
            modello = f"{parts[1]}-{parts[2]}"
        elif len(parts) > 1:
            modello = parts[1]
        else:
            modello = "ModelloSconosciuto"

        # Trova il prompt
        prompt = next((p.replace("PromptType-", "Prompt-") for p in parts if p.startswith("PromptType-")),
                      "PromptSconosciuto")

        # Trova indice prompt per prendere la lingua subito dopo
        indice_prompt = next((i for i, p in enumerate(parts) if p.startswith("PromptType-")), None)
        if indice_prompt is not None and indice_prompt + 1 < len(parts):
            lingua = parts[indice_prompt + 1]
        else:
            lingua = "LinguaSconosciuta"

        nome_completo = f"{prompt}-{lingua}"

        risultati.append({
            "nome": nome_completo,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "modello": modello
        })
    # ===== 3. CREAZIONE GRAFICO A BARRE COMPARATIVO =====
    etichette = [r["nome"] for r in risultati]
    # Moltiplico i valori per 100 per avere percentuali
    accuracy = [r["accuracy"] * 100 for r in risultati]
    precision = [r["precision"] * 100 for r in risultati]
    recall = [r["recall"] * 100 for r in risultati]

    x = np.arange(len(etichette)) * 2  # posizioni sull'asse x
    width = 0.5  # larghezza barre

    fig, ax = plt.subplots(figsize=(12, 4))
    rects1 = ax.bar(x - 1.5 * width, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x - 0.5 * width, precision, width, label='Precision')
    rects3 = ax.bar(x + 0.5 * width, recall, width, label='Recall')

    autolabel(rects1, ax, 5, False)
    autolabel(rects2, ax, 5, False)
    autolabel(rects3, ax, 5, False)

    modelli_presenti = list(set(r["modello"] for r in risultati))
    titolo_modello = ", ".join(modelli_presenti)
    if Uncertain:
        if OneShot:
            ax.set_title(f"Confronto Metriche per modello {titolo_modello} - Incertezza e OneShot")
            nome_file = f"{titolo_modello}-Uncertain-Oneshot.png"
        else:
            ax.set_title(f"Confronto Metriche per modello {titolo_modello} - Incertezza")
            nome_file = f"{titolo_modello}-Uncertain.png"
    else:
        if OneShot:
            ax.set_title(f"Confronto Metriche per modello {titolo_modello} - OneShot")
            nome_file = f"{titolo_modello}-Oneshot.png"
        else:
            ax.set_title(f"Confronto Metriche per modello {titolo_modello}")
            nome_file = f"{titolo_modello}.png"

    ax.set_ylabel("Valore %")
    ax.set_xticks(x)
    ax.set_xticklabels(etichette, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    cartella_grafici = "plots/promptBar/"
    os.makedirs(cartella_grafici, exist_ok=True)  # crea la cartella se non esiste

    # Percorso completo file immagine
    path_grafico = os.path.join(cartella_grafici, nome_file)

    # Salva il grafico
    plt.savefig(path_grafico, dpi=300, bbox_inches='tight')  # dpi=300 per alta qualità


def autolabel(rects, ax, fontsize, decimal):
    for rect in rects:
        height = rect.get_height()
        if decimal:
            title = f" {height:.2f}"
        else:
            title = f" {height:.1f}"
        ax.annotate(title,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=fontsize, fontweight='bold', color='black')


def graphItaEng(dirName, Uncertain=False, OneShot=False):
    cartella = dirName
    fileList = glob.glob(os.path.join(cartella, "*.json"))
    modello = None
    # Dizionario per accumulare metriche per lingua
    datiLingua = {
        "ENG": {"accuracy": [], "precision": [], "recall": []},
        "ITA": {"accuracy": [], "precision": [], "recall": []}
    }
    for file_path in fileList:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data["metrics"]

        # Estrai la lingua dal nome file
        baseName = os.path.splitext(os.path.basename(file_path))[0]
        parts = baseName.split("_")
        modello = f"{parts[1]}-{parts[2]}"

        lingua = parts[4]
        # Accumula metriche
        datiLingua[lingua]["accuracy"].append(metrics["accuracy"])
        datiLingua[lingua]["precision"].append(metrics["precision"])
        datiLingua[lingua]["recall"].append(metrics["recall"])
    # Calcola le medie
    lingue = list(datiLingua.keys())
    accuracy_medie = [np.mean(datiLingua[ling]["accuracy"]) if datiLingua[ling]["accuracy"] else 0 for ling in lingue]
    precision_medie = [np.mean(datiLingua[ling]["precision"]) if datiLingua[ling]["precision"] else 0 for ling in
                       lingue]
    recall_medie = [np.mean(datiLingua[ling]["recall"]) if datiLingua[ling]["recall"] else 0 for ling in lingue]

    # Grafico a barre
    x = np.arange(len(lingue))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, accuracy_medie, width, label="Accuracy")
    rects2 = ax.bar(x, precision_medie, width, label="Precision")
    rects3 = ax.bar(x + width, recall_medie, width, label="Recall")

    ax.set_xticks(x)
    ax.set_xticklabels(lingue)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valore medio")
    if Uncertain:
        if OneShot:
            ax.set_title(f"Confronto metriche medie ENG vs ITA ({modello}) - Incertezza e OneShot")
            nome_file = f"{modello}-Uncertain-Oneshot.png"
        else:
            ax.set_title(f"Confronto metriche medie ENG vs ITA ({modello}) - Incertezza")
            nome_file = f"{modello}-Uncertain.png"
    else:
        if OneShot:
            ax.set_title(f"Confronto metriche medie ENG vs ITA ({modello}) - OneShot")
            nome_file = f"{modello}-Oneshot.png"
        else:
            ax.set_title(f"Confronto metriche medie ENG vs ITA ({modello})")
            nome_file = f"{modello}.png"
    ax.legend()

    autolabel(rects1, ax, 10, True)
    autolabel(rects2, ax, 10, True)
    autolabel(rects3, ax, 10, True)

    cartella_grafici = "plots/graphITAvsENG/"
    os.makedirs(cartella_grafici, exist_ok=True)  # crea la cartella se non esiste

    # Percorso completo file immagine
    path_grafico = os.path.join(cartella_grafici, nome_file)

    # Salva il grafico
    plt.savefig(path_grafico, dpi=300, bbox_inches='tight')  # dpi=300 per alta qualità


# inserire una funzione che in base al tipo di prompt analizza tutti i dati e faccia una media di tutti i tipi
def plotStatsAboutPrompt(promptType, isEng):
    cartella = "resultsJSON/newFormats"
    namePrompt = " "
    lengTag = " "
    if isEng:
        lengTag = "ENG"
    else:
        lengTag = "ITA"
    namePrompt = f'PromptType-{promptType}_{lengTag}'
    pattern = re.compile(rf'PromptType-{promptType}_{lengTag}')

    file_trovati = []

    for dirpath, dirnames, filenames in os.walk(cartella):
        for filename in filenames:
            if filename.endswith('.json') and pattern.search(filename):
                file_trovati.append(os.path.join(dirpath, filename))
    accuracyVals = []
    recallVals = []
    precisionVals = []
    rejectionRate = []
    fakePositive = []
    fakeNegative = []

    for trovati in file_trovati:
        with open(trovati, "r", encoding="utf-8") as f:
            data = json.load(f)
            metrics = data["metrics"]
            try:
                accuracyVals.append(float(metrics.get("accuracy", np.nan)))
                recallVals.append(float(metrics.get("recall", np.nan)))
                precisionVals.append(float(metrics.get("precision", np.nan)))
                rejectionRate.append(float(metrics.get("rejection_total_rate", np.nan)))
                fakePositive.append(float(metrics.get("fake_positive_rate", np.nan)))
                fakeNegative.append(float(metrics.get("false_negative_rate", np.nan)))
            except (TypeError, ValueError):
                print(f"Attenzione: dati metriche non validi in {trovati}")
    # Calcolo le medie ignorando eventuali NaN
    accuracyMean = np.nanmean(accuracyVals)
    recallMean = np.nanmean(recallVals)
    precisionMean = np.nanmean(precisionVals)
    rejectionRateMean = np.nanmean(rejectionRate)
    fakePositiveMean = np.nanmean(fakePositive)
    fakeNegativeMean = np.nanmean(fakeNegative)

    metriche = ["Accuracy", "Precision", "Recall", "Fake Positive", "Fake Negative", "Rejection Rate"]
    valori = [accuracyMean, precisionMean, recallMean, fakePositiveMean, fakeNegativeMean, rejectionRateMean]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metriche, valori, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Media")
    ax.set_title(f"Metriche medie PromptType-{promptType}_{lengTag}")

    autolabel(bars, ax, fontsize=10, decimal=True)

    cartella_grafici = "plots/singlePromptSTAT/"
    os.makedirs(cartella_grafici, exist_ok=True)  # crea la cartella se non esiste

    nome_file = f"statistics-{namePrompt}.png"
    # Percorso completo file immagine
    path_grafico = os.path.join(cartella_grafici, nome_file)

    # Salva il grafico
    plt.savefig(path_grafico, dpi=300, bbox_inches='tight')  # dpi=300 per alta qualità


def getInfoPromptByModel(dirName, uncertain=False, oneshot=False):
    baseDir = "promptSection/sure/"
    if uncertain:
        baseDir = "promptSection/uncertain/"
    elif oneshot:
        baseDir = "promptSection/oneshot/"
    cartella = f"{baseDir}{dirName}"
    fileList = glob.glob(os.path.join(cartella, "*.json"))
    results = []
    for fileName in fileList:
        with open(fileName, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data["metrics"]
        baseName = os.path.splitext(os.path.basename(fileName))[0]
        parts = baseName.split("_")

        # Prendi il modello come seconda parte (indice 1)
        if len(parts) > 2:
            modello = f"{parts[1]}-{parts[2]}"
        elif len(parts) > 1:
            modello = parts[1]
        else:
            modello = "ModelloSconosciuto"

        results.append({
            "modello": modello,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "false_negative_rate": metrics["false_negative_rate"],
            "false_positive_rate": metrics["false_positive_rate"],
            "rejection_real_rate": metrics["rejection_real_rate"],
            "rejection_fake_rate": metrics["rejection_fake_rate"]
        })

    return results


def createBarPlot(results, dirName, metrics, positive, uncertain, oneshot):
    etichette = [r["modello"] for r in results]
    values_list = [[r[m] for r in results] for m in metrics]

    x = np.arange(len(etichette))
    width = 0.8 / len(metrics)  # spazio per ogni barra

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, values) in enumerate(zip(metrics, values_list)):
        rects = ax.bar(x + (i - len(metrics) / 2) * width, values, width, label=metric)
        autolabel(rects, ax, 10, True)
    if positive:
        flagTitle = "Positive_values"
        cartella_grafici = "plots/modelsBar/positiveStats/"
    else:
        flagTitle = "Negative_values"
        cartella_grafici = "plots/modelsBar/negativeStats/"
    ax.set_ylabel("Valore")
    ax.set_title(f" Prestazioni per modello-{dirName}-{flagTitle}")
    cartella = f"{cartella_grafici}sure/"
    if uncertain:
        ax.set_title(f" Prestazioni per modello-{dirName}-{flagTitle} (uncertain)")
        cartella = f"{cartella_grafici}uncertain/"
    elif oneshot:
        ax.set_title(f" Prestazioni per modello-{dirName}-{flagTitle} (oneshot)")
        cartella = f"{cartella_grafici}oneshot/"
    ax.set_xticks(x)
    ax.set_xticklabels(etichette, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    os.makedirs(cartella, exist_ok=True)
    path_grafico = os.path.join(cartella, f"{dirName}.png")
    plt.savefig(path_grafico, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Fixme Al posto del grafico a barre di FPR e FNR, mostra la matrice di confusione per ogni llm
def plotStatsPromptDividedByModel(dirName, positive=True, uncertain=False, oneshot=False):
    results = getInfoPromptByModel(dirName, uncertain, oneshot)
    if positive:
        createBarPlot(results, dirName, metrics=["accuracy", "precision", "recall"], positive=positive,
                      uncertain=uncertain, oneshot=oneshot)
    else:
        if uncertain:
            createBarPlot(results, dirName,
                          metrics=["false_positive_rate", "false_negative_rate", "rejection_real_rate",
                                   "rejection_fake_rate"], positive=positive,
                          uncertain=uncertain, oneshot=oneshot)
        else:
            createBarPlot(results, dirName,
                          metrics=["false_positive_rate", "false_negative_rate"], positive=positive,
                          uncertain=uncertain, oneshot=oneshot)


# funzione che raccoglie tutte le spiegazioni di un modello in base al tipo al momento solo uncertain, Uncertain real, uncertain fake
def captureOneTypeResponse(path, Type):
    Type_lower = Type.lower()
    risultati = []

    # Determina i file JSON da processare
    if os.path.isfile(path) and path.endswith(".json"):
        fileList = [path]
    elif os.path.isdir(path):
        fileList = glob.glob(os.path.join(path, "*.json"))
    else:
        print(f"Percorso non valido: {path}")
        return risultati

    for file_path in fileList:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Salta i file non leggibili
            continue

        responses = data.get("responses")
        if not isinstance(responses, list):
            continue

        for response in responses:
            prediction = str(response.get("prediction", "")).lower().replace("[", "").replace("]", "").strip()
            ground_truth = str(response.get("ground_truth", "")).lower()

            if Type_lower == "uncertain" and "uncertain" in prediction:
                risultati.append({
                    "image_path": response.get("image_path"),
                    "ground_truth": ground_truth,
                    "explanation": response.get("explanation"),
                    "type": Type_lower
                })

    return risultati


# def saveSummaryToFile(globalSummary, modelName, type, base_folder="plots/infoText"):
#     os.makedirs(base_folder, exist_ok=True)
#     filename = f"plots/infoText/summary-{type}-{modelName}.txt"
#
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(globalSummary)
#     print(f"Riassunto salvato in {filename}")
#
#
#
# # def analyze_and_plot_uncertain(path, modelName, prompt, chunk_size=50):
# #     risultati = captureOneTypeResponse(path, "uncertain")
# #     explanations_all = [r["explanation"] for r in risultati if r.get("explanation")]
# #     explanations_real = [r["explanation"] for r in risultati if r["ground_truth"] == "real" and r.get("explanation")]
# #     explanations_fake = [r["explanation"] for r in risultati if r["ground_truth"] == "fake" and r.get("explanation")]
# #
# #     # Riassunti
# #     summary_general = classifier.summarize_uncertain_patterns_large(explanations_all,
# #                                                                     chunk_size) if explanations_all else ""
# #     saveSummaryToFile(summary_general, modelName, prompt)
# #     summary_real = classifier.summarize_uncertain_patterns_large(explanations_real,
# #                                                                  chunk_size) if explanations_real else ""
# #     saveSummaryToFile(summary_real, modelName, prompt)
# #     summary_fake = classifier.summarize_uncertain_patterns_large(explanations_fake,
# #                                                                  chunk_size) if explanations_fake else ""
# #     saveSummaryToFile(summary_fake, modelName, prompt)
# #
# #     # Conta pattern principali
# #     counter_general = classifier.count_patterns_from_bullets(summary_general)
# #     counter_real = classifier.count_patterns_from_bullets(summary_real)
# #     counter_fake = classifier.count_patterns_from_bullets(summary_fake)
# #
# #     # Assicurati che la cartella esista
# #     output_dir = "plots/uncertainGraph"
# #     os.makedirs(output_dir, exist_ok=True)
# #
# #     # Nome file base (puoi sostituire con nome specifico del dataset o modello)
# #     file_base = f"pattern-incertezza-{modelName}-{prompt}"
# #
# #     # --- Grafico generale ---
# #     if counter_general:
# #         plt.figure(figsize=(8, 5))
# #         plt.bar(counter_general.keys(), counter_general.values(), color="skyblue")
# #         plt.xticks(rotation=45, ha="right")
# #         plt.ylabel("Numero di occorrenze")
# #         plt.title("Pattern di incertezza - Generale")
# #         plt.tight_layout()
# #
# #         general_path = os.path.join(output_dir, f"{file_base}_general.png")
# #         plt.savefig(general_path)
# #         plt.close()  # chiude la figura per liberare memoria
# #
# #     # --- Grafico comparativo real vs fake ---
# #     patterns = list(set(counter_real.keys()) | set(counter_fake.keys()))
# #     real_counts = [counter_real.get(p, 0) for p in patterns]
# #     fake_counts = [counter_fake.get(p, 0) for p in patterns]
# #
# #     x = range(len(patterns))
# #     width = 0.35
# #     fig, ax = plt.subplots(figsize=(10, 5))
# #     ax.bar([i - width / 2 for i in x], real_counts, width, label="Real", color="salmon")
# #     ax.bar([i + width / 2 for i in x], fake_counts, width, label="Fake", color="lightgreen")
# #     ax.set_xticks(x)
# #     ax.set_xticklabels(patterns, rotation=30, ha="right")
# #     ax.set_ylabel("Numero di occorrenze")
# #     ax.set_title("Confronto pattern di incertezza Real vs Fake")
# #     ax.legend()
# #     plt.tight_layout()
# #
# #     comparative_path = os.path.join(output_dir, f"{file_base}_real_vs_fake.png")
# #     plt.savefig(comparative_path)
# #     plt.close()  # chiude la figura
# #
# #     return summary_general, summary_real, summary_fake
#
#
# # Fixme Utilizza google/flan-t5-small su tutto il dataset. Poi puoi clusterizzare gli embedding che ottieni e
# #  calcolare le accuratezze relative a ciascun cluster.
# --- Analisi e clustering con descrizione dei cluster ---
def analyze_and_cluster_uncertain(path, modelName, prompt, n_clusters=5, device="cpu"):
    risultati = captureOneTypeResponse(path, "uncertain")
    explanations = [r["explanation"] for r in risultati if r.get("explanation")]
    labels_gt = [r["ground_truth"] for r in risultati if r.get("explanation")]
    if not explanations:
        print(" Nessuna spiegazione trovata.")
        return

    # 1. Ottieni embedding
    print(" Generazione embedding con flan-t5-small...")
    X = classifier.get_embeddings(explanations, device=device)

    # 2. Clustering
    print(" Clustering con KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # 3. Pipeline per sintesi cluster
    summarizer = pipeline("text2text-generation", model="google/flan-t5-small", device=0 if device == "cuda" else -1)

    # 4. Analisi per cluster
    cluster_stats = {}
    for cluster in range(n_clusters):
        idx = [i for i, c in enumerate(cluster_labels) if c == cluster]
        if not idx:
            continue

        true_labels = [labels_gt[i] for i in idx]
        cluster_explanations = [explanations[i] for i in idx]

        # Conta real/fake
        counts = Counter(true_labels)
        majority_class, majority_count = counts.most_common(1)[0]
        accuracy_cluster = majority_count / len(idx)

        # Riassunto automatico del cluster
        cluster_text = " ".join(cluster_explanations[:20])  # limitiamo a 20 frasi max per prompt
        prompt_text = (
                "Queste sono spiegazioni di incertezze del modello. "
                "Trova i motivi ricorrenti e sintetizzali in una breve frase:\n\n"
                + cluster_text
        )
        description = summarizer(prompt_text, max_new_tokens=60, do_sample=False)[0]['generated_text']

        cluster_stats[cluster] = {
            "num_samples": len(idx),
            "distribution": dict(counts),
            "majority_class": majority_class,
            "cluster_accuracy": accuracy_cluster,
            "description": description.strip()
        }

    # 5. Stampa risultati
    print("\n Risultati per cluster:")
    for cl, stats in cluster_stats.items():
        print(f"Cluster {cl}:")
        print(f"  Campioni: {stats['num_samples']}")
        print(f"  Distribuzione: {stats['distribution']}")
        print(f"  Classe dominante: {stats['majority_class']}")
        print(f"  Accuratezza di cluster: {stats['cluster_accuracy']:.3f}")
        print(f"  Descrizione: {stats['description']}\n")

        # 6. Salvataggio risultati
    output_dir = f"plots/uncertainClusters"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{modelName}-{prompt}_clusters.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "cluster_stats": cluster_stats,
            "cluster_labels": cluster_labels.tolist(),
            "explanations": explanations
        }, f, ensure_ascii=False, indent=4)

    print(f" Risultati salvati in {save_path}")

    return cluster_stats, cluster_labels, explanations
# Todo calcolare la one-class accuracy. In sostanza consiste nel calcolare l'accuracy separatamente per tutte le
#  immagini vere (e false)


# TODO altre funzioni di plotting (dipende da cosa mi serve nella relazione)
if __name__ == "__main__":
    # analyzeSummarizeAndVisualize("resultsJSON/newFormats/qwenVL3b/Uncertain", "uncertain", "qwenVL-3b")
    promptList = ["Prompt-0-Eng", "Prompt-0-Ita", "Prompt-1-Eng", "Prompt-1-Ita", "Prompt-2-Eng", "Prompt-2-Ita",
                  "Prompt-3-Eng", "Prompt-3-Ita", "Prompt-4-Eng", "Prompt-4-Ita", "Prompt-5-Eng", "Prompt-5-Ita",
                  "Prompt-6-Eng", "Prompt-6-Ita"]
    results = captureOneTypeResponse(
        "resultsJSON/newFormats/gemma3/uncertain/real-vs-fake_gemma3_4b_PromptType-1_ENG_20250812-182524_result.json",
        "uncertain")
    print(results)
    for prompt in promptList:
        plotStatsPromptDividedByModel(prompt, True)
    for prompt in promptList:
        plotStatsPromptDividedByModel(prompt, False)
