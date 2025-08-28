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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Clusterizzazione prompt
CLUSTER_PROMPT = {
    "neutri": ["Prompt-0", "Prompt-1", "Prompt-3"],
    "veri": ["Prompt-4", "Prompt-5"],
    "falsi": ["Prompt-2", "Prompt-6"]
}


def plotStatsPrompt(dirName, grafici="tutti"):
    # Cerca in tutte le sottocartelle prompt-* i file *_mean-result.json
    fileList = glob.glob(os.path.join(dirName, "prompt-*", "*_mean-result.json"))

    risultati = []
    for file_path in fileList:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        baseName = os.path.splitext(os.path.basename(file_path))[0]
        parts = baseName.split("_")

        # Modello (es: real-vs-fake_gemma3_4b_PromptType-0_ENG_mean-result)
        if len(parts) > 2:
            modello = f"{parts[1]}-{parts[2]}"
        elif len(parts) > 1:
            modello = parts[1]
        else:
            modello = "ModelloSconosciuto"

        # Prompt
        prompt = next((p.replace("PromptType-", "Prompt-") for p in parts if p.startswith("PromptType-")),
                      "PromptSconosciuto")

        # Lingua (ultima parte prima di mean-result)
        indice_prompt = next((i for i, p in enumerate(parts) if p.startswith("PromptType-")), None)
        if indice_prompt is not None and indice_prompt + 1 < len(parts):
            lingua = parts[indice_prompt + 1]
        else:
            lingua = "LinguaSconosciuta"

        nome_completo = f"{prompt}-{lingua}"

        risultati.append({
            "nome": nome_completo,
            "prompt": prompt,
            "accuracy": data.get("accuracy_mean", 0),
            "precision": data.get("precision_mean", 0),
            "recall": data.get("recall_mean", 0),
            "f1": data.get("F1_score", 0),
            "f2": data.get("F2_score", 0),
            "one_class_real": data.get("one_class_accuracy_real_mean", 0),
            "one_class_fake": data.get("one_class_accuracy_fake_mean", 0),
        })
    crea_tutti_grafici(risultati, modello, grafici)


def crea_tutti_grafici(risultati, modello, grafici="tutti" ):
    # ===== 1. Grafici divisi per tipo prompt =====
    if grafici in ("tutti", "prompt"):
        for categoria, prompts in CLUSTER_PROMPT.items():
            filtrati = [r for r in risultati if r["prompt"] in prompts]
            genera_grafico(
                filtrati, ["accuracy", "precision", "recall"],
                f"Metriche base - {categoria} ({modello})",
                f"{categoria}-{modello}.png", f"plots/promptBar/{categoria}"
            )
    # ===== 2. Grafici per one_class_accuracy =====
    if grafici in ("tutti", "oneclass"):
        for categoria, prompts in CLUSTER_PROMPT.items():
            filtrati = [r for r in risultati if r["prompt"] in prompts]
            genera_grafico(
                filtrati, ["one_class_real", "one_class_fake"],
                f"One Class Accuracy - {categoria} ({modello})",
                f"one_class-{categoria}-{modello}.png",
                f"plots/oneClass/{categoria}"
            )

        # ===== 3. Grafico singolo F1 e F2 per modello =====
        if grafici in ("tutti", "f1f2"):
            modelli_presenti = set(r["modello"] for r in risultati)
            for modello in modelli_presenti:
                filtrati = [r for r in risultati if r["modello"] == modello]
                genera_grafico(
                    filtrati, ["f1", "f2"],
                    f"F1 e F2 - {modello}",
                    f"F1F2-{modello}.png",
                    "plots/F1F2"
                )


def genera_grafico(risultati_filtrati, metriche, titolo, nome_file, folder):
    if not risultati_filtrati:
        return

    etichette = [r["nome"] for r in risultati_filtrati]
    x = np.arange(len(etichette)) * 2
    width = 0.5

    fig, ax = plt.subplots(figsize=(12, 4))
    rects = []

    for i, m in enumerate(metriche):
        valori = [r[m] * 100 for r in risultati_filtrati]
        rect = ax.bar(x + (i - len(metriche) / 2) * width, valori, width, label=m)
        autolabel(rect, ax, 10, False)
        rects.append(rect)

    ax.set_title(titolo)
    ax.set_ylabel("Valore %")
    ax.set_xticks(x)
    ax.set_xticklabels(etichette, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()

    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, nome_file), dpi=300, bbox_inches='tight')
    plt.close(fig)


def autolabel(rects, ax, fontsize, decimal=True):
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


def graphLangAvg(modelName, metrics=["accuracy", "precision", "recall"], tag="",
                 Uncertain=False, OneShot=False, baseDir="JsonMeanStats"):
    subDir = "Uncertain" if Uncertain else "Sure"
    modelDir = os.path.join(baseDir, subDir, modelName)
    cartella_grafici = "plots/graphITAvsENG/"
    os.makedirs(cartella_grafici, exist_ok=True)

    # cerca file ENG/ITA
    files_eng = glob.glob(os.path.join(modelDir, "prompt-*-Eng", "*_ENG_mean-result.json"))
    files_ita = glob.glob(os.path.join(modelDir, "prompt-*-Ita", "*_ITA_mean-result.json"))

    datiLingua = {"ENG": {m: [] for m in metrics}, "ITA": {m: [] for m in metrics}}
    modello = None

    # ENG
    for fp in files_eng:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        parts = os.path.splitext(os.path.basename(fp))[0].split("_")
        modello = f"{parts[1]}-{parts[2]}"
        for m in metrics:
            key = f"{m}_mean" if f"{m}_mean" in data else m
            if key in data:
                datiLingua["ENG"][m].append(data[key])

    # ITA
    for fp in files_ita:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        parts = os.path.splitext(os.path.basename(fp))[0].split("_")
        modello = f"{parts[1]}-{parts[2]}"
        for m in metrics:
            key = f"{m}_mean" if f"{m}_mean" in data else m
            if key in data:
                datiLingua["ITA"][m].append(data[key])

    # calcola medie
    lingue = ["ENG", "ITA"]
    medie = {m: [np.mean(datiLingua[ling][m]) if datiLingua[ling][m] else 0 for ling in lingue]
             for m in metrics}

    # grafico grouped bar
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, [medie[m][0] for m in metrics], width, label="ENG")
    rects2 = ax.bar(x + width / 2, [medie[m][1] for m in metrics], width, label="ITA")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Valore medio")

    # titolo + nome file
    # normalizza il tag
    tag_str = f"_{tag}" if tag else ""

    # titolo + nome file
    if Uncertain:
        if OneShot:
            titolo = f"Confronto medio ENG vs ITA ({modello}) - Incertezza e OneShot"
            nome_file = f"{modello}-ENGvsITA-Uncertain-Oneshot{tag_str}.png"
        else:
            titolo = f"Confronto medio ENG vs ITA ({modello}) - Incertezza"
            nome_file = f"{modello}-ENGvsITA-Uncertain{tag_str}.png"
    else:
        if OneShot:
            titolo = f"Confronto medio ENG vs ITA ({modello}) - OneShot"
            nome_file = f"{modello}-ENGvsITA-Oneshot{tag_str}.png"
        else:
            titolo = f"Confronto medio ENG vs ITA ({modello})"
            nome_file = f"{modello}-ENGvsITA{tag_str}.png"

    ax.set_title(titolo)
    ax.legend()

    autolabel(rects1, ax, 10)
    autolabel(rects2, ax, 10)

    path_grafico = os.path.join(cartella_grafici, nome_file)
    plt.savefig(path_grafico, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvato: {path_grafico}")


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
            "accuracy": data["accuracy_mean"],
            "precision": data["precision_mean"],
            "recall": data["recall_mean"],
            "rejection_real_rate": data["rejection_real_rate_mean"],
            "rejection_fake_rate": data["rejection_fake_rate_mean"],
            "one_class_accuracy_real": data["one_class_accuracy_real_mean"],
            "one_class_accuracy_fake": data["one_class_accuracy_fake_mean"],
            "TP": data["TP_total"],
            "TN": data["TN_total"],
            "FP": data["FP_total"],
            "FN": data["FN_total"]
        })

    return results


def createBarPlot(results, dirName, metrics, positive, uncertain, oneshot, oneclass=False):
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
        if oneclass:
            cartella_grafici = "plots/compareAccuracy/"
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
    elif oneclass:
        ax.set_title("Prestazioni per modello per one class accuracy")
        cartella = cartella_grafici
    ax.set_xticks(x)
    ax.set_xticklabels(etichette, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    os.makedirs(cartella, exist_ok=True)
    path_grafico = os.path.join(cartella, f"{dirName}.png")
    plt.savefig(path_grafico, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotStatsPromptDividedByModel(dirName, positive=True, uncertain=False, oneshot=False):
    results = getInfoPromptByModel(dirName, uncertain, oneshot)
    if positive:
        createBarPlot(results, dirName, metrics=["accuracy", "precision", "recall"], positive=positive,
                      uncertain=uncertain, oneshot=oneshot)
        createBarPlot(results, dirName, metrics=["one_class_accuracy_real", "one_class_accuracy_fake"],
                      positive=positive, uncertain=uncertain, oneshot=oneshot, oneclass=True)
    else:
        if uncertain:
            createBarPlot(results, dirName,
                          metrics=["rejection_real_rate", "rejection_fake_rate"],
                          positive=positive, uncertain=uncertain, oneshot=oneshot)
            createConfusionMatrix(results, dirName, "uncertain")
        else:
            createConfusionMatrix(results, dirName)


# funzione che raccoglie tutte le spiegazioni di un modello in base al tipo al momento solo uncertain,
# Uncertain real, uncertain fake
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


def createConfusionMatrix(results, dirName, promptType="sure"):
    """
    Crea le matrici di confusione per ogni file/modello e un plot generale con tutte le matrici.
    TP = fake su fake
    TN = real su real
    FP = real predetto come fake
    FN = fake predetto come real
    results: lista di dizionari con TP, TN, FP, FN e modello
    dirName: nome della cartella del prompt
    promptType: 'sure', 'uncertain' o 'oneshot'
    """
    num_results = len(results)
    # Layout subplot (2 colonne, numero di righe necessario)
    ncols = 2
    nrows = (num_results + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()  # per iterare facilmente

    for i, res in enumerate(results):
        TP = res.get("TP", 0)
        TN = res.get("TN", 0)
        FP = res.get("FP", 0)
        FN = res.get("FN", 0)

        # Matrice di confusione secondo la tua definizione
        cm = np.array([[TN, FN],
                       [FP, TP]])

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                    ax=axes[i])
        axes[i].set_title(res.get("modello", f"Modello {i + 1}"))
        axes[i].set_ylabel("Predicted")
        axes[i].set_xlabel("Actual")

    # Nascondi eventuali assi vuoti
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Salvataggio cartella
    save_dir = os.path.join("plots", "confusionMatrix")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{promptType}_{dirName}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Matrice di confusione salvata in: {save_path}")


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


# TODO altre funzioni di plotting (dipende da cosa mi serve nella relazione)
if __name__ == "__main__":
    # analyzeSummarizeAndVisualize("resultsJSON/newFormats/qwenVL3b/Uncertain", "uncertain", "qwenVL-3b")
    # promptList = ["Prompt-0-Eng", "Prompt-0-Ita", "Prompt-1-Eng", "Prompt-1-Ita", "Prompt-2-Eng", "Prompt-2-Ita",
    #               "Prompt-3-Eng", "Prompt-3-Ita", "Prompt-4-Eng", "Prompt-4-Ita", "Prompt-5-Eng", "Prompt-5-Ita",
    #               "Prompt-6-Eng", "Prompt-6-Ita"]
    # # for prompt in promptList:
    # #     plotStatsPromptDividedByModel(prompt, True)
    # for prompt in promptList:
    #     plotStatsPromptDividedByModel(prompt, False, uncertain=True)
    # graphLangAvg("gemma3")
    # graphLangAvg("gemma3", Uncertain=True)
    # graphLangAvg("gemma3", ["one_class_accuracy_real", "one_class_accuracy_fake"], "OCA")
    # graphLangAvg("gemma3", ["rejection_real_rate", "rejection_fake_rate"], tag="NEG", Uncertain=True)
    # graphLangAvg("gemma3", ["one_class_accuracy_real", "one_class_accuracy_fake"], "OCA", Uncertain=True)
    # graphLangAvg("llava")
    # graphLangAvg("llava", Uncertain=True)
    # graphLangAvg("llava", ["one_class_accuracy_real", "one_class_accuracy_fake"], "OCA")
    # graphLangAvg("llava", ["rejection_real_rate", "rejection_fake_rate"], tag="NEG", Uncertain=True)
    # graphLangAvg("llava", ["one_class_accuracy_real", "one_class_accuracy_fake"], "OCA", True)
    # graphLangAvg("qwen3b")
    # graphLangAvg("qwen3b", Uncertain=True)
    # graphLangAvg("qwen3b", ["one_class_accuracy_real", "one_class_accuracy_fake"], "OCA")
    # graphLangAvg("qwen3b", ["rejection_real_rate", "rejection_fake_rate"], tag="NEG", Uncertain=True)
    # graphLangAvg("qwen3b", ["one_class_accuracy_real", "one_class_accuracy_fake"], "OCA", True)
    # graphLangAvg("qwen7b")
    # graphLangAvg("qwen7b", ["one_class_accuracy_real", "one_class_accuracy_fake"], "OCA")
    # graphLangAvg("qwen7b", ["rejection_real_rate", "rejection_fake_rate"], tag="NEG", Uncertain=True)
    # graphLangAvg("qwen7b", ["one_class_accuracy_real", "one_class_accuracy_fake"], "OCA", True)
    plotStatsPrompt("JsonMeanStats/Sure/gemma3")
    plotStatsPrompt("JsonMeanStats/Sure/llava")
    plotStatsPrompt("JsonMeanStats/Sure/qwen3b")
    plotStatsPrompt("JsonMeanStats/Sure/qwen7b")
