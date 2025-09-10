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

from matplotlib.lines import Line2D
from natsort import natsorted
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

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
            "modello": modello
        })
    crea_tutti_grafici(risultati, modello, grafici)


def crea_tutti_grafici(risultati, modello, grafici="tutti"):
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
        autolabel(rect, ax, 10, True)
        rects.append(rect)

    ax.set_title(titolo)
    ax.set_ylabel("Valore %")
    ax.set_xticks(x)
    ax.set_xticklabels(etichette, rotation=45, ha="right")
    ax.set_ylim(0, 115)
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
    ax.set_xticklabels(metrics, rotation=30, ha="right")
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

        # Estrai il tipo di prompt (es. prompt-0-eng o prompt-0-ita)
        tipo_prompt = parts[0] if len(parts) > 0 else "prompt-sconosciuto"
        if len(parts) > 1 and parts[0].startswith("prompt"):
            tipo_prompt += f"-{parts[1]}"  # esempio: prompt-0-eng

        results.append({
            "prompt": tipo_prompt,
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


def plotStatsPromptDividedByModel(dirName, positive=True, uncertain=False, oneshot=False, single=False):
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
            createConfusionMatrix(results, dirName, "uncertain", single)
        else:
            createConfusionMatrix(results, dirName, single=single)


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


def createConfusionMatrix(results, dirName, promptType="sure", single=False):
    """
    Crea le matrici di confusione:
    - Se single=False: un'immagine con tutte le matrici (come ora).
    - Se single=True: un'immagine per ogni modello (cartella plots/confusionMatrix/single/...).
    """
    save_dir = os.path.join("plots", "confusionMatrix")
    if single:
        save_dir = os.path.join(save_dir, "single")
    os.makedirs(save_dir, exist_ok=True)

    if single:
        # Una immagine per ogni modello
        for i, res in enumerate(results):
            TP = res.get("TP", 0)
            TN = res.get("TN", 0)
            FP = res.get("FP", 0)
            FN = res.get("FN", 0)

            cm = np.array([[TN, FN],
                           [FP, TP]])

            plt.figure(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
            plt.title(res.get("modello", f"Modello {i + 1}"))
            plt.ylabel("Predicted")
            plt.xlabel("Actual")

            save_path = os.path.join(save_dir, f"{promptType}_{dirName}_{res.get('modello', i)}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Matrice singola salvata in: {save_path}")
    else:
        # Plot unico con tutte le matrici
        num_results = len(results)
        ncols = 2
        nrows = (num_results + 1) // 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = axes.flatten()

        for i, res in enumerate(results):
            TP = res.get("TP", 0)
            TN = res.get("TN", 0)
            FP = res.get("FP", 0)
            FN = res.get("FN", 0)

            cm = np.array([[TN, FN],
                           [FP, TP]])

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                        ax=axes[i])
            axes[i].set_title(res.get("modello", f"Modello {i + 1}"))
            axes[i].set_ylabel("Predicted")
            axes[i].set_xlabel("Actual")

        # Nascondi assi vuoti
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{promptType}_{dirName}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Matrice multipla salvata in: {save_path}")


def createPromptStatsTable(modello_dir):
    """
    Crea un DataFrame con le statistiche TP, TN, FP, FN per ogni prompt di un modello.
    Ogni riga: tipo prompt | TP | TN | FP | FN
    """
    pattern = os.path.join(modello_dir, "**", "*mean-result.json")
    fileList = glob.glob(pattern, recursive=True)

    rows = []
    for filePath in fileList:
        with open(filePath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Estrai tipo prompt dalla cartella padre
        parts = filePath.replace("\\", "/").split("/")
        prompt_folder = parts[-2]  # es. Prompt-0-Eng
        tipo_prompt = prompt_folder.lower()

        rows.append({
            "tipo_prompt": tipo_prompt,
            "TP": data.get("TP_total", 0),
            "TN": data.get("TN_total", 0),
            "FP": data.get("FP_total", 0),
            "FN": data.get("FN_total", 0)
        })

    df = pd.DataFrame(rows)
    # Ordinamento naturale dei prompt
    df = df.loc[natsorted(df.index)].reset_index(drop=True)
    return df


def savePromptStatsTable(modello_dir, model_name, save_path_csv=None, save_path_img="plots/tableStats/"):
    """
    Crea e salva una tabella con TP, TN, FP, FN per ogni prompt di un modello.
    """
    df = createPromptStatsTable(modello_dir)

    # Salva CSV se richiesto
    if save_path_csv:
        os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)
        df.to_csv(save_path_csv, index=False)
        print(f"Tabella salvata come CSV in: {save_path_csv}")

    # Salva immagine tabella
    if save_path_img:
        os.makedirs(save_path_img, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, len(df) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Titolo della tabella
        ax.set_title(f"Statistica prompt per modello: {model_name}", fontweight='bold', pad=20)

        plt.tight_layout()
        save_file = os.path.join(save_path_img, f"{model_name}.png")
        plt.savefig(save_file, dpi=300)
        plt.close()
        print(f"Tabella salvata come immagine in: {save_file}")


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


def visualize_cluster_uncertain(path, modelName, prompt, n_cluster=5, device="cpu"):
    cluster_stats, cluster_labels, explanations = classifier.analyze_and_cluster_uncertain(path, modelName, prompt,
                                                                                           n_clusters=n_cluster,
                                                                                           device=device)
    # --- 6. Stampa risultati ---
    print("\n Risultati per cluster:")
    for cl, stats in cluster_stats.items():
        print(f"Cluster {cl}:")
        print(f"  Campioni: {stats['num_samples']}")
        print(f"  Distribuzione: {stats['distribution']}")
        print(f"  Classe dominante: {stats['majority_class']}")
        print(f"  Accuratezza di cluster: {stats['cluster_accuracy']:.3f}")
        print(f"  Descrizione: {stats['description']}\n")

    # --- 7. Salvataggio risultati ---
    output_dir = "plots/uncertainClusters"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{modelName}-{prompt}_clusters.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "cluster_stats": cluster_stats,
            "cluster_labels": cluster_labels.tolist(),
            "explanations": explanations
        }, f, ensure_ascii=False, indent=4)

    print(f" Risultati salvati in {save_path}")


def _normalize_pred(s: str) -> str:
    """Mappa la stringa di output del modello in {real,fake,uncertain}."""
    if s is None:
        return "uncertain"
    t = str(s).strip().lower()

    real_set = {"real", "real face", "[real]", "[real face]", "agreed", "no", "[no]"}
    fake_set = {"fake", "generated", "generated face", "[generated]", "yes", "[yes]", "didn't agree"}
    uncertain_set = {"uncertain", "unknown", "not sure", "cannot determine",
                     "n/a", "reject", "rejection", "skip", "[uncertain]"}

    if t in real_set: return "real"
    if t in fake_set: return "fake"
    if t in uncertain_set: return "uncertain"

    # fallback per varianti testuali
    if "real" in t: return "real"
    if "gen" in t or "fake" in t or "ai" in t: return "fake"
    return "uncertain"


def plot_tsne_prediction_with_errors(json_path, model_name,
                                     save_path="plots/tsne/prediction_with_{model}.png",
                                     show=False, perplexity=30, random_state=42,
                                     show_errors=True):
    with open(json_path, "r") as f:
        data = json.load(f)
    responses = data["responses"]

    embs, preds_raw, gts_raw = [], [], []
    for r in responses:
        if r.get("embedding_mean") is not None:
            embs.append(r["embedding_mean"])
            preds_raw.append(r.get("prediction"))
            gts_raw.append(r.get("ground_truth"))

    if not embs:
        print("⚠️ Nessun embedding_mean trovato.")
        return

    # Normalizza classi
    preds = [_normalize_pred(p) for p in preds_raw]
    gts = [("real" if str(gt).lower().startswith("real") else
            "fake" if str(gt).lower().startswith("fake") else "unknown")
           for gt in gts_raw]

    X = np.array(embs, dtype=np.float32)

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    X2 = tsne.fit_transform(X)

    # Colori per classe prevista; marker per correttezza se richiesto
    class_order = ["real", "fake", "uncertain"]
    class_colors = {"real": "#1f77b4", "fake": "#d62728", "uncertain": "#7f7f7f"}  # blu/rosso/grigio
    marker_for = []
    for p, gt in zip(preds, gts):
        if not show_errors:
            marker_for.append("o")  # tutti cerchietti
        else:
            if p == "uncertain" or gt not in {"real", "fake"}:
                marker_for.append("s")  # quadrato per incerto
            else:
                marker_for.append("o" if p == gt else "x")

    fig, ax = plt.subplots(figsize=(10, 7))
    for cls in class_order:
        idxs = [i for i, p in enumerate(preds) if p == cls]
        if not idxs:
            continue
        for i in idxs:
            ax.scatter(X2[i, 0], X2[i, 1],
                       c=[class_colors[cls]],
                       marker=marker_for[i],
                       alpha=0.75,
                       edgecolors="none")

    # Legenda
    if show_errors:
        legend_handles = []

        # Classi con i colori
        for cls in class_order:
            if cls not in preds:
                continue
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w',
                       label=f"Pred: {cls}",
                       markerfacecolor=class_colors[cls], markersize=9)
            )

        # Marker con il significato degli esiti
        legend_handles.extend([
            Line2D([0], [0], marker='o', color='w',
                   label='Correct', markerfacecolor='black', markersize=9),
            Line2D([0], [0], marker='x', color='w',
                   label='Wrong', markeredgecolor='black', markersize=9),
            Line2D([0], [0], marker='s', color='w',
                   label='Uncertain', markerfacecolor='black', markersize=9),
        ])

        ax.legend(handles=legend_handles, title="Legend", loc="best")

    else:
        # Solo classi (senza outcome)
        color_legend = [Line2D([0], [0], marker='o', color='w',
                               label=f"Pred: {cls}",
                               markerfacecolor=class_colors[cls], markersize=9)
                        for cls in class_order if cls in preds]
        ax.legend(handles=color_legend, title="Predicted class", loc="upper left")

    ax.set_title(f"t-SNE visualization ({model_name})", fontsize=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()

    # Riepilogo
    print("Pred counts:", Counter(preds))
    print("GT counts:", Counter(gts))
    if show_errors:
        correct = sum(1 for p, gt in zip(preds, gts) if p in {"real", "fake"} and p == gt)
        wrong = sum(1 for p, gt in zip(preds, gts) if p in {"real", "fake"} and gt in {"real", "fake"} and p != gt)
        uncertain = preds.count("uncertain")
        print(f"Correct: {correct}, Wrong: {wrong}, Uncertain: {uncertain}")

    # Salvataggio
    out_path = save_path.format(model=model_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f" Grafico salvato in {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def analyze_embeddings_from_json(json_path, model_name="model",
                                 tsne_perplexity=30, pca_components=2,
                                 n_clusters=5, show=False, save_dir="analysis_results"):
    """
    Analizza embeddings letti da JSON con t-SNE, PCA e KMeans, salva CSV e produce grafici.
    """
    # Carica JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    responses = data.get("responses", [])
    filtered_responses = [r for r in responses if r.get("embedding_mean") is not None]
    if not filtered_responses:
        print("⚠️ Nessun embedding_mean trovato nel JSON.")
        return

    X = np.array([r["embedding_mean"] for r in filtered_responses], dtype=np.float32)
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # 1️⃣ t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
    X_tsne = tsne.fit_transform(X)

    tsne_df = pd.DataFrame({
        "x": X_tsne[:, 0],
        "y": X_tsne[:, 1],
        "image_path": [r["image_path"] for r in filtered_responses],
        "prediction": [r.get("prediction", "uncertain") for r in filtered_responses],
        "ground_truth": [r.get("ground_truth", "unknown") for r in filtered_responses],
    })
    tsne_df.to_csv(os.path.join(save_dir, f"{model_name}_tsne.csv"), index=False)

    plt.figure(figsize=(10, 7))
    class_colors = {"real": "#1f77b4", "fake": "#d62728", "uncertain": "#7f7f7f"}
    for cls, color in class_colors.items():
        subset = tsne_df[tsne_df["prediction"] == cls]
        if not subset.empty:
            plt.scatter(subset["x"], subset["y"], c=color, label=cls, alpha=0.7)
    if plt.gca().has_data():
        plt.title(f"t-SNE ({model_name})")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{model_name}_tsne.png"), dpi=300, bbox_inches="tight")
        if show: plt.show()
        plt.close()

    # -------------------------
    # 2️⃣ PCA
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    pca_df = pd.DataFrame({f"PC{i + 1}": X_pca[:, i] for i in range(pca_components)})
    pca_df["image_path"] = [r["image_path"] for r in filtered_responses]
    pca_df["prediction"] = [r.get("prediction", "uncertain") for r in filtered_responses]
    pca_df["ground_truth"] = [r.get("ground_truth", "unknown") for r in filtered_responses]
    pca_df.to_csv(os.path.join(save_dir, f"{model_name}_pca.csv"), index=False)

    plt.figure(figsize=(10, 7))
    for cls, color in class_colors.items():
        subset_idx = [i for i, p in enumerate(pca_df["prediction"]) if p == cls]
        if subset_idx:
            plt.scatter(X_pca[subset_idx, 0], X_pca[subset_idx, 1], c=color, label=cls, alpha=0.7)
    if plt.gca().has_data():
        plt.title(f"PCA ({model_name})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{model_name}_pca.png"), dpi=300, bbox_inches="tight")
        if show: plt.show()
        plt.close()

    # -------------------------
    # 3️⃣ KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # -------------------------
    # Creazione frasi descrittive per cluster
    custom_stopwords = {
        "the", "a", "and", "is", "with", "of", "in", "it", "to", "or", "on", "at",
        "for", "by", "from", "as", "this", "that", "an", "be", "are", "was",
        "were", "image", "photo", "photograph", "photographs", "picture", "face", "person", "background",
        "pixel", "real", "fake", "uncertain", "generated", "shows", "show", "but", "he", "she",
    }

    all_explanations = [r.get("explanation") for r in filtered_responses if r.get("explanation")]
    all_words_global = []
    for s in all_explanations:
        if s:
            all_words_global.extend(re.findall(r'\b[a-zA-Z]+\b', s.lower()))
    all_words_global = [w for w in all_words_global if w not in custom_stopwords]
    global_counter = Counter(all_words_global)

    cluster_keywords = {}
    for c in range(n_clusters):
        subset = [r.get("explanation") for i, r in enumerate(filtered_responses)
                  if labels[i] == c and r.get("explanation")]
        all_words = []
        for s in subset:
            if s:
                all_words.extend(re.findall(r'\b[a-zA-Z]+\b', s.lower()))
        meaningful_words = [w for w in all_words if w not in custom_stopwords]

        if meaningful_words:
            counter = Counter(meaningful_words)
            scores = {w: counter[w] / (global_counter.get(w, 1)) for w in counter}
            top_words = sorted(scores, key=scores.get, reverse=True)[:5]
            cluster_keywords[c] = "Questo cluster riguarda " + ", ".join(top_words)
        else:
            cluster_keywords[c] = f"Cluster {c}"

    cluster_df = pd.DataFrame({
        "image_path": [r["image_path"] for r in filtered_responses],
        "prediction": [r.get("prediction") for r in filtered_responses],
        "ground_truth": [r.get("ground_truth") for r in filtered_responses],
        "cluster": labels,
        "cluster_keyword": [cluster_keywords[l] for l in labels]
    })
    cluster_df.to_csv(os.path.join(save_dir, f"{model_name}_clusters.csv"), index=False)

    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")
    for c in range(n_clusters):
        subset_idx = cluster_df[cluster_df["cluster"] == c].index
        if len(subset_idx):
            plt.scatter(X_tsne[subset_idx, 0], X_tsne[subset_idx, 1],
                        c=[cmap(c)] * len(subset_idx), label=cluster_keywords[c], alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='black', marker='x', s=100, label='Centroids')
    if plt.gca().has_data():
        plt.title(f"KMeans clusters ({model_name})")
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{model_name}_clusters.png"), dpi=300, bbox_inches="tight")
        if show: plt.show()
        plt.close()

    # -------------------------
    # 4️⃣ Cosine similarity
    sim_matrix = cosine_similarity(X)

    return {
        "tsne_df": tsne_df,
        "X_tsne": X_tsne,
        "pca_df": pca_df,
        "X_pca": X_pca,
        "cluster_df": cluster_df,
        "kmeans_centroids": kmeans.cluster_centers_,
        "cosine_sim_matrix": sim_matrix
    }


if __name__ == "__main__":
    plot_tsne_prediction_with_errors("resultsJSON/tsne info/real-vs-fake_gemma3_4b_PromptType-6_ENG_20250906-001904_result.json", "gemma3", show_errors=True)
    plot_tsne_prediction_with_errors("resultsJSON/tsne info/real-vs-fake_llava_7b_PromptType-3_ENG_20250907-094857_result.json", "llava", show_errors=True)
    plot_tsne_prediction_with_errors("resultsJSON/tsne info/real-vs-fake_qwen2.5VL_3B_PromptType-3_ENG_20250906-104757_result.json", "qwen3b")
    plot_tsne_prediction_with_errors("resultsJSON/tsne info/real-vs-fake_qwen2.5VL_7B_PromptType-6_ENG_20250906-104817_result.json", "qwen7b")
    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # # =====================
    # # Dizionario con i tuoi dati
    # # =====================
    # data = {
    #     "Veri": {
    #         "4 ENG": {
    #             "LLaVA:7b": [32, 111, 39, 118],
    #             "Gemma3:4b": [31, 114, 36, 119],
    #             "Qwen2.5VL:3b": [86, 62, 88, 64],
    #             "Qwen2.5VL:7b": [0, 150, 0, 150],
    #         },
    #         "4 ITA": {
    #             "LLaVA:7b": [117, 39, 106, 29],
    #             "Gemma3:4b": [44, 112, 38, 106],
    #             "Qwen2.5VL:3b": [5, 143, 7, 145],
    #             "Qwen2.5VL:7b": [0, 150, 0, 150],
    #         },
    #         "5 ENG": {
    #             "LLaVA:7b": [58, 106, 43, 89],
    #             "Gemma3:4b": [0, 150, 0, 150],
    #             "Qwen2.5VL:3b": [0, 147, 3, 150],
    #             "Qwen2.5VL:7b": [0, 150, 0, 150],
    #         },
    #         "5 ITA": {
    #             "LLaVA:7b": [57, 83, 66, 92],
    #             "Gemma3:4b": [0, 150, 0, 150],
    #             "Qwen2.5VL:3b": [0, 149, 1, 150],
    #             "Qwen2.5VL:7b": [0, 150, 0, 150],
    #         },
    #     },
    #     "Falsi": {
    #         "2 ENG": {
    #             "LLaVA:7b": [37, 113, 36, 113],
    #             "Gemma3:4b": [48, 101, 48, 102],
    #             "Qwen2.5VL:3b": [145, 2, 148, 5],
    #             "Qwen2.5VL:7b": [4, 145, 5, 146],
    #         },
    #         "2 ITA": {
    #             "LLaVA:7b": [106, 37, 110, 40],
    #             "Gemma3:4b": [27, 121, 29, 123],
    #             "Qwen2.5VL:3b": [102, 31, 117, 48],
    #             "Qwen2.5VL:7b": [0, 148, 2, 150],
    #         },
    #         "6 ENG": {
    #             "LLaVA:7b": [131, 20, 125, 16],
    #             "Gemma3:4b": [68, 71, 78, 82],
    #             "Qwen2.5VL:3b": [99, 34, 116, 51],
    #             "Qwen2.5VL:7b": [22, 129, 21, 128],
    #         },
    #         "6 ITA": {
    #             "LLaVA:7b": [107, 43, 105, 38],
    #             "Gemma3:4b": [54, 94, 56, 96],
    #             "Qwen2.5VL:3b": [19, 126, 24, 131],
    #             "Qwen2.5VL:7b": [4, 143, 7, 146],
    #         },
    #     },
    #     "Neutri": {
    #         "0 ENG": {
    #             "LLaVA:7b": [108, 29, 119, 39],
    #             "Gemma3:4b": [0, 144, 6, 150],
    #             "Qwen2.5VL:3b": [35, 108, 42, 115],
    #             "Qwen2.5VL:7b": [1, 149, 1, 149],
    #         },
    #         "0 ITA": {
    #             "LLaVA:7b": [79, 55, 93, 67],
    #             "Gemma3:4b": [1, 148, 2, 149],
    #             "Qwen2.5VL:3b": [0, 147, 3, 150],
    #             "Qwen2.5VL:7b": [0, 149, 1, 150],
    #         },
    #         "1 ENG": {
    #             "LLaVA:7b": [109, 42, 108, 41],
    #             "Gemma3:4b": [19, 134, 16, 131],
    #             "Qwen2.5VL:3b": [23, 109, 41, 127],
    #             "Qwen2.5VL:7b": [1, 148, 2, 149],
    #         },
    #         "1 ITA": {
    #             "LLaVA:7b": [86, 54, 96, 64],
    #             "Gemma3:4b": [0, 150, 0, 150],
    #             "Qwen2.5VL:3b": [5, 144, 6, 145],
    #             "Qwen2.5VL:7b": [0, 149, 1, 150],
    #         },
    #         "3 ENG": {
    #             "LLaVA:7b": [136, 13, 135, 14],
    #             "Gemma3:4b": [7, 144, 6, 143],
    #             "Qwen2.5VL:3b": [57, 93, 57, 93],
    #             "Qwen2.5VL:7b": [0, 147, 3, 150],
    #         },
    #         "3 ITA": {
    #             "LLaVA:7b": [98, 47, 102, 50],
    #             "Gemma3:4b": [44, 112, 38, 106],
    #             "Qwen2.5VL:3b": [31, 29, 11, 9],
    #             "Qwen2.5VL:7b": [0, 150, 0, 150],
    #         },
    #     },
    # }
    #
    # # =====================
    # # Creazione DataFrame unificato
    # # =====================
    # rows = []
    # for categoria, prompts in data.items():
    #     for prompt, modelli in prompts.items():
    #         for modello, valori in modelli.items():
    #             TP, TN, FP, FN = valori
    #             rows.append([categoria, prompt, modello, TP, TN, FP, FN])
    #
    # columns = ["Categoria", "Prompt", "Modello", "TP", "TN", "FP", "FN"]
    # df = pd.DataFrame(rows, columns=columns)
    #
    # # Ordinamento per Categoria e Prompt
    # df = df.sort_values(by=["Categoria", "Prompt", "Modello"])
    #
    # # =====================
    # # Creazione della tabella come immagine
    # # =====================
    # fig, ax = plt.subplots(figsize=(16, 10))
    # ax.axis("off")
    #
    # # Tabella pandas → matplotlib
    # table = ax.table(
    #     cellText=df.values,
    #     colLabels=df.columns,
    #     loc="center",
    #     cellLoc="center"
    # )
    #
    # # Stile tabella
    # table.auto_set_font_size(False)
    # table.set_fontsize(8)
    # table.scale(1.2, 1.2)
    #
    # # Salvataggio in PNG
    # plt.savefig("tabella_unificata_test.png", bbox_inches="tight", dpi=300)
    # plt.close()
    #
    # print("Tabella unificata salvata in 'tabella_unificata_test.png'")

    graphLangAvg("llava",
                 metrics=["accuracy", "precision", "recall", "one_class_accuracy_real", "one_class_accuracy_fake"],
                 tag="all")
    graphLangAvg("gemma3",
                 metrics=["accuracy", "precision", "recall", "one_class_accuracy_real", "one_class_accuracy_fake"],
                 tag="all")
    graphLangAvg("qwen3b",
                 metrics=["accuracy", "precision", "recall", "one_class_accuracy_real", "one_class_accuracy_fake"],
                 tag="all")
    graphLangAvg("qwen7b",
                 metrics=["accuracy", "precision", "recall", "one_class_accuracy_real", "one_class_accuracy_fake"],
                 tag="all")
