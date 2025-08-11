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
import classifier
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

nltk.download('stopwords')


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


# FIXME vautare se tenere in considerazione per tipo di prompt
# funzione che raccoglie tutte le spiegazioni di un modello in base al tipo al momento solo uncertain, fp, fn
def captureOneTypeResponse(dirName, Type):
    cartella = dirName
    fileList = glob.glob(os.path.join(cartella, "*.json"))
    Type_lower = Type.lower()
    risultati = []

    for file_path in fileList:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Errore nel leggere {file_path}")
                continue

        responses = data.get("responses", [])
        if not isinstance(responses, list):
            continue

        for response in responses:
            prediction = str(response.get("prediction", "")).lower().replace("[", "").replace("]", "").strip()
            ground_truth = str(response.get("ground_truth", "")).lower()

            if Type_lower == "uncertain":
                if "uncertain" in prediction:
                    risultati.append({
                        "image_path": response.get("image_path"),
                        "ground_truth": ground_truth,
                        "explanation": response.get("explanation"),
                        "type": Type_lower
                    })
                print(response.get("explanation"))

            elif Type_lower == "fp":
                if (prediction in ("generated", "yes")) and ground_truth == "real":
                    risultati.append({
                        "image_path": response.get("image_path"),
                        "ground_truth": ground_truth,
                        "explanation": response.get("explanation"),
                        "type": Type_lower
                    })

            elif Type_lower == "fn":
                if (prediction in ("real face", "no")) and ground_truth == "fake":
                    risultati.append({
                        "image_path": response.get("image_path"),
                        "ground_truth": ground_truth,
                        "explanation": response.get("explanation"),
                        "type": Type_lower
                    })

    return risultati


def saveSummaryToFile(globalSummary, modelName, type, base_folder="plots/infoText"):
    os.makedirs(base_folder, exist_ok=True)
    filename = f"plots/infoText/summary-{type}-{modelName}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(globalSummary)
    print(f"Riassunto salvato in {filename}")


def plotWordcloudFromExplanations(explanations, modelName, Type):
    text = " ".join(explanations).lower()

    # Unisci stopwords inglesi e italiane
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('italian')))

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stop_words,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"WordCloud delle spiegazioni-{Type} ({modelName})")
    cartella_grafici = "plots/wordCloud/"
    nome_file = f"worldCloud-{modelName}-{Type}.png"
    os.makedirs(cartella_grafici, exist_ok=True)  # crea la cartella se non esiste
    # Percorso completo file immagine
    path_grafico = os.path.join(cartella_grafici, nome_file)

    # Salva il grafico
    plt.savefig(path_grafico, dpi=300, bbox_inches='tight')


def analyzeSummarizeAndVisualize(dirName, Type, modelName):
    results = captureOneTypeResponse(dirName, Type)
    df = classifier.repair_dates(results)
    explanations = df['explanation'].tolist()

    # Riassunto globale
    #global_summary = classifier.summarizeALL(explanations)
    global_summary = classifier.hierarchical_summarize_with_prompt(explanations)
    print("\nRiassunto globale finale:")
    print(global_summary)

    # Salva su file
    saveSummaryToFile(global_summary, modelName, Type)

    # WordCloud
    plotWordcloudFromExplanations(explanations, modelName, Type)


if __name__ == "__main__":
    analyzeSummarizeAndVisualize("resultsJSON/newFormats/qwenVL3b/Uncertain", "uncertain", "qwenVL-3b")
