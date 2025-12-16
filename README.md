# PromptEngineering

## Descrizione del Progetto

Progetto di tesi triennale focalizzato sull'analisi e valutazione di modelli Vision-Language (VLM) nella classificazione di volti reali vs. generati artificialmente. Il lavoro esplora diverse strategie di prompt engineering e tecniche di few-shot learning per ottimizzare le prestazioni dei modelli nella discriminazione tra immagini autentiche e sintetiche.

## Obiettivi

- Valutare l'efficacia di diverse tipologie di prompt nella classificazione di volti
- Confrontare le performance di modelli multimodali (LLaVA, Gemma3, Qwen2.5-VL)
- Analizzare l'impatto del prompt engineering su accuratezza e bias
- Studiare le differenze tra prompt in inglese e italiano
- Implementare e testare tecniche di one-shot learning

## Dataset

**140K Real and Fake Faces** (Kaggle)
- 150 immagini di volti reali
- 150 immagini di volti generati artificialmente
- Dataset bilanciato per garantire equità nella valutazione

## Modelli Testati

Tutti i modelli sono disponibili sia in formato **Ollama** che **Hugging Face**:

1. **LLaVA 7B** - Large Language and Vision Assistant
2. **Gemma3 4B** - Modello multimodale di Google
3. **Qwen2.5-VL 3B/7B** - Qwen Vision-Language Models

## Tipologie di Prompt

Il progetto implementa **7 prompt** organizzati in 3 categorie:

- **Neutri** (0, 1, 3): Domande dirette senza bias
- **Orientati "Real"** (4, 5): Bias verso classificazione reale
- **Orientati "Fake"** (2, 6): Focus su artefatti e immagini generate

Ogni prompt è disponibile in **inglese** e **italiano**.

## Struttura del Progetto
```
PromptEngineering/
│
├── classifier.py          # Logica di classificazione
├── dataset.py            # Gestione dataset
├── main.py               # Script principale
├── metrics.py            # Calcolo metriche
├── plot.py               # Visualizzazioni
├── prompt.py             # Definizione prompt
│
├── resultsJSON/          # Risultati esperimenti
├── plots/                # Grafici e visualizzazioni
└── README.md
```


## Utilizzo

### Esecuzione Base
```python
python main.py
```

### Configurazione Parametri
```python
# Dataset
NAME = "test_OS"              # Nome dataset
MAX_IMAGES = 300              # Numero immagini

# Prompt
INDEX_PROMPT = 6              # Tipo prompt (0-6)
IS_ITALIAN = False            # Lingua
UNCERTAIN_EN = True           # Abilita opzione "uncertain"

# Modello
MODEL_NAME = "gemma3:4b"      # Modello da usare

# One-Shot
ONESHOT = True                # Abilita esempio one-shot
```

### Modalità Automatica
```python
AUTO_ON = True  # Esegue tutti gli esperimenti
```

## Metriche Calcolate

- **Accuracy, Precision, Recall**
- **F1-Score e F2-Score**
- **One-class Accuracy** (per classe separata)
- **Rejection Rate** (frequenza risposte incerte)
- **False Positive/Negative Rate**
- **Confusion Matrix**

## Analisi e Visualizzazioni
```python
import plot

# Confronto prompt
plot.plotStatsPrompt("JsonMeanStats/Sure/gemma3")

# Confronto ENG vs ITA
plot.graphLangAvg("llava", metrics=["accuracy", "precision", "recall"])

# Visualizzazione t-SNE
plot.plot_tsne_prediction_with_errors("path/to/results.json", "gemma3")

# Clustering incertezze
plot.visualize_cluster_uncertain("path/to/results.json", "gemma3", "Prompt-0")
```



**Per analisi dettagliate, grafici completi e discussione dei risultati, consultare il report di tesi.**

## Output

Ogni esperimento genera:
- File JSON con metriche e risposte dettagliate
- Grafici comparativi (bar chart, confusion matrix)
- Visualizzazioni embeddings (t-SNE, PCA)
- Analisi cluster per risposte incerte


## Limitazioni

- Dataset limitato a 300 immagini per esperimento
- Focus su volti frontali in condizioni standard
- Vincoli computazionali per esecuzione locale

## Sviluppi Futuri

- Espansione a dataset più ampi
- Test su condizioni di illuminazione variabili
- Integrazione modelli più recenti (GPT-4V, Claude)
- Analisi video e sequenze temporali


## Licenza

Mit


## Riferimenti

- Dataset: [140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- Ollama: [https://ollama.ai](https://ollama.ai)
- Hugging Face: [https://huggingface.co](https://huggingface.co)

---
