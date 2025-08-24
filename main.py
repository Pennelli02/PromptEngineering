from tqdm import tqdm

import classifier
import dataset
import metrics
import prompt

# VALORI PER CARICARE IL DATASET
installDataset = False  # serve nel caso si perdesse tutto
startMiniDt = False
NAME = "test_2"
MAX_IMAGES = 100
SHUFFLE = False
# ======================================

# VALORI PER IL PROMPT
INDEX_PROMPT = 0  # (0-6)
IS_ITALIAN = False
SHOW_IMAGES = False
ONESHOT = False
UNCERTAIN_EN = True  # abilitare l'opzione al modello di rispondere incerto
# ===================================
# VALORI PER IL MODELLO

MODEL_NAME = "llava:7b"
# ===================================
# MODALITA' AUTOMATICA
AUTO_ON = True
# dataset section
if startMiniDt:
    images_with_labels, fakes, reals = dataset.loadDataset(MAX_IMAGES)
    dataset.saveDataset(images_with_labels, NAME)
else:
    images_with_labels, fakes, reals = dataset.loadExistingDataset(NAME)

if SHUFFLE:
    dataset.shuffleDataset(images_with_labels)

# prompt section
oneShotMessage = None
if ONESHOT:
    oneShotMessage = prompt.usingOneShot()
if AUTO_ON:
    for INDEX_PROMPT in range(7):  # Prompt da 0 a 6
        for IS_ITALIAN in [False, True]:  # Prima inglese, poi italiano
            userPrompt = prompt.chooseAPrompt(INDEX_PROMPT, IS_ITALIAN)
            systemPrompt = prompt.getSystemPrompt(IS_ITALIAN, UNCERTAIN_EN)

            print(f"\n--- Prompt {INDEX_PROMPT} | Language: {'Italian' if IS_ITALIAN else 'English'} ---")
            print("Prompt:", userPrompt)

            # Inizializza le metriche
            counters = metrics.initMetrics()
            results = []

            # Ciclo principale di analisi immagini
            for img_path, label in tqdm(images_with_labels,
                                        desc=f"Analyzing images (Prompt {INDEX_PROMPT}, {'IT' if IS_ITALIAN else 'EN'})"):
                print(img_path)
                result, counters = classifier.analyze_image(
                    img_path, label, userPrompt, MODEL_NAME,
                    ONESHOT, oneShotMessage, systemPrompt,
                    counters, SHOW_IMAGES
                )
                results.append(result)

            # Salvataggio dei risultati
            jsonINFO = metrics.analyzeMetrics(
                counters, images_with_labels, userPrompt,
                systemPrompt, ONESHOT, oneShotMessage,
                reals, fakes
            )

            metrics.saveAllJson(jsonINFO, results, IS_ITALIAN, MODEL_NAME, INDEX_PROMPT)

else:
    userPrompt = prompt.chooseAPrompt(INDEX_PROMPT, IS_ITALIAN)
    systemPrompt = prompt.getSystemPrompt(IS_ITALIAN, UNCERTAIN_EN)

    print("you choose " + userPrompt)
    # initialize metrics
    counters = metrics.initMetrics()
    results = []
    # Main analysis loop
    for img_path, label in tqdm(images_with_labels, desc=" Analyzing images"):
        print(img_path)
        result, counters = classifier.analyze_image(img_path, label, userPrompt, MODEL_NAME, ONESHOT, oneShotMessage,
                                                    systemPrompt, counters, SHOW_IMAGES)
        results.append(result)
    jsonINFO = metrics.analyzeMetrics(counters, images_with_labels, userPrompt, systemPrompt, ONESHOT, oneShotMessage,
                                      reals, fakes)
    metrics.saveAllJson(jsonINFO, results, IS_ITALIAN, MODEL_NAME, INDEX_PROMPT)