from tqdm import tqdm

import classifier
import dataset
import metrics
import prompt

# VALORI PER CARICARE IL DATASET
installDataset = False  # serve nel caso si perdesse tutto
startMiniDt = False
NAME = "test_OS"
MAX_IMAGES = 300
SHUFFLE = False
# ======================================

# VALORI PER IL PROMPT
INDEX_PROMPT = 6  # (0-6)
IS_ITALIAN = False
SHOW_IMAGES = False
ONESHOT = True
UNCERTAIN_EN = False  # abilitare l'opzione al modello di rispondere incerto
# ===================================
# VALORI PER IL MODELLO

MODEL_NAME = "gemma3:4b"
# ===================================
# MODALITA' AUTOMATICA
AUTO_ON = False

# =============================
# GEMMA3 ONESHOT
imagePath = "photoEx/real/woman3-4.jpg"
description = ("This is a close-up, head-and-shoulders portrait of a young woman. She has shoulder-length, dark brown "
               "hair that is parted on her left side, with some strands falling forward around her face. Her face is "
               "a smooth, light tone. She has large, brown eyes that are looking directly at the camera, "
               "a small nose, and full, light-toned lips. Her expression is calm and neutral. The lighting is soft, "
               "evenly illuminating her face and hair. The background is blurred, but you can make out the vague "
               "shapes and colors of what appear to be party balloons, including a reddish one on the upper left and "
               "a light-colored one on the upper right with a faint, darker pattern on it.")
labelImage = "real face"
reason = ("The image shows a person with natural skin texture, including visible pores and slight unevenness in tone, "
          "particularly around the nose. The eyes have realistic reflections and subtle imperfections. The hair has a "
          "natural variation in strand thickness and highlights, and there are realistic, soft focus gradients in the "
          "background, consistent with a photograph taken with a camera.")
# ho usato la spiegazione fornita da gemini (stessa casa di produzione di gemma3)
# ======================================
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

if AUTO_ON:
    # DATASETS = ["test_2", "test_3"]
    # for data in DATASETS:
    #     print(f"\n=== Running dataset: {data} ===")
    #
    #     images_with_labels, fakes, reals = dataset.loadExistingDataset(data)
    #     if SHUFFLE:
    #         dataset.shuffleDataset(images_with_labels)
    #     if data == "test_2":
    #         startPoint = 2
    #     else:
    #         startPoint = 0
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
    if ONESHOT:
        oneShotMessage = prompt.usingOneShot(imagePath, description, labelImage, userPrompt, IS_ITALIAN, reason)

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
