from tqdm import tqdm

import classifier
import dataset
import metrics
import prompt

# VALORI PER CARICARE IL DATASET
installDataset = False  # serve nel caso si perdesse tutto
startMiniDt = False
NAME = "test_1"
MAX_IMAGES = 100
SHUFFLE = False
# ======================================

# VALORI PER IL PROMPT
INDEX_PROMPT = 4  # (0-6)
IS_ITALIAN = False
SHOW_IMAGES = False
ONESHOT = False
UNCERTAIN_EN = False  # abilitare l'opzione al modello di rispondere incerto
# ===================================
# VALORI PER IL MODELLO

MODEL_NAME = "gemma3:4b"
# ===================================
# dataset section
if installDataset:
    dataset.installDataset()

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
