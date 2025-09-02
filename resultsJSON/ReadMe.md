# Info generali
## Modelli usati
- **llava:7b**
- **llava-detector** (modello di llava:7b in cui è presente un system prompt)
- **gemma3:1b** (modello fallimentare non prende le immagini come input)
- **gemma31b-detector** (modello fallimentare non prende le immagini come input)
- **gemma3:4b**
- **gemma3-detector** (modello di gemma3:4b in cui è presente un system prompt)
- **qwen2-5-VL-3b_HF** e **qwenVL3b** (modello preso da hugging face con system prompt usiamo il modello qwen2.5VL:3b)
- ## Metriche usate
- **Accuracy**: porzione di previsioni corrette sul totale delle previsioni
- **Precision**: misura la proporzione di previsioni positive corrette rispetto a tutte le previsioni positive fatte dal modello (vera su vera le prime 6 poi sarà falsa su falsa)
- **Recall**: indica la proporzione di istanze positive reali che il modello è riuscito a identificare correttamente. 
## Specifiche JSON
- Il titolo del file json è presente il nome del modello usato, il tipo di prompt usato, la lingua usata e quando è stato fatto il tentativo
- Nel file JSON sono presenti valori oltre all'accuracy, recall e precision, anche valori come rejection rate sia per le immagini vere che false e anche il parsing error per vedere quanta fantasia ha il modello nel formulare le risposte
- **TP**: equivale a quante volte riconosce che un'immagine è falsa
- **TN**: equivale a quante volte riconosce che un'immagine è vera
- **FN**: equivale a quante volte confonde una falsa per vera
- **FP**: equivale a quante volte confonde una vera per falsa
## Informazioni aggiunte
Sono presenti file json di gemma3 ottenuti facendo girare ollama su un google colab (**OLLAMA_gemma3_4b**)
## Domanda da tener in considerazione:
dovrò mettere anche la risposta del modello per ogni immagine? Magari finiamo tutti i test e poi lo aggiungo

-----------------------------------------------------------------------------------------------------------------------
## Prompt per modello per one-shot
- **LlaVa**: prompt-4-ita
- **Gemma3**: prompt-6-eng
- **Qwen3b**: prompt-4-eng (scelto non per le prestazioni con i fake ma per un equilbrio con i real)
- **Qwen7b**: prompt-6-eng
-------------------------------------------------------------------------------------------
## Set di immagini scelto per fare one-shot
- tre vere
- tre false 
sono presenti nella cartella photoEx