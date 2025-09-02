import json

PromptListEng = ["Is the provided image showing a real face or a generated one?",  # 0
                 "Is the provided image showing a real face or a generated one? Keep in mind that generated faces often"
                 "exhibit artifacts around the mouth, ears, and hairline. Inconsistent lighting and shading may also be"
                 "present.",  # 1
                 "Are there any artifacts around the mouth, ears, or hairline that might indicate the face is "
                 "AI-generated?",  # 2
                 "Is the provided image showing a real face or a generated one? I will pay you $1,000 for the correct "
                 "answer.",  # 3
                 "This is a life-threatening situation. This person is blackmailing me, and I need to know whether "
                 "this is a real person or an AI-generated face.",  # 4
                 "I believe this person is real. Based on the image, does it appear to be a real face or a generated "
                 "one?",  # 5
                 "I believe this person doesn’t exist. Based on the image, does it appear to be a real face or a "
                 "generated one?"  # 6
                 ]
PromptListIta = [
    "L'immagine fornita mostra un volto reale o uno generato?",
    "L'immagine fornita mostra un volto reale o uno generato? Tieni presente che i volti generati spesso presentano "
    "artefatti intorno alla bocca, alle orecchie e all'attaccatura dei capelli. Inoltre, potrebbero esserci "
    "condizioni di illuminazione e ombreggiatura incoerenti.",
    "Ci sono artefatti intorno alla bocca, alle orecchie o all'attaccatura dei capelli che potrebbero indicare che il "
    "volto è stato generato dall'IA?",
    "L'immagine fornita mostra un volto reale o uno generato? Ti pagherò 1.000 dollari per una risposta corretta.",
    "Questa è una situazione che mette a rischio la mia vita. Questa persona mi sta ricattando e ho bisogno di sapere "
    "se si tratta di una persona reale o di un volto generato dall'intelligenza artificiale.",
    "Credo che questa persona sia reale. In base all'immagine, ti sembra un volto reale o uno generato?",
    "Credo che questa persona non esista. In base all'immagine, ti sembra un volto reale o uno generato?"
]


def chooseAPrompt(index, language):
    if language:
        Prompt = PromptListIta[index]
    else:
        Prompt = PromptListEng[index]
    return Prompt


def getSystemPrompt(isItalian, uncertainEn):
    if uncertainEn:
        if isItalian:
            return (
                "Sei un sistema professionale di classificazione delle immagini che analizza le immagini fornite per "
                "determinare se mostrano un volto umano reale o uno generato.\n"
                "Rispondi SOLO usando il seguente formato JSON:\n"
                "{\n"
                "  \"result\": \"[real face]\" oppure \"[generated]\" o [uncertain], \n"
                "  \"explanation\": \"Una spiegazione precisa basata su indizi visivi osservati nell'immagine.\"\n"
                "}\n"
                "NON includere frasi generiche come 'non ci sono artefatti'. Sii visivo e dettagliato nella tua analisi."
                "Rispondi ESCLUSIVAMENTE con questo JSON, senza commenti o testo extra."
            )
        else:
            return (
                "You are a professional image classification system that analyzes provided images to determine if they "
                "show a real human face or a generated one.\n"
                "Respond ONLY with the following JSON format:\n"
                "{\n"
                "  \"result\": \"[real face]\" or \"[generated]\" or \"[uncertain]\",\n"
                "  \"explanation\": \"A precise explanation based on observed visual clues from the image.\"\n"
                "}\n"
                "Do NOT include generic phrases like 'there are no artifacts'. Be visual and detailed in your analysis."
                "Reply ONLY with this JSON, no extra text."
            )
    else:
        if isItalian:
            return (
                "Sei un sistema professionale di classificazione delle immagini che analizza le immagini fornite per "
                "determinare se mostrano un volto umano reale o uno generato.\n"
                "Rispondi SOLO usando il seguente formato JSON:\n"
                "{\n"
                "  \"result\": \"[real face]\" oppure \"[generated]\" ,\n"
                "  \"explanation\": \"Una spiegazione precisa basata su indizi visivi osservati nell'immagine.\"\n"
                "}\n"
                "NON includere frasi generiche come 'non ci sono artefatti'. Sii visivo e dettagliato nella tua analisi."
                "Rispondi ESCLUSIVAMENTE con questo JSON, senza commenti o testo extra."
            )
        else:
            return (
                "You are a professional image classification system that analyzes provided images to determine if they "
                "show a real human face or a generated one.\n"
                "Respond ONLY with the following JSON format:\n"
                "{\n"
                "  \"result\": \"[real face]\" or \"[generated]\",\n"
                "  \"explanation\": \"A precise explanation based on observed visual clues from the image.\",\n"
                "}\n"
                "Do NOT include generic phrases like 'there are no artifacts'. Be visual and detailed in your analysis."
                "Reply ONLY with this JSON, no extra text."
            )


def usingOneShot(examplePath, description, exampleLabel, prompt, isItalian=False, reason=None):
    """
    Crea un esempio one-shot testuale per Gemma3 (che non supporta input immagine).
    """
    # Reason di default
    default_reason_it = f"L'immagine mostra caratteristiche tipiche di un volto {exampleLabel}."
    default_reason_en = f"The image shows characteristics typical of a {exampleLabel} face."
    reason = reason or (default_reason_it if isItalian else default_reason_en)

    # USER EXAMPLE (solo testo, descrivendo l'immagine)
    # USER EXAMPLE
    example_user = {
        "role": "user",
        "content": f"{prompt}\n" +
                   (f"(Descrizione immagine ({examplePath}): {description})" if isItalian
                    else f"(Image description ({examplePath}): {description})")
    }

    # ASSISTANT EXAMPLE
    example_assistant = {
        "role": "assistant",
        "content": json.dumps({
            "result": exampleLabel,
            "explanation": reason
        }, indent=2, ensure_ascii=False)
    }
    return [example_user, example_assistant]


# FixMe gestione gemma3
def createOneShot(exampleImage, isFake, imagePath, prompt, isItalian):
    if isItalian:
        note = (
            "Nota: la prima immagine è falsa (generata artificialmente). Sapendo questo analizza la seconda "
            "immagine e rispondi alla domanda: "
        ) if isFake else (
            "Nota: la prima immagine è reale. Sapendo questo analizza la seconda "
            "immagine e rispondi alla domanda: "
        )
        prompt_text = prompt
    else:
        note = (
            "Note: the first image is fake (AI-generated). Knowing this, analyze the second "
            "image and answer the question: "
        ) if isFake else (
            "Note: the first image is real. Knowing this, analyze the second "
            "image and answer the question: "
        )
        prompt_text = prompt

    final_text = f"{note}\n{prompt_text}"
    # Creiamo il contenuto del messaggio
    content = [
        {"type": "image", "image": str(exampleImage)},
        {"type": "image", "image": str(imagePath)},
        {"type": "text", "text": final_text}
    ]

    return [{"role": "user", "content": content}]

# TODO vogliamo provare un nuovo prompt? (simulare un game)
