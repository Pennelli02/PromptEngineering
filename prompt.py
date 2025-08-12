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


def usingOneShot():
    oneShotMessages = [
        {
            "role": "user",
            "content": "Is the provided image showing a real face or a generated one?",
            "images": ["real_vs_fake/real-vs-fake/test/fake/0A266M95TD.jpg"]
        },
        {
            "role": "assistant",
            "content": '''{ "result": "generated", "reason": "The image shows distinctive characteristics of 
            artificially generated faces, typical of GAN models (such as StyleGAN). First, there are slight 
            asymmetries in the glasses: the right and left parts are not perfectly aligned, a common detail in 
            synthetic faces. Second, the background appears blurred and lacks realistic depth, 
            with no distinguishable elements, showing an unnatural blending of hair and head contours. Additionally, 
            the lighting on the face is too uniform and lacks consistent shadows; for example, there are no shadows 
            cast by the glasses, as would be expected in a real photo. The teeth are overly regular and symmetrical, 
            and the eyes appear perfectly centered and devoid of complex reflections or imperfections, 
            which are normally found in real human faces. All these subtle signals combined strongly indicate that 
            this is an artificially generated face." }'''
        }
    ]
    return oneShotMessages

# TODO gestione multipla ONESHOT

# TODO vogliamo provare un nuovo prompt? (simulare un game)
