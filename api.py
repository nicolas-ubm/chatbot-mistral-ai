from fastapi import FastAPI
from transformers import pipeline
from datetime import datetime

app = FastAPI()

# Initialize the pipeline for text generation
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")

@app.get("/infos")
def get_general_information(prompt: str = "Que puis-je faire pour vous ?"):
    t0 = datetime.now()

    # Define the instruction for the "Informations générales" agent
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant spécialisé en fourniture d'informations générales sur l'Université Bordeaux Montaigne. "
                "Réponds aux questions concernant les étudiants, enseignants, programmes, infrastructures et tout autre aspect général. "
                "Réponds toujours de manière concise et factuelle."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = pipe(messages, max_length=200, num_return_sequences=1)[0]["generated_text"]
    except Exception as e:
        return {"error": str(e)}

    t1 = datetime.now()
    execution_time = (t1 - t0).total_seconds()

    return {
        "response": response,
        "execution_time": execution_time
    }

@app.get("/salles")
def get_room_information(prompt: str = "Quelles salles sont disponibles ?"):
    t0 = datetime.now()

    # Define the instruction for the "Salles" agent
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant spécialisé dans la gestion et la réservation des salles de l'Université Bordeaux Montaigne. "
                "Fournis des informations sur les capacités, équipements et disponibilités des salles. "
                "Réponds toujours de manière claire et structurée."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = pipe(messages, max_length=200, num_return_sequences=1)[0]["generated_text"]
    except Exception as e:
        return {"error": str(e)}

    t1 = datetime.now()
    execution_time = (t1 - t0).total_seconds()

    return {
        "response": response,
        "execution_time": execution_time
    }

@app.get("/juridique")
def get_legal_information(prompt: str = "Quels sont les aspects juridiques à connaître ?"):
    t0 = datetime.now()

    # Define the instruction for the "Juridique" agent
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant juridique spécialisé pour l'Université Bordeaux Montaigne. "
                "Réponds aux questions sur les aspects juridiques, contrats, réglementations et autres informations pertinentes. "
                "Réponds toujours avec précision et dans le cadre des réglementations actuelles."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = pipe(messages, max_length=200, num_return_sequences=1)[0]["generated_text"]
    except Exception as e:
        return {"error": str(e)}

    t1 = datetime.now()
    execution_time = (t1 - t0).total_seconds()

    return {
        "response": response,
        "execution_time": execution_time
    }

@app.get("/autres")
def handle_other_requests(prompt: str = "Puis-je vous aider avec autre chose ?"):
    t0 = datetime.now()

    # Define the instruction for the "Autres" agent
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant conçu pour traiter toutes les demandes ne relevant pas des autres agents. "
                "Si la demande ne peut pas être satisfaite, réponds : \"Non, impossible.\" "
                "Sinon, essaie de guider l'utilisateur vers une ressource appropriée ou donne une réponse générique utile."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = pipe(messages, max_length=200, num_return_sequences=1)[0]["generated_text"]
    except Exception as e:
        return {"error": str(e)}

    t1 = datetime.now()
    execution_time = (t1 - t0).total_seconds()

    return {
        "response": response,
        "execution_time": execution_time
    }


@app.get('/sentiments')
def sentiment(prompt: str = "Que puis-je faire pour vous ?"):
    t0 = datetime.now()

    # Prepare the input messages for the model
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant spécialisé en analyse de sentiment."
                " Pour chaque texte donné, détermine si le sentiment est POSITIF, NÉGATIF ou NEUTRE."
                " Réponds toujours sous forme de dictionnaire JSON avec les clés suivantes :"
                " {"
                "   'sentiment': '<POSITIVE|NEGATIVE|NEUTRAL>',"
                "   'confidence': <float>,"
                "   'reason': 'explication du choix'"
                " }"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Call the model and get the response
    response = pipe(messages, max_length=200, num_return_sequences=1)[0]['generated_text']

    t1 = datetime.now()
    execution_time = (t1 - t0).total_seconds()

    # Return the response and execution time
    return {
        "response": response,
        "execution_time": execution_time
    }


@app.get('/')
def bonjour(user_message: str = "Que puis-je faire pour vous ?"):
    t0 = datetime.now()

    # Prepare a greeting response
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un chatbot d'accueil."
                " Ton rôle est d'interagir avec l'utilisateur pour comprendre sa demande,"
                " qualifier son intention en utilisant les points d'entrée '/demande' et '/sentiment',"
                " et récupérer les informations pertinentes pour contextualiser sa demande."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Call the model and get the response
    response = pipe(messages, max_length=200, num_return_sequences=1)[0]['generated_text']

    t1 = datetime.now()
    execution_time = (t1 - t0).total_seconds()

    # Return the response and execution time
    return {
        "response": response,
        "execution_time": execution_time
    }

import datetime

