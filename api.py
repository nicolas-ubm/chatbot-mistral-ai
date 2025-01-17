from fastapi import FastAPI
from transformers import pipeline
from datetime import datetime
import json

app = FastAPI()

# Initialize the pipeline for text generation
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")

# Load agent information from agents.json
with open("agents.json", "r", encoding="utf-8") as file:
    agents_data = json.load(file)

def get_agent_info(agent_id):
    agent = agents_data.get(agent_id, {})
    docs_content = "\n\n".join([
        f"{doc['title']}\n{doc['content']}"
        for doc in agent.get("agent_docs", {}).values()
    ])
    return docs_content


@app.get("/")
def root():
    return {"message": "L'API Back est opérationnelle."}

@app.get("/infos")
def get_general_information(prompt: str = "Que puis-je faire pour vous ?"):
    t0 = datetime.now()
    print(prompt)
    # Get agent documentation
    agent_id = "id_agent0"  # ID for "Informations générales"
    agent_docs = get_agent_info(agent_id)

    # Define the instruction for the "Informations générales" agent
    messages = [
        {
            "role": "system",
            "content":
                "Tu es un assistant spécialisé en fourniture d'informations générales sur l'Université Bordeaux Montaigne. "
                "Réponds aux questions concernant les étudiants, enseignants, programmes, infrastructures et tout autre aspect général. "
                "Réponds toujours de manière concise et factuelle."
                "Voici les informations pour t'aider à répondre :"
                f"{agent_docs}"
            
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = pipe(messages, max_new_tokens=2050, max_length=500)[0]["generated_text"][2]["content"]
        print(response)
    
    except Exception as e:
        print(e)
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
    print(prompt)
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
        response = pipe(messages, max_new_tokens=2050, max_length=500)[0]["generated_text"][2]["content"]
        print(response)

    except Exception as e:
        print(e)
        return {"error": str(e)}

    t1 = datetime.now()
    execution_time = (t1 - t0).total_seconds()

    return {
        "response": response,
        "execution_time": execution_time
    }



@app.get("/finances")
def get_legal_information(prompt: str = "Quels sont les aspects financiers à connaître ?"):
    t0 = datetime.now()
    print(prompt)
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
        response = pipe(messages, max_new_tokens=2050, max_length=500)[0]["generated_text"][2]["content"]
        print(response)

    except Exception as e:
        print(e)
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
    print(prompt)
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
        response = pipe(messages, max_new_tokens=2050, max_length=500)[0]["generated_text"][2]["content"]
        print(response)

    except Exception as e:
        print(e)
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
    print(prompt)
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
    response = pipe(messages, max_new_tokens=2050, max_length=500)[0]["generated_text"][2]["content"]
    print(response)

    t1 = datetime.now()
    execution_time = (t1 - t0).total_seconds()

    # Return the response and execution time
    return {
        "response": response,
        "execution_time": execution_time
    }


@app.get('/chat')
def bonjour(prompt: str = "Que puis-je faire pour vous ?"):
    t0 = datetime.now()
    print(prompt)
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
            "content": prompt
        }
    ]

    # Call the model and get the response
    #response = pipe(messages, max_length=200, num_return_sequences=1)[0]['generated_text']
    response = pipe(messages, max_new_tokens=2050, max_length=500)[0]["generated_text"][2]["content"]
    print(response)
    t1 = datetime.now()
    execution_time = (t1 - t0).total_seconds()

    # Return the response and execution time
    return {
        "response": response,
        "execution_time": execution_time
    }
