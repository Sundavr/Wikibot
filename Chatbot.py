# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:03:40 2020

@author: Johan
"""
import os 
import pickle
import schedule
import time
import wikipedia

from os import path
from typing import Dict
from flask import Flask, request
from dialogflow_fulfillment import WebhookClient

dir_name = os.path.dirname(os.path.realpath(__file__))
if (not path.exists(dir_name + "/sentiment-analysis")):
    from transformers import pipeline
    classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
    pickle.dump(classifier, open(dir_name + "/sentiment-analysis", "wb"))
else:
    classifier = pickle.load(open(dir_name + "/sentiment-analysis", "rb"))

# "session id": ["lang", "last_update"]
sessions_lang = dict()

wikipedia.set_lang("fr")
app = Flask(__name__)

# TESTS
def run_tests():
    print(classifier('We are very happy to show you the 🤗 Transformers library.')) # POSITIVE
    print(classifier("Alad'2 est clairement le meilleur film de l'année 2018.")) # POSITIVE
    print(classifier("Juste whoaaahouuu !")) # POSITIVE
    print(classifier("NUL...A...CHIER ! FIN DE TRANSMISSION.")) # NEGATIVE
    print(classifier("Je m'attendais à mieux de la part de Franck Dubosc !")) # NEGATIVE
    print(classifier("ça pu la merde !!")) # NEGATIVE

run_tests()

# ---------- INTENTS START ---------- #
def get_summary(queryResult: dict, agent: WebhookClient, session: str) -> WebhookClient:
    information = queryResult["parameters"]["information"]
    print("summary :", information)
    try:
        if (wikipedia.API_URL.split(".")[0].split("/")[-1] != sessions_lang[session]):
            wikipedia.set_lang(sessions_lang[session][0])
        response = wikipedia.summary(information).replace(" ()",'').replace("()", '')
        if len(response) > 4500:
            response = response[:4500] + " [...]"
        agent.add(response) 
    except Exception:
        agent.add("Désolé, mais je suis incapable de te dire ce qu'est '" + information + "'")
    return agent

def how_are_you(queryResult: dict, agent: WebhookClient, session: str) -> WebhookClient:
    agent.add(str(classifier(queryResult.get("queryText"))))
    return agent

def quick_summary(queryResult: dict, agent: WebhookClient, session: str) -> WebhookClient:
    information = queryResult["parameters"]["information"]
    print("quick summary :", information)
    try:
        if (wikipedia.API_URL.split(".")[0].split("/")[-1] != sessions_lang[session]):
            wikipedia.set_lang(sessions_lang[session][0])
        response = wikipedia.summary(information, sentences=1).replace(" ()",'').replace("()", '')
        if len(response) > 4500:
            response = response[:4500] + " [...]"
        agent.add(response)
    except Exception:
        agent.add("Désolé, mais je suis incapable de te dire ce qu'est '" + information + "'")
    return agent

def set_language(queryResult: dict, agent: WebhookClient, session: str) -> WebhookClient:
    language = queryResult["parameters"]["language"].lower()
    print("set_language", language)
    language_changed = False
    if ((language == "fr") or (language == "français") or (language == "french")):
        sessions_lang[session][0] = "fr"
        language_changed = True
    elif ((language == "en") or (language == "anglais") or (language == "english")):
        sessions_lang[session][0] = "en"
        language_changed = True
    if (language_changed):
        agent.add("Et hop ! Tes renseignements seront maintenant en " + language)
    else:
        agent.add("Désolé mais je ne suis pas en mesure de te trouver des renseignements dans cette langue :(.")
    return agent

def which_language(queryResult: dict, agent: WebhookClient, session: str) -> WebhookClient:
    agent.add("Je peux récupérer des informations en français ou en anglais si tu veux !")
    return agent

# ---------- INTENTS END ---------- #

def intent_switch(intent_name: str, queryResult: dict, agent: WebhookClient, session: str) -> str:
    switcher = {
        "get_summary": get_summary,
        "how_are_you": how_are_you,
        "quick_summary": quick_summary,
        "set_language": set_language,
        "which_language": which_language,
    }
    func = switcher.get(intent_name)
    if (func != None):
        return func(queryResult, agent, session)
    print("Unknown intent : " + intent_name)
    agent.add("Toute mes excuses, mais mon serveur à eu un léger soucis à comprendre ton intention :(")
    return agent

def get_response(agent: WebhookClient) -> dict:
    agent_response = agent.response
    message = agent_response["fulfillmentMessages"]
    del agent_response["fulfillmentMessages"]
    response = dict()
    for rep in message:
        if (rep.get("text") != None):
            response["fulfillmentText"] = rep.get("text").get("text")[0]
    response.update(agent_response)
    return response

@app.route('/fulfillment', methods=['POST'])
def webhook() -> Dict:
    request_ = request.get_json(force=True)
    session = request_['session'].split("/")[-1]
    if (not session in sessions_lang):
        sessions_lang[session] = ["fr", int(time.time())]
    else:
        sessions_lang[session][1] = int(time.time())
    queryResult = request_['queryResult']
    agent = WebhookClient(request_)
    agent = intent_switch(queryResult['intent']['displayName'], queryResult, agent, session)
    return get_response(agent)

@app.route('/', methods=['POST'])
def index():
    return "Chatbot-johan server"

def clear_expired_sessions():
    current_time = int(time.time())
    for session, lang_date in sessions_lang:
        if (current_time - lang_date[1] > 1800): # 30min afk
            del sessions_lang[session]

if __name__ == '__main__':
    schedule.every(1).minutes.do(clear_expired_sessions)
    app.run()
    
# TODO amélioration
# 1 thread/client sur le serv si besoin de gérer des variables comme la langue
# pouvoir changer la langue du bot dynamiquement (contrainte imposée par dialogflow)