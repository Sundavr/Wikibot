# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:03:40 2020

@author: Johan
"""
import os 
import pickle
import wikipedia

from os import path
from typing import Dict
from flask import Flask, request
from dialogflow_fulfillment import WebhookClient, QuickReplies

dir_name = os.path.dirname(os.path.realpath(__file__))
if (not path.exists(dir_name + "/sentiment-analysis")):
    from transformers import pipeline
    classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
    pickle.dump(classifier, open(dir_name + "/sentiment-analysis", "wb"))
else:
    classifier = pickle.load(open(dir_name + "/sentiment-analysis", "rb"))

language = "fr" # by default
wikipedia.set_lang(language)
app = Flask(__name__)

# TESTS
print(classifier('We are very happy to show you the 🤗 Transformers library.')) # POSITIVE
print(classifier("Alad'2 est clairement le meilleur film de l'année 2018.")) # POSITIVE
print(classifier("Juste whoaaahouuu !")) # POSITIVE
print(classifier("NUL...A...CHIER ! FIN DE TRANSMISSION.")) # NEGATIVE
print(classifier("Je m'attendais à mieux de la part de Franck Dubosc !")) # NEGATIVE
print(classifier("ça pu la merde !!")) # NEGATIVE

# ---------- INTENTS START ---------- #
def fullfilment_test(queryResult: dict, agent: WebhookClient) -> WebhookClient:
    agent.add("How are you feeling today ?")
    agent.add(QuickReplies(quick_replies=["Happy :)", "Sad :("]))
    return agent

def get_summary(queryResult: dict, agent: WebhookClient) -> WebhookClient:
    information = queryResult["parameters"]["information"]
    try:
        agent.add(wikipedia.summary(information).replace(" ()",'').replace("()", '')) 
    except Exception:
        agent.add("Désolé, mais je suis incapable de te dire ce qu'est '" + information + "'")
    return agent

def how_are_you(queryResult: dict, agent: WebhookClient) -> WebhookClient:
    agent.add(str(classifier(queryResult.get("queryText"))))
    return agent

def quick_summary(queryResult: dict, agent: WebhookClient) -> WebhookClient:
    information = queryResult["parameters"]["information"]
    try:
        agent.add(wikipedia.summary(information, sentences=1).replace(" ()",'').replace("()", '')) 
    except Exception:
        agent.add("Désolé, mais je suis incapable de te dire ce qu'est '" + information + "'")
    return agent

def set_language(queryResult: dict, agent: WebhookClient) -> WebhookClient:
    language = queryResult["parameters"]["language"].lower()
    language_changed = False
    if ((language == "fr") or (language == "french")):
        language = "fr"
        language_changed = True
    elif ((language == "en") or (language == "english")):
        language = "en"
        language_changed = True
    if (language_changed):
        wikipedia.set_lang(language)
        agent.add("Et hop ! Je peux maintenant parler en " + language) # TODO
    else:
        agent.add("Désolé mais je n'ai pas la chance de parler cette langue")
    return agent

# ---------- INTENTS END ---------- #

def intent_switch(intent_name: str, queryResult: dict, agent: WebhookClient) -> str:
    switcher = {
        "fullfilment_test": fullfilment_test,
        "get_summary": get_summary,
        "how_are_you": how_are_you,
        "quick_summary": quick_summary,
        "set_language": set_language,
    }
    func = switcher.get(intent_name)
    if (func != None):
        return func(queryResult, agent)
    print("Unknown intent : " + intent_name)
    agent.add("Toute mes excuses, mais mon serveur à eu un léger soucis à comprendre ton intention :(")
    return agent

def handler(agent: WebhookClient) -> None:
    pass

@app.route('/', methods=['POST'])
def webhook() -> Dict:
    request_ = request.get_json(force=True)
    #session = request_['session'].split("/")[-1] # TODO
    queryResult = request_['queryResult']
    agent = WebhookClient(request_)
    agent = intent_switch(queryResult['intent']['displayName'], queryResult, agent)
    return agent.response


if __name__ == '__main__':
    app.run()