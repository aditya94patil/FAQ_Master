import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "FAQ Master"

import re
def is_valid_operator_expression(sentence):
    pattern = r"\d+(\.\d+)?\s*[\+\-\*/]\s*\d+(\.\d+)?"
    return re.search(pattern, sentence) is not None

def evaluate_arithmetic_expression(expression):
    try:
        return eval(expression)
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except:
        return "Error: Invalid arithmetic expression."

def get_response(msg):

    if is_valid_operator_expression(msg):
        result = evaluate_arithmetic_expression(msg)
        return "The result is: " + str(result)
    
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)

    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    if torch.all(X == 0) :
        return "I do not understand..."

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("User: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print("Bot: " + resp)

