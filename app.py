from flask import Flask, render_template, request, jsonify

#get_response is a function in chat.py for prediction
from chat import get_response

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    #check if text is valid
    responce = get_response(text)
    message = {"answer" : responce}
    return jsonify(message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    #app.run(debug=True)