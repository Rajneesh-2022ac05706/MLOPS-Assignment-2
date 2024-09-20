from flask import Flask, request, jsonify
from model_development.model_scoring import predict

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Working"

@app.route('/predict', methods=['POST'])
def batch_predict():
    json_list = request.json
    predictions = predict(json_list)
    return jsonify(["Congratulations! Your application is Approved" if i>0.7 else 
                    "Sorry, Your application is Not Approved" for i in predictions])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
