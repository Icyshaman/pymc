from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

class SBot:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_routes()
        API_KEY = os.getenv("API_KEY")
        PROJECT_ID = os.getenv("PROJECT_ID")
        REGION_URL = os.getenv("REGION_URL")
        credentials = {
            "apikey": API_KEY,
            "url": REGION_URL
        }
        self.api_client = APIClient(credentials)
        self.api_client.set.default_project(PROJECT_ID)
        self.model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=credentials,
            project_id=PROJECT_ID
        )
    
    def get_problem_solution(self, user_input):
        prompt = f"How can we fix the following: '{user_input}'. Return only plain text without any examples"
        parameters = {
            "decoding_method": "sample",
            "temperature": 0.5,
            "top_k": 60,
            "top_p": 0.8,
            "max_new_tokens": 250
        }
        response = self.model.generate(
            prompt=prompt, params=parameters
        )
        return response["results"][0]["generated_text"].strip()

    def setup_routes(self):
        @self.app.route("/get-solution", methods=["POST"])
        def get_solution():
            data = request.json
            if "sentence" not in data:
                return jsonify({"error": "No sentence provided"}), 400
            user_input = data["sentence"].lower()
            if not user_input or not user_input.strip():
                return jsonify({"error": "No sentence provided"}), 400
            response = self.get_problem_solution(user_input)
            return jsonify({"solution": response}), 200
    
sbot = SBot()
app = sbot.app
