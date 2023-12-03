from flask import Blueprint, jsonify, request
from ..utils.filters import suggestions, feedback
import json

auth_blueprint = Blueprint('auth_blueprint', __name__)

@auth_blueprint.route('/api/suggestions', methods=['GET'])
def index():
    sug = suggestions()
    return jsonify({"results": sug})

@auth_blueprint.route('/api/feedback', methods=['GET'])
def feedback_service():
    args = request.args
    api_response_openai = json.loads(args.get("api_response_openai"))
    feedback = json.loads(args.get("feedback"))
    sug = feedback(api_response_openai, feedback)
    # return sug
    return jsonify({"results": sug})
