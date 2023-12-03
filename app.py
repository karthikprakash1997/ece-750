from flask import Flask, request
from flask_cors import CORS
# from .src.utils.filters import suggestions

from src.api.auth import auth_blueprint
# from src.api.dashboard import dashboard_blueprint
# from src.api.data_dump import data_dump_blueprint
# from src.api.filters import filter_blueprint
# from src.api.overview import overview_blueprint
# from src.api.report import report_blueprint


app = Flask(__name__)
app.register_blueprint(auth_blueprint)
# app.register_blueprint(dashboard_blueprint)
# app.register_blueprint(data_dump_blueprint)
# app.register_blueprint(filter_blueprint)
# app.register_blueprint(overview_blueprint)
# app.register_blueprint(report_blueprint)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",
                "https://dev-client-nrcan.esg.uwaterloo.ca",
            ]
        }
    },
)

@app.route('/api/create-profile', methods=['GET'])
def index():
    # args = request.args
    # countryCode = json.loads(args.get('countryCode'))
    return "Welcome ece 750 project api"

# @app.route('/api/suggestion', methods=['GET'])
# def index():
#     sug = suggestions()
#     return sug

if __name__ == "__main__":
    app.run(debug=True)
