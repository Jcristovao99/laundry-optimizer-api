from flask import Flask, request, jsonify
from laundry_optimizer import optimize_order

app = Flask(__name__)

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.get_json()
    items = data["items"]
    delivery = data.get("delivery_location", "default")
    total, breakdown, _ = optimize_order(items, delivery)
    return jsonify({"total_cost": total, **breakdown})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))  # Use the port provided by Render
    app.run(host='0.0.0.0', port=port, debug=True)
