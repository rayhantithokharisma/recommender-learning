from flask import Flask, request, jsonify
import pandas as pd
import implicit
import numpy as np
import pickle
from scipy.sparse import csr_matrix

app = Flask(__name__)


with open('user_to_type.pickle', 'rb') as handle:
    user_to_type = pickle.load(handle)

with open('user_to_id.pickle', 'rb') as handle:
    user_to_id = pickle.load(handle)
    
with open('id_to_item.pickle', 'rb') as handle:
    id_to_item = pickle.load(handle)
    
high_selling_items = pd.read_parquet('high_selling_items.parquet')
undersale_items = pd.read_parquet('undersale_items.parquet')
loaded_model = implicit.als.AlternatingLeastSquares().load("recommender_v1.model.npz")

user_items = np.load('user_items_matrix.npy')
user_items = csr_matrix(user_items)

@app.route('/recommend_items', methods=['POST'])
def recommend_items():
    try:
        data = request.get_json()

        # Check if 'user_id' is present in the request
        if 'user_id' not in data:
            return jsonify({'error': 'User ID is missing'}), 400

        user_id = data['user_id']

        # Check if the user ID exists in the user_type_dict
        if user_id not in user_to_type:
            return jsonify(
            {'user_type': 'Cold Start',
            'recommended_items':high_selling_items.iloc[:30]['product_id'].tolist()
            })

        # Get the user type
        user_type = user_to_type[user_id]
        
        if user_type == 'High View, High Purchase, High Unique':
            return jsonify({'user_type':user_type,
                           'recommended_items':undersale_items.iloc[:30]['product_id'].tolist()})
        
        elif user_type == 'Low View, Low Purchase, Low Unique':
            return jsonify(
            {'user_type': user_type,
            'recommended_items':high_selling_items.iloc[:30]['product_id'].tolist()
            })
        
        else:
            user = user_to_id[user_id]
            recommended_items = loaded_model.recommend(
                user, user_items[user], N=30, filter_already_liked_items=True
            )[0].tolist()
            recommended_items = [id_to_item[x] for x in recommended_items]
            
            return jsonify({'user_type': user_type, 'recommended_items': recommended_items})

    except Exception as e:
        print(str(e))
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
