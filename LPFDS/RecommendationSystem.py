import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM


data = fetch_movielens(min_rating = 3.0)

print(repr(data['train']))
print(repr(data['test']))

# Create Model
model = LightFM(loss='warp')
# Train Model
model.fit(data['train'], epochs=30, num_threads=3)

def sample_recommendation(model, data, user_id_arr):
    n_users, n_items = data['train'].shape
    for user_id in user_id_arr:
        #Movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("       Known positives:")
        
        for x in known_positives[:3]:
            print("          %s" % x)

        print("       Recommended:")
        
        for x in top_items[:3]:
            print("          %s" % x)
    #End For
#End sample_recommendation

sample_recommendation(model, data, [3, 25, 450])