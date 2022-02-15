import json
import numpy as np
import os

with open("data/test/test_client400.json") as f:
   test_data = json.load(f)

user_names = test_data['users']

for u in user_names:
   print(u)
   np.savez('data/test/test_np/'+str(u), x=test_data['user_data'][u]['x'], y=test_data['user_data'][u]['y'])


with open("data/train/train_client400.json") as f:
   train_data = json.load(f)

user_names = train_data['users']

for u in user_names:
   print(u)
   np.savez('data/train/train_np/'+str(u), x=train_data['user_data'][u]['x'], y=train_data['user_data'][u]['y'])
