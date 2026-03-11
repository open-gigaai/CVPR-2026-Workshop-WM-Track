# run_remote_model.py
import pickle
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "rb") as f:
    policy_name, model_name, args, kwargs = pickle.load(f)

policy_model = __import__(policy_name)
func = getattr(policy_model, model_name)
result = func(*args, **kwargs)

with open(output_file, "wb") as f:
    pickle.dump(result, f)
