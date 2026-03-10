import os
import sys

project_name = None
cur_dir = os.path.dirname(os.path.abspath(__file__))
python_paths = [
    os.path.join(cur_dir, '../third_party/giga-datasets/'),
    os.path.join(cur_dir, '../'),
    os.path.join(cur_dir, '../third_party/giga-train/'),
    os.path.join(cur_dir, '../third_party/diffusers_src/'),
    os.path.join(cur_dir, '../third_party/Video-Depth-Anything/'),
]
print("python_paths: ", python_paths)
if project_name is not None:
    python_paths.append(os.path.join(cur_dir, project_name))
for python_path in python_paths:
    sys.path.insert(0, python_path)
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] += ':{}'.format(python_path)
    else:
        os.environ['PYTHONPATH'] = python_path