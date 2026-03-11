import os

HUGGINGFACE_MODEL_CACHE = "/shared_disk/models/huggingface"
DEFAULT_PROMPT_EMBEDDING_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "asserts", "default_prompt_embeds.pth")  # code repo assert os.path.exists(DEFAULT_PROMPT_EMBEDDING_PATH)
DATA_DIR = "/shared_disk/users/yukun.zhou/dataset/CVPR-2026-Workshop-WM-Track-Dataset/"
assert os.path.exists(DEFAULT_PROMPT_EMBEDDING_PATH), os.path.abspath(DEFAULT_PROMPT_EMBEDDING_PATH) + "not exist"

model_config = {
    'wan2.2-5b-diffusers': os.path.join(HUGGINGFACE_MODEL_CACHE, "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"),
    'paligemma': os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--google--paligemma-3b-pt-224'),
    'fast-tokenizer': os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--physical-intelligence--fast'),
    'video-depth-anything': os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--depth-anything--Video-Depth-Anything-Large'),
    "cvpr-2026-worldmodel-track-model-task1": os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--open-gigaai--CVPR-2026-WorldModel-Track-Model-Task1'),
    "cvpr-2026-worldmodel-track-model-task2": os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--open-gigaai--CVPR-2026-WorldModel-Track-Model-Task2'),
    "cvpr-2026-worldmodel-track-model-task3": os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--open-gigaai--CVPR-2026-WorldModel-Track-Model-Task3'),
    "cvpr-2026-worldmodel-track-model-task4": os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--open-gigaai--CVPR-2026-WorldModel-Track-Model-Task4'),
    "cvpr-2026-worldmodel-track-model-task5": os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--open-gigaai--CVPR-2026-WorldModel-Track-Model-Task5'),
    "cvpr-2026-worldmodel-track-model-task6": os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--open-gigaai--CVPR-2026-WorldModel-Track-Model-Task6'),
    "cvpr-2026-worldmodel-track-model-task7": os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--open-gigaai--CVPR-2026-WorldModel-Track-Model-Task7'),
    "cvpr-2026-worldmodel-track-model-task8": os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--open-gigaai--CVPR-2026-WorldModel-Track-Model-Task8'),
}

huggingface_model_config = {
    "wan2.2-5b-diffusers": {
        "model_name": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "repo_type": "model"
    },
    "paligemma": {
        "model_name": "google/paligemma-3b-pt-224",
        "repo_type": "model"
    },
    "fast-tokenizer": {
        "model_name": "physical-intelligence/fast",
        "repo_type": "model"
    },
    "video-depth-anything": {
        "model_name": "depth-anything/Video-Depth-Anything-Large",
        "repo_type": "model",
    },
}


gigabrain_model_config = {
    "cvpr-2026-worldmodel-track-model-task1": {
        "model_name": "open-gigaai/CVPR-2026-WorldModel-Track-Model-Task1",
        "repo_type": "model"
    },
    "cvpr-2026-worldmodel-track-model-task2": {
        "model_name": "open-gigaai/CVPR-2026-WorldModel-Track-Model-Task2",
        "repo_type": "model"
    },
    "cvpr-2026-worldmodel-track-model-task3": { 
        "model_name": "open-gigaai/CVPR-2026-WorldModel-Track-Model-Task3",
        "repo_type": "model"
    },
    "cvpr-2026-worldmodel-track-model-task4": {
        "model_name": "open-gigaai/CVPR-2026-WorldModel-Track-Model-Task4",
        "repo_type": "model"
    },
    "cvpr-2026-worldmodel-track-model-task5": {     
        "model_name": "open-gigaai/CVPR-2026-WorldModel-Track-Model-Task5",
        "repo_type": "model"
    },
    "cvpr-2026-worldmodel-track-model-task6": {
        "model_name": "open-gigaai/CVPR-2026-WorldModel-Track-Model-Task6",
        "repo_type": "model"
    },
    "cvpr-2026-worldmodel-track-model-task7": {
        "model_name": "open-gigaai/CVPR-2026-WorldModel-Track-Model-Task7",
        "repo_type": "model"
    },
    "cvpr-2026-worldmodel-track-model-task8": {             
        "model_name": "open-gigaai/CVPR-2026-WorldModel-Track-Model-Task8",
        "repo_type": "model"
    },
}