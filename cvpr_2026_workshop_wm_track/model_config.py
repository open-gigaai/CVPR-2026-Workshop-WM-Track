import os

HUGGINGFACE_MODEL_CACHE = "/shared_disk/models/huggingface"

model_config = {
    'wan2.2-5b-diffusers': os.path.join(HUGGINGFACE_MODEL_CACHE, "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"),
    'paligemma': os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--google--paligemma-3b-pt-224'),
    'fast-tokenizer': os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--physical-intelligence--fast'),
    'video-depth-anything': os.path.join(HUGGINGFACE_MODEL_CACHE, 'models--depth-anything--Video-Depth-Anything-Large'),
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