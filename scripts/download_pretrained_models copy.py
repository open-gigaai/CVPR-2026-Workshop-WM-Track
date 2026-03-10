import subprocess
import os
import sys
import shutil
from cvpr_2026_workshop_wm_track.model_config import HUGGINGFACE_MODEL_CACHE, huggingface_model_config

# os.environ['HF_TOKEN'] = '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'

cur_dir = os.path.dirname(__file__)
python = sys.executable

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = HUGGINGFACE_MODEL_CACHE
# os.environ['HUGGINGFACE_HUB_CACHE'] = '/shared_disk/datasets/public_datasets'
# os.environ['HUGGINGFACE_HUB_CACHE'] = '/shared_disk/users/yukun.zhou/models'
# os.environ['HUGGINGFACE_HUB_CACHE'] = './huggingface'
# 使用可用的令牌

def list_dir(root_dir, recursive, exts=None):
    file_paths = []
    if recursive:
        for cur_dir, _, file_names in os.walk(root_dir, followlinks=True):
            file_names.sort()
            for file_name in file_names:
                file_path = os.path.join(cur_dir, file_name)
                if os.path.isfile(file_path):
                    if exts is None:
                        file_paths.append(file_path)
                    else:
                        suffix = os.path.splitext(file_name)[1].lower()
                        if suffix in exts:
                            file_paths.append(file_path)
    else:
        for file_name in sorted(os.listdir(root_dir)):
            file_path = os.path.join(root_dir, file_name)
            if os.path.isfile(file_path):
                if exts is None:
                    file_paths.append(file_path)
                else:
                    suffix = os.path.splitext(file_name)[1].lower()
                    if suffix in exts:
                        file_paths.append(file_path)
    return file_paths


def run(command, folder_path=None, try_until_success=True, **kwargs):
    run_kwargs = {
        'args': command,
        'shell': True,
        'env': os.environ,
        'encoding': 'utf8',
        'errors': 'ignore',
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
    }
    run_kwargs.update(kwargs)
    while True:
        print(command)
        result = subprocess.run(**run_kwargs)
        if result.returncode == 0:
            break
        else:
            if try_until_success:
                print(result.stderr, result.stdout)
                if folder_path is not None and os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
            else:
                raise ValueError(result.stderr, result.stdout)


def git_clone(
    url_path,
    repo_path=None,
    repo_name=None,
    recursive=True,
    prefix='https://mirror.ghproxy.com/',
    force=False,
    **kwargs,
):
    if repo_path is None:
        if repo_name is None:
            repo_name = url_path.split('/')[-1]
            assert repo_name.endswith('.git')
            repo_name = repo_name[:-4]
        repo_path = os.path.join(cur_dir, repo_name)
    if os.path.exists(repo_path):
        if force:
            shutil.rmtree(repo_path)
        else:
            return repo_path
    if prefix is not None:  # ref: https://ghproxy.com/
        url_path = prefix + url_path
    if recursive:
        command = f'git clone --recursive {url_path} {repo_path}'
    else:
        command = f'git clone {url_path} {repo_path}'
    run(command, folder_path=repo_path, try_until_success=True, **kwargs)
    return repo_path


def run_pip(command, **kwargs):
    run(f'{python} -m pip {command}', try_until_success=False, **kwargs)


def run_python(command, **kwargs):
    run(f'{python} {command}', try_until_success=False, **kwargs)


def download_huggingface_model(model_name, file_name=None, local_model_dir=None, repo_type=None, token="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", force=False, include=None, **kwargs):
    if local_model_dir is None:
        local_model_dir = os.environ['HUGGINGFACE_HUB_CACHE']
    assert len(model_name.split('/')) == 2
    local_model_name = 'models--' + model_name.replace('/', '--')
    local_model_path = os.path.join(local_model_dir, local_model_name)
    if file_name is not None:
        local_file_path = os.path.join(local_model_path, file_name)
        if os.path.exists(local_file_path):
            if force:
                os.remove(local_file_path)
            else:
                return local_file_path
        url_path = os.path.join(os.environ['HF_ENDPOINT'], model_name, 'resolve/main', file_name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        command = f'wget {url_path} -O {local_file_path}'
        run(command, try_until_success=True, **kwargs)
        return local_file_path
    else:
        if os.path.exists(local_model_path):
            if force:
                shutil.rmtree(local_model_path)
            else:
                return local_model_path
        cache_path = os.path.join(local_model_path, '_cache')
        command = 'huggingface-cli download --resume-download'
        command += ' --local-dir-use-symlinks False'
        if repo_type is not None:
            assert repo_type in ('model', 'dataset', 'space')
            command += f' --repo-type {repo_type}'

        if include is not None:
            command += f' --include "{include}"'

        if token is not None:
            command += f' --token {token}'
        command += f' --cache-dir {cache_path}'
        command += f' {model_name} --local-dir {local_model_path}'
        run(command, try_until_success=True, **kwargs)
        shutil.rmtree(cache_path)
        return local_model_path



def main():
    for model_name, model_config in huggingface_model_config.items():
        download_huggingface_model(
            model_name=model_config['model_name'],
            repo_type=model_config['repo_type'],
        )

    return 0

if __name__ == '__main__':
    main()

    # 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
