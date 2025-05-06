import os
import requests

# Constants for model source links
MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/NeoPy/rvc-base/resolve/main/'

# Base and model directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MDX_MODELS_DIR = os.path.join(BASE_DIR, 'mdxnet_models')
RVC_MODELS_DIR = os.path.join(BASE_DIR, 'rvc_models')

# Ensure directories exist
os.makedirs(MDX_MODELS_DIR, exist_ok=True)
os.makedirs(RVC_MODELS_DIR, exist_ok=True)


def download_model(base_url: str, filename: str, target_dir: str):
    """Download a model file from the specified URL into the target directory."""
    url = f'{base_url}{filename}'
    target_path = os.path.join(target_dir, filename)

    print(f'Downloading {filename}...')
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def download_models():
    """Download all required MDX and RVC models."""
    mdx_models = [
        'UVR-MDX-NET-Voc_FT.onnx',
        'UVR_MDXNET_KARA_2.onnx',
        'Reverb_HQ_By_FoxJoy.onnx'
    ]
    rvc_models = [
        'hubert_base.pt',
        'rmvpe.pt',
        'fcpe.pt'
    ]

    for model in mdx_models:
        download_model(MDX_DOWNLOAD_LINK, model, MDX_MODELS_DIR)

    for model in rvc_models:
        download_model(RVC_DOWNLOAD_LINK, model, RVC_MODELS_DIR)

    print('All models downloaded successfully!')


if __name__ == '__main__':
    download_models()
