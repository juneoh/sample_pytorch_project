# Sample PyTorch project

A sample PyTorch project that trains, validates, and tests ResNet-34 on the
Fashion MNIST dataset.

## Usage

```bash
python3 run.py
```

Use the `--help` option to list available arguments.

## Running on Docker

PyTorch | CUDA | CuDNN | Notes
:-----: | :--: | :---: | :---:
 0.3.0  | 9.0  |   7   | Requires >=`nvidia-384` on host

### 1. Install [Docker](https://www.docker.com/).

### 2. Install NVIDIA driver and CUDA library.

On Ubuntu:

```bash
sudo apt-get update && \
sudo apt-get install -y --no-install-recommends nvidia-384 libcuda-384
```

If you wish to avoid automatic upgrades:

```bash
sudo apt-mark hold nvidia-384 libcuda-384
```

### 3. Install [nvidia-docker](https://github.com/nvidia/nvidia-docker).

### 4. Build the image.

```bash
sudo docker build -t pytorch:v0.3.0 pytorch-v0.3.0
```

### 5. Run the container.

```bash
sudo docker run --rm --runtime nvidia --ipc host --pid host -dit \
-v $(pwd):/root --restart unless-stopped \
-h $(hostname)-pytorch --name $(hostname)-pytorch
pytorch:v0.3.0 bash -c 'pip install -r requirements.txt && python3 run.py'
```
