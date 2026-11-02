# Train RaCo

RaCo training uses three stages: detector and backbone, covariance, then ranker.
The configs are in `gluefactory/configs/raco_training/`.

## Install

```bash
git clone https://github.com/cvg/glue-factory.git
cd glue-factory
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[raco]'
```

The PyTorch command is for CUDA 12.8. Use the PyTorch install command for the
target system when its platform differs.

## Data

Training uses RevisitOP1M with this layout:

```text
data/revisitop1m/
├── revisitop1m.txt
└── jpg/
    └── ...
```

Each line of `revisitop1m.txt` is a path relative to `jpg/`. For data stored
elsewhere, add `data.data_dir=/absolute/path/to/revisitop1m` to each command.
The existing downloader retrieves the full collection (about 450 GB):

```bash
python -c "from gluefactory.datasets import get_dataset; get_dataset('homographies')({})"
```

## Train

Run the commands from the Glue Factory repository root. Train the detector
first, then train the covariance and ranker heads in either order. Each head has
its own loss and can be trained from the detector checkpoint or from a
checkpoint containing the other trained head.

### Detector and backbone

```bash
python -m gluefactory.train raco_detector \
    --conf gluefactory/configs/raco_training/detector.yaml
```

Output: `outputs/training/raco_detector/checkpoint_best.tar`

### Covariance

```bash
python -m gluefactory.train raco_covariance \
    --conf gluefactory/configs/raco_training/covariance.yaml \
    model.extractor.weights=outputs/training/raco_detector/checkpoint_best.tar
```

Output: `outputs/training/raco_covariance/checkpoint_best.tar`

### Ranker

```bash
python -m gluefactory.train raco_ranker \
    --conf gluefactory/configs/raco_training/ranker.yaml \
    model.extractor.weights=outputs/training/raco_covariance/checkpoint_best.tar
```

Output: `outputs/training/raco_ranker/checkpoint_best.tar`

To train the ranker before covariance instead, run:

```bash
python -m gluefactory.train raco_ranker \
    --conf gluefactory/configs/raco_training/ranker.yaml \
    model.extractor.weights=outputs/training/raco_detector/checkpoint_best.tar

python -m gluefactory.train raco_covariance \
    --conf gluefactory/configs/raco_training/covariance.yaml \
    model.extractor.weights=outputs/training/raco_ranker/checkpoint_best.tar
```

In this order, the final checkpoint is
`outputs/training/raco_covariance/checkpoint_best.tar`.

The configs use 400,000 training images and 1,000 validation images. The
covariance stage uses 40,000 training images.

## Load the final checkpoint in RaCo

The Glue Factory extractor accepts its checkpoint directly. The standalone
[RaCo release](https://github.com/cvg/RaCo) also loads it with the companion
checkpoint-loading update:

```python
from raco import RaCo

model = RaCo(weights="outputs/training/raco_ranker/checkpoint_best.tar").eval()
```

See the [RaCo guide](raco.md) for Glue Factory inference configs and links to the
standalone inference and matching demos.
