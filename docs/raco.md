# RaCo

[RaCo](https://github.com/cvg/RaCo) predicts keypoints, ranking scores, and spatial
covariances. Its three training stages use synthetic homographies of unlabelled
Oxford–Paris 1M distractor images. No depth, poses, or covariance labels are required.

## Installation

Install PyTorch and torchvision for your hardware, then install Glue Factory:

```bash
python -m pip install -e '.[raco]'
```

The `raco` extra includes the differentiable sorting dependency for ranker training
and the Albumentations version used by the photometric recipe. Inference and the
other training stages do not import differentiable sorting. No submodules or
equivariant convolution dependencies are needed.

## Training

Follow [Train RaCo](train_raco.md) for installation, dataset preparation, and the
detector, covariance, and ranker commands. Training configs live in
`gluefactory/configs/raco_training/`.

## Inference and checkpoints

```python
from gluefactory.models.extractors.raco import RaCo, WEIGHTS_URL

model = RaCo({"trainable": False, "weights": WEIGHTS_URL}).eval()
pred = model({"image": image})  # RGB or grayscale, [B, C, H, W], values in [0, 1]
```

Outputs include `keypoints` (`[B, N, 2]`, pixel-center coordinates),
`keypoint_scores` and `ranker_scores` (`[B, N]`), and `covariances`
(`[B, N, 2, 2]`, in squared pixels). Set `ranker` or `covariance_estimator` to
`False` to omit that head. `sort_by_ranker=True` orders all sparse outputs together.
`max_num_keypoints` controls the inference budget; `max_sampled_keypoints` controls
the training budget. Detector training and validation additionally return
`raw_logits` for supervision.

The `raco.yaml` config loads official weights into the upstream two-view pipeline.
Local `weights` may be either a public state dictionary or a Glue Factory training
checkpoint. A newly trained head can be absent when initializing its stage;
missing frozen weights and incompatible shapes fail explicitly.

Training produces standard Glue Factory checkpoints under
`outputs/training/<run_name>_<stage>/`. No export or conversion is needed: with
the companion checkpoint-loading update in `cvg/RaCo`, use the final checkpoint
directly in the standalone package:

```python
from raco import RaCo

model = RaCo(weights="outputs/training/raco_ranker/checkpoint_best.tar").eval()
```

## Matching examples

The configs in `gluefactory/configs/` combine RaCo detections with ALIKED
descriptors, as in the public
[`matching_demo.ipynb`](https://github.com/cvg/RaCo/blob/main/matching_demo.ipynb).
The descriptor adapter describes RaCo's points directly and preserves their order,
ranking scores, and covariances. It uses the image resolution supplied to the
pipeline without an additional resize.

| Config | Matcher |
| --- | --- |
| `raco+aliked+lightglue-official.yaml` | RaCo-trained LightGlue+ (`raco-aliked` weights), matching the public notebook |
| `raco+aliked+NN.yaml` | Mutual nearest neighbors on ALIKED descriptors |
| `raco.yaml` | RaCo extraction without descriptors or matching |

Load a complete matching pipeline with:

```python
from omegaconf import OmegaConf
from gluefactory.models import get_model

conf = OmegaConf.load("gluefactory/configs/raco+aliked+lightglue-official.yaml")
model = get_model(conf.model.name)(conf.model).eval()
pred = model({"view0": {"image": image0}, "view1": {"image": image1}})
```

Each image has shape `[B, C, H, W]` with values in `[0, 1]`. In addition to RaCo's
outputs, the pipeline returns `descriptors0/1`, `matches0/1`, and matching scores.
Use the upstream evaluation commands once the corresponding datasets are available:

```bash
python -m gluefactory.eval.hpatches \
    --conf gluefactory/configs/raco+aliked+lightglue-official.yaml
python -m gluefactory.eval.megadepth1500 \
    --conf gluefactory/configs/raco+aliked+NN.yaml
```

For standalone inference and visualization, use the RaCo release's
[keypoint/covariance demo](https://github.com/cvg/RaCo/blob/main/demo.ipynb) or
[matching demo](https://github.com/cvg/RaCo/blob/main/matching_demo.ipynb). The
configs above are available in Glue Factory for its evaluation pipeline.

Within a mixed extractor, detector settings are nested under
`model.extractor.detector`; for example, set
`model.extractor.detector.max_num_keypoints=1024`. Both configs include HPatches
and MegaDepth-1500 benchmark settings.
