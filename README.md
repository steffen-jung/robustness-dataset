# Neural Architecture Design and Robustness: A Dataset

This repository contains the accompanying code for the ICLR 2023 publication "**Neural Architecture Design and Robustness: A Dataset**" by [Steffen Jung](http://jung.vision/), [Jovita Lukasik](https://jovitalukasik.github.io/), and [Margret Keuper](https://www.vc.informatik.uni-siegen.de/en/keuper-margret/). Visit our project page at http://robustness.vision/ for more information.

## Dataset

You can download the dataset from http://data.robustness.vision/.
The dataset is split into data sources (cifar10, cifar100, and ImageNet16-120), evaluation results (accuracies, confidence, cm), and attack type (adversarial, corruption) to keep file sizes manageable and evaluation results selectable. `cifar10.zip` contains all evaluations on cifar10 and `<dataset>-accuracies.zip` includes all attack types. You need to download the meta data file contained in `meta.zip` in any case if you want to use the provided helper class.

## Usage

This repository contains a helper class to access the data `robustness_dataset.py` as well as an example notebook `dataset.ipynb` that shows how to use the helper class. See below for a short introduction.

```python
from robustness_dataset import RobustnessDataset
data = RobustnessDataset(path="path_to_data_root")
results = data.query(
    # data specifies the evaluated dataset
    data = ["cifar10", "cifar100", "ImageNet16-120"],
    # measure specifies the evaluation type
    measure = "accuracy" # ["accuracy", "confidence", "cm"],
    # key specifies the attack types
    key = RobustnessDataset.keys_clean + RobustnessDataset.keys_adv + RobustnessDataset.keys_cc
)

# clean accuracy of architecture #13433 on cifar10
# get_uid returns unique architecture id (if given id is isomorph)
result["cifar10"]["clean"]["accuracy"][data.get_uid(13433)]
# 0.893

# pgd accuracy of architecture #13433 on cifar10 with eps=1.0
result["cifar10"]["pgd@Linf"]["accuracy"][data.get_uid(13433)][data.meta["epsilons"]["pgd@Linf"].index(1.0)]
# 0.336
```

## Citation

```bibtex
@inproceedings{Jung2023,
  author = {Steffen Jung and Jovita Lukasik and Margret Keuper},
  title = {Neural Architecture Design and Robustness: A Dataset},
  booktitle = {ICLR},
  year = {2023}
}
```

## Links
- Project website: http://robustness.vision/
- GitHub repository: http://code.robustness.vision/
- Dataset download: http://data.robustness.vision/
- OpenReview: https://openreview.net/forum?id=p8coElqiSDw
