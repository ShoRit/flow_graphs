import fire

from chemu import create_chemu
from efgc import create_japflow
from mscorpus import create_mscorpus
from risec import create_risec

DATASET_TO_FUNCTION = {
    "chemu": create_chemu,
    "efgc": create_japflow,
    "mscorpus": create_mscorpus,
    "risec": create_risec,
}


def standardize_dataset_format(dataset):
    standardize_fn = DATASET_TO_FUNCTION.get(dataset)
    if function is None:
        print(
            f"Specified dataset {dataset} not in list of allowable options: {DATASET_TO_FUNCTION.keys()}"
        )
        exit(1)
    else:
        standardize_fn()


if __name__ == "__main__":
    fire.Fire()
