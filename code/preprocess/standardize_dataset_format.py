import fire

from preprocess.chemu import standardize_chemu
from preprocess.efgc import standardize_efgc
from preprocess.mscorpus import standardize_mscorpus
from preprocess.risec import standardize_risec

DATASET_TO_FUNCTION = {
    "chemu": standardize_chemu,
    "efgc": standardize_efgc,
    "mscorpus": standardize_mscorpus,
    "risec": standardize_risec,
}


def standardize_dataset_format_wrapper(dataset):
    standardize_fn = DATASET_TO_FUNCTION.get(dataset)
    if standardize_fn is None:
        print(
            f"Specified dataset {dataset} not in list of allowable options: {DATASET_TO_FUNCTION.keys()}"
        )
        exit(1)
    else:
        standardize_fn()


if __name__ == "__main__":
    fire.Fire(standardize_dataset_format_wrapper)
