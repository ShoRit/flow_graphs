import fire

from preprocess.standardize_dataset_format import standardize_dataset_format
from preprocess.align_and_add_parses import align_and_add_parses

fire.Fire(
    {
        "standardize_dataset_format": standardize_dataset_format,
        "align_and_add_parses": align_and_add_parses,
    }
)
