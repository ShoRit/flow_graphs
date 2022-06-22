import dill
import fire


def has_valid_masks(instance):
    return instance["amr_data"].n1_mask.sum() > 0 and instance["amr_data"].n2_mask.sum() > 0


def filter_by_amr_validity(data):
    filtered_data = {}

    for split, split_data in data.items():
        filtered_instances = [
            instance for instance in split_data["rels"] if has_valid_masks(instance)
        ]
        filtered_data[split] = {"rels": filtered_instances}
    return filtered_data


def create_valid_amr_subset_wrapper(data_path: str, output_path: str):
    with open(data_path, "rb") as f:
        amr_data = dill.load(f)

    filtered_data = filter_by_amr_validity(amr_data)

    with open(output_path, "wb") as f:
        dill.dump(filtered_data, f)


if __name__ == "__main__":
    fire.Fire(create_valid_amr_subset_wrapper)
