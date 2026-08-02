import json
from collections import Counter

from config import DATASET_PATH


def clean_text(text):
    """
    Basic text preprocessing.
    """
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def load_cuad_dataset(file_path=DATASET_PATH):
    """
    Load the official CUAD dataset and extract
    clause text with corresponding labels.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = []

    for contract in data["data"]:

        for paragraph in contract["paragraphs"]:

            for qa in paragraph["qas"]:

                # Ignore clauses that don't exist
                if qa.get("is_impossible", False):
                    continue

                label = qa["id"].split("__",1)[-1]

                for answer in qa["answers"]:

                    text = clean_text(answer["text"])

                    if len(text) == 0:
                        continue

                    dataset.append(
                        {
                            "text": text,
                            "label": label
                        }
                    )

    return dataset


def encode_labels(dataset):
    """
    Encode labels into integer IDs.
    """

    unique_labels = sorted(
        list(set(sample["label"] for sample in dataset))
    )

    label2id = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    for sample in dataset:
        sample["label_id"] = label2id[sample["label"]]

    return dataset, label2id, id2label


def dataset_statistics(dataset):
    """
    Print dataset statistics.
    """

    counter = Counter()

    for sample in dataset:
        counter[sample["label"]] += 1

    print("=" * 60)
    print("OFFICIAL CUAD DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal Samples : {len(dataset)}")
    print(f"Total Labels  : {len(counter)}")

    print("\nTop 10 Labels:\n")

    for label, count in counter.most_common(10):
        print(f"{label:<35} {count}")


if __name__ == "__main__":

    dataset = load_cuad_dataset(DATASET_PATH)

    dataset, label2id, id2label = encode_labels(dataset)

    print(f"Total Samples : {len(dataset)}")
    print(f"Total Labels  : {len(label2id)}")

    dataset_statistics(dataset)
