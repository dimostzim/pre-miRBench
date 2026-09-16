#!/usr/bin/env python3
"""Estimate AP intervals and paired differences on the four fixed test sets."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
RESAMPLES = 10_000
SEED = 42

SPLITS = {
    "test_known_species_known_family": "Test 1",
    "test_known_species_heldout_family": "Test 2",
    "test_heldout_species_known_family": "Test 3",
    "test_heldout_species_heldout_family": "Test 4",
}

METHODS = {
    "deepmir": "DeepMir",
    "deepmirgene": "deepMiRGene",
    "dnnpremir": "dnnPreMiR",
    "mirdnn": "mirDNN",
    "mire2e": "miRe2e",
    "mustard": "MuStARD",
    "premirbench": "pre-miRBench",
}


def prepare_split(predictions, split):
    records = predictions[predictions.split == split]
    reference = records[records.tool == "premirbench"].sort_values("input_order")
    record_ids = reference.record_id.tolist()
    labels = reference.set_index("record_id").label.loc[record_ids]
    scores = records.pivot(index="record_id", columns="tool", values="score")
    scores = scores.loc[record_ids, list(METHODS)]
    assert not scores.isna().any().any()
    for method in METHODS:
        method_labels = records[records.tool == method].set_index("record_id").label.loc[record_ids]
        pd.testing.assert_series_equal(method_labels, labels, check_names=False)
    return labels.to_numpy(dtype=int), scores.to_numpy(dtype=float)


def bootstrap_split(task):
    split, labels, scores = task
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    generator = np.random.default_rng(SEED)
    bootstrap_ap = np.empty((RESAMPLES, len(METHODS)), dtype=float)

    for replicate in range(RESAMPLES):
        indices = np.concatenate(
            [
                generator.choice(positive_indices, len(positive_indices), replace=True),
                generator.choice(negative_indices, len(negative_indices), replace=True),
            ]
        )
        replicate_labels = labels[indices]
        for method_index in range(len(METHODS)):
            bootstrap_ap[replicate, method_index] = average_precision_score(
                replicate_labels, scores[indices, method_index]
            )

    point_estimates = np.array(
        [average_precision_score(labels, scores[:, index]) for index in range(len(METHODS))]
    )
    lower, upper = np.quantile(bootstrap_ap, [0.025, 0.975], axis=0)
    reference_index = list(METHODS).index("premirbench")
    differences = bootstrap_ap[:, reference_index, None] - bootstrap_ap
    difference_lower, difference_upper = np.quantile(differences, [0.025, 0.975], axis=0)

    return {
        "split": split,
        "positives": len(positive_indices),
        "negatives": len(negative_indices),
        "point_estimates": point_estimates,
        "lower": lower,
        "upper": upper,
        "difference_lower": difference_lower,
        "difference_upper": difference_upper,
    }


def main():
    published = pd.read_csv(RESULTS / "predictions" / "published_tools.csv")
    premirbench = pd.read_csv(RESULTS / "predictions" / "premirbench_model.csv")
    predictions = pd.concat([published, premirbench], ignore_index=True)

    assert set(predictions.tool) == set(METHODS)
    assert set(predictions.split) == set(SPLITS)
    assert not predictions.duplicated(["tool", "split", "record_id"]).any()

    tasks = []
    for split in SPLITS:
        labels, scores = prepare_split(predictions, split)
        tasks.append((split, labels, scores))

    with ProcessPoolExecutor(max_workers=len(SPLITS)) as executor:
        results = list(executor.map(bootstrap_split, tasks))

    interval_rows = []
    difference_rows = []
    for result in results:
        split = result["split"]
        for method_index, (method, display_name) in enumerate(METHODS.items()):
            interval_rows.append(
                {
                    "test": SPLITS[split],
                    "split": split,
                    "method": display_name,
                    "positives": result["positives"],
                    "negatives": result["negatives"],
                    "AP": result["point_estimates"][method_index],
                    "CI_2.5%": result["lower"][method_index],
                    "CI_97.5%": result["upper"][method_index],
                    "resamples": RESAMPLES,
                    "seed": SEED,
                }
            )
            if method != "premirbench":
                reference_index = list(METHODS).index("premirbench")
                difference_rows.append(
                    {
                        "test": SPLITS[split],
                        "split": split,
                        "comparison": f"pre-miRBench minus {display_name}",
                        "AP_difference": (
                            result["point_estimates"][reference_index]
                            - result["point_estimates"][method_index]
                        ),
                        "CI_2.5%": result["difference_lower"][method_index],
                        "CI_97.5%": result["difference_upper"][method_index],
                        "resamples": RESAMPLES,
                        "seed": SEED,
                    }
                )

    intervals = pd.DataFrame(interval_rows)
    differences = pd.DataFrame(difference_rows)
    intervals.to_csv(HERE / "ap_bootstrap_95ci.csv", index=False, float_format="%.10f")
    differences.to_csv(
        HERE / "ap_difference_vs_premirbench_95ci.csv", index=False, float_format="%.10f"
    )
    formatted = intervals.assign(
        value=intervals.apply(
            lambda row: (
                f'{row["AP"]:.4f} '
                f'({row["CI_2.5%"]:.4f}–{row["CI_97.5%"]:.4f})'
            ),
            axis=1,
        )
    ).pivot(index="method", columns="test", values="value")
    formatted = formatted.loc[list(METHODS.values()), list(SPLITS.values())]
    formatted.index.name = "Predictor"
    formatted.to_csv(HERE / "table2_ap_with_ci.tsv", sep="\t")

    joint = differences[
        (differences.test == "Test 4")
        & (differences.comparison == "pre-miRBench minus dnnPreMiR")
    ].iloc[0]
    np.testing.assert_allclose(
        [joint["CI_2.5%"], joint["CI_97.5%"]],
        [-0.0445085822, 0.0341162899],
        rtol=0,
        atol=5e-11,
    )
    print(intervals.to_string(index=False))
    print("\nVerified the saved Test 4 paired interval.")


if __name__ == "__main__":
    main()
