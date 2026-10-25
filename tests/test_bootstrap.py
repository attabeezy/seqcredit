"""The comparison uses the same borrower draws for every model."""

from bootstrap_predictions import bootstrap


def test_paired_differences_and_single_class_skips(tmp_path):
    with (tmp_path / "predictions.csv").open("w", encoding="utf-8") as handle:
        handle.write("borrower_id,partition,target,logistic_regression,lightgbm,gru\n")
        for i in range(8):
            y = int(i == 0)
            score = 0.8 if y else 0.2
            handle.write(f"u{i},test,{y},{score},{score},{score}\n")
    report = bootstrap(tmp_path, draws=20)
    assert report["draws_used"] + report["single_class_draws_skipped"] == 20
    for comparison in report["paired_differences"].values():
        assert comparison["roc_auc"]["lower"] == comparison["roc_auc"]["upper"] == 0
