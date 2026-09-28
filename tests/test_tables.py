"""Benchmark tables and figure data built from a small benchmark run."""
import pandas as pd
import pytest
from conftest import load_script


def test_tables_and_figure_data(benchmark_dir, tmp_path):
    out = tmp_path / "tables"
    load_script("make_tables").main(["--results", str(benchmark_dir / "results" / "results.csv"),
                                     "--runs-dir", str(benchmark_dir / "runs"), "--out", str(out),
                                     "--bootstrap-samples", "200", "--profile", str(tmp_path / "missing.csv")])
    lines = (out / "main.md").read_text().splitlines()
    assert lines[2].startswith("| Persistence") and lines[3].startswith("| Climatology")
    fno = next(line for line in lines if line.startswith("| FNO (ox)"))
    assert fno.count("[") >= 5  # CIs on RMSE, MAE, MB, r and the paired difference vs U-Net
    unet = next(line for line in lines if line.startswith("| U-Net (ox)"))
    assert unet.rstrip(" |").endswith("–")  # U-Net is the reference, no difference with itself
    assert "--" in (out / "main.tex").read_text() and "–" not in (out / "main.tex").read_text()
    for name in ("lead_rmse.md", "lead_rmse.tex", "figures/lead_rmse.csv", "figures/spectra.csv",
                 "figures/lead_rmse.png", "figures/spectra.png", "figures/error_maps.png"):
        assert (out / name).exists(), name
    lead = pd.read_csv(out / "figures" / "lead_rmse.csv")
    assert set(lead.lead_hour) == set(range(1, 11)) and lead.ci95_lo.notna().all()
    assert (out / "figures" / "error_maps" / "fno_ox_lead1_error.csv").exists()


def test_tables_are_never_built_from_smoke_rows(tmp_path):
    rows = pd.DataFrame([{"run_id": "r", "model": "unet", "split": "smoke_test", "lead_hour": "all", "metric": "rmse",
                          "value": 1.0, "config_hash": "h", "git_commit": "c"}])
    rows.to_csv(tmp_path / "results.csv", index=False)
    with pytest.raises(SystemExit, match="No test rows"):
        load_script("make_tables").main(["--results", str(tmp_path / "results.csv"), "--out", str(tmp_path / "out")])
