#!/usr/bin/env python3
"""Exact leave-one-participant-out influence analysis for primary VPCs.

Each participant is removed in turn and both calibrated primary models are
refitted. Predictor scaling is kept fixed at the complete-sample values so
that changes reflect case deletion rather than a change in parameter scale.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, cmdstan_path, set_cmdstan_path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
FIT_ROOT = Path(os.environ.get("HIFT_LOO_FIT_ROOT", str(ROOT / "results" / "fits" / "loo")))
MODEL_ROOT = Path(os.environ.get("HIFT_STAN_MODEL_ROOT", ROOT / "analysis" / "models"))
SEED = 20260928
LEVELS = np.arange(5, 11)


def quantiles(values: np.ndarray) -> dict[str, float]:
    lo, median, hi = np.quantile(values, [0.025, 0.5, 0.975])
    return {"q025": float(lo), "median": float(median), "q975": float(hi)}


def common_design(data: pd.DataFrame) -> tuple[list[int], np.ndarray, np.ndarray]:
    ids = sorted(data.id.astype(int).unique())
    index = {pid: i + 1 for i, pid in enumerate(ids)}
    participant = data.id.astype(int).map(index).to_numpy(int)
    x_common = np.column_stack([
        (data.workout == "FT").astype(float),
        (data.workout == "EMOM").astype(float),
        data.work_repetitions_z.to_numpy(float),
    ])
    return ids, participant, x_common


def rpe_stan_data(data: pd.DataFrame) -> dict:
    ids, participant, x = common_design(data)
    lookup = {value: i + 1 for i, value in enumerate(LEVELS)}
    return {
        "N": len(data), "J": len(ids), "K": len(LEVELS), "P": 3,
        "y": data.rpe.astype(int).map(lookup).to_numpy(int),
        "participant": participant, "X": x,
        "beta_prior_sd": [0.75, 0.75, 0.50],
        "person_prior_sd": 0.75, "prior_only": 0,
    }


def lactate_stan_data(data: pd.DataFrame, full_log_pre_mean: float,
                      full_log_pre_sd: float) -> dict:
    ids, participant, x_common = common_design(data)
    baseline = (np.log(data.lactate_pre.to_numpy(float)) - full_log_pre_mean) / full_log_pre_sd
    return {
        "N": len(data), "J": len(ids), "P": 4,
        "y_log": np.log(data.lactate_post.to_numpy(float)),
        "participant": participant,
        "X": np.column_stack([x_common, baseline]),
        "alpha_prior_mean": float(np.log(10)), "alpha_prior_sd": 0.20,
        "beta_prior_sd": [0.15, 0.15, 0.10, 0.10],
        "person_prior_sd": 0.15, "residual_prior_sd": 0.20,
        "prior_only": 0,
    }


def fit_one(model: CmdStanModel, data: dict, output_dir: Path, seed: int,
            iter_warmup: int = 1000, iter_sampling: int = 1000):
    output_dir.mkdir(parents=True, exist_ok=True)
    return model.sample(
        data=data, chains=4, parallel_chains=4,
        iter_warmup=iter_warmup, iter_sampling=iter_sampling,
        seed=seed, adapt_delta=0.99, max_treedepth=15,
        inits=0, output_dir=str(output_dir), show_progress=False,
    )


def diagnostics(fit) -> dict[str, float | int]:
    summary = fit.summary()
    method = fit.method_variables()
    return {
        "max_rhat": float(summary.R_hat.max()),
        "min_ess_bulk": float(summary.ESS_bulk.min()),
        "min_ess_tail": float(summary.ESS_tail.min()),
        "divergences": int(np.asarray(method["divergent__"]).sum()),
        "max_treedepth_observed": int(np.asarray(method["treedepth__"]).max()),
    }


def result_row(fit, outcome: str, omitted_id: int) -> dict:
    vpc = quantiles(fit.stan_variable("vpc"))
    sigma_person = quantiles(fit.stan_variable("sigma_person"))
    row = {
        "outcome": outcome, "omitted_id": int(omitted_id),
        **{f"vpc_{key}": value for key, value in vpc.items()},
        **{f"sigma_person_{key}": value for key, value in sigma_person.items()},
        **diagnostics(fit),
    }
    if outcome == "Lactate":
        sigma_residual = quantiles(fit.stan_variable("sigma_residual"))
        row.update({f"sigma_residual_{key}": value
                    for key, value in sigma_residual.items()})
    return row


def write_progress(rows: list[dict]) -> None:
    path = OUT / "tables" / "leave_one_participant_out_vpc.csv"
    pd.DataFrame(rows).sort_values(["outcome", "omitted_id"]).to_csv(path, index=False)


def main() -> None:
    for path in [OUT / "tables", OUT / "summaries"]:
        path.mkdir(parents=True, exist_ok=True)
    FIT_ROOT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("HIFT_CMDSTAN"):
        set_cmdstan_path(os.environ["HIFT_CMDSTAN"])
    else:
        cmdstan_path()

    data = pd.read_csv(ROOT / "data" / "hift_long_complete_case.csv")
    ids = sorted(data.id.astype(int).unique())
    full_log_pre = np.log(data.lactate_pre.to_numpy(float))
    full_log_pre_mean = float(full_log_pre.mean())
    full_log_pre_sd = float(full_log_pre.std(ddof=1))

    rpe_model = CmdStanModel(stan_file=str(MODEL_ROOT / "ordinal_calibrated.stan"))
    lactate_model = CmdStanModel(stan_file=str(MODEL_ROOT / "log_lactate_calibrated.stan"))

    progress_file = OUT / "tables" / "leave_one_participant_out_vpc.csv"
    rows = pd.read_csv(progress_file).to_dict("records") if progress_file.exists() else []
    completed = {(row["outcome"], int(row["omitted_id"])) for row in rows}

    for sequence, omitted_id in enumerate(ids):
        subset = data[data.id.astype(int) != omitted_id].copy()
        specifications = [
            ("RPE", rpe_model, rpe_stan_data(subset)),
            ("Lactate", lactate_model,
             lactate_stan_data(subset, full_log_pre_mean, full_log_pre_sd)),
        ]
        for outcome_index, (outcome, model, stan_data) in enumerate(specifications):
            if (outcome, omitted_id) in completed:
                continue
            fit = fit_one(
                model, stan_data,
                FIT_ROOT / outcome.lower() / f"omit_{omitted_id}",
                SEED + sequence * 10 + outcome_index,
            )
            row = result_row(fit, outcome, omitted_id)
            rows.append(row)
            completed.add((outcome, omitted_id))
            write_progress(rows)
            print(json.dumps({"completed": row}, separators=(",", ":")), flush=True)

    # Refit only numerically marginal cases with longer chains. This makes the
    # final influence comparison depend on well-resolved posterior draws while
    # retaining the shorter, inexpensive default for clearly converged cases.
    initial = pd.DataFrame(rows)
    flagged = initial[
        (initial.max_rhat > 1.01)
        | (initial.min_ess_bulk < 400)
        | (initial.min_ess_tail < 400)
        | (initial.divergences > 0)
    ][["outcome", "omitted_id"]]
    for rerun_index, flagged_row in flagged.reset_index(drop=True).iterrows():
        outcome = str(flagged_row.outcome)
        omitted_id = int(flagged_row.omitted_id)
        subset = data[data.id.astype(int) != omitted_id].copy()
        if outcome == "RPE":
            model, stan_data = rpe_model, rpe_stan_data(subset)
        else:
            model = lactate_model
            stan_data = lactate_stan_data(subset, full_log_pre_mean, full_log_pre_sd)
        fit = fit_one(
            model, stan_data,
            FIT_ROOT / outcome.lower() / f"omit_{omitted_id}_long",
            SEED + 10000 + rerun_index,
            iter_warmup=2000, iter_sampling=3000,
        )
        replacement = result_row(fit, outcome, omitted_id)
        rows = [row for row in rows
                if not (row["outcome"] == outcome
                        and int(row["omitted_id"]) == omitted_id)]
        rows.append(replacement)
        write_progress(rows)
        print(json.dumps({"rerun_completed": replacement},
                         separators=(",", ":")), flush=True)

    table = pd.DataFrame(rows).sort_values(["outcome", "omitted_id"]).reset_index(drop=True)
    full = {
        "RPE": json.loads((OUT / "summaries" / "rpe_calibrated.json").read_text())["vpc"],
        "Lactate": json.loads((OUT / "summaries" / "lactate_calibrated.json").read_text())["vpc"],
    }
    table["full_vpc_median"] = table.outcome.map({k: v["median"] for k, v in full.items()})
    table["delta_vpc_median"] = table.vpc_median - table.full_vpc_median
    table["absolute_delta_vpc_median"] = table.delta_vpc_median.abs()
    table.to_csv(progress_file, index=False)

    result = {"full_sample_vpc": full, "outcomes": {}}
    for outcome in ["RPE", "Lactate"]:
        d = table[table.outcome == outcome]
        most = d.loc[d.absolute_delta_vpc_median.idxmax()]
        result["outcomes"][outcome] = {
            "leave_one_out_vpc_median_range": {
                "min": float(d.vpc_median.min()), "max": float(d.vpc_median.max())
            },
            "leave_one_out_vpc_interval_endpoint_range": {
                "q025_min": float(d.vpc_q025.min()), "q025_max": float(d.vpc_q025.max()),
                "q975_min": float(d.vpc_q975.min()), "q975_max": float(d.vpc_q975.max()),
            },
            "largest_absolute_change": {
                "omitted_id": int(most.omitted_id),
                "vpc_median": float(most.vpc_median),
                "change_from_full": float(most.delta_vpc_median),
            },
            "maximum_rhat": float(d.max_rhat.max()),
            "minimum_bulk_ess": float(d.min_ess_bulk.min()),
            "total_divergences": int(d.divergences.sum()),
        }
    (OUT / "summaries" / "leave_one_participant_out_influence.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
