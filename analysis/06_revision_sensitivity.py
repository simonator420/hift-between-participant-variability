#!/usr/bin/env python3
"""Reviewer-requested prior sensitivity and observed-scale RPE VPC.

This script contains only numerical analysis. It does not create publication
figures. Generated chain files are written below ``results/revision/fits`` and
are excluded from version control; concise CSV/JSON summaries are retained.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, cmdstan_path, from_csv, set_cmdstan_path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hift_long_complete_case.csv"
MODEL_DIR = ROOT / "analysis" / "models"
OUT = ROOT / "results" / "revision"
FIT_ROOT = OUT / "fits"
POWERS = [0.50, 0.75, 1.00, 1.25, 1.50]
LEVELS = np.arange(5, 11, dtype=float)
FORMATS = ["AMRAP", "FT", "EMOM"]
SEED = 20261114


def quantiles(values: np.ndarray) -> dict[str, float]:
    q025, median, q975 = np.quantile(np.asarray(values, dtype=float), [0.025, 0.5, 0.975])
    return {"q025": float(q025), "median": float(median), "q975": float(q975)}


def diagnostics(fit) -> dict[str, float | int]:
    summary = fit.summary()
    method = fit.method_variables()
    return {
        "max_rhat": float(np.nanmax(np.asarray(summary.R_hat, dtype=float))),
        "min_ess_bulk": float(np.nanmin(np.asarray(summary.ESS_bulk, dtype=float))),
        "divergences": int(np.asarray(method["divergent__"]).sum()),
    }


def configure_cmdstan() -> None:
    if os.environ.get("HIFT_CMDSTAN"):
        set_cmdstan_path(os.environ["HIFT_CMDSTAN"])
    else:
        cmdstan_path()


def common_data(data: pd.DataFrame):
    ids = sorted(data.id.astype(int).unique())
    participant = data.id.astype(int).map({pid: i + 1 for i, pid in enumerate(ids)}).to_numpy(int)
    ft = (data.workout == "FT").astype(float).to_numpy()
    emom = (data.workout == "EMOM").astype(float).to_numpy()
    work = data.work_repetitions_z.to_numpy(float)
    return ids, participant, ft, emom, work


def rpe_data(data: pd.DataFrame, specification: str, power_name: str,
             power: float, prior_only: int) -> dict:
    ids, participant, ft, emom, work = common_data(data)
    lookup = {int(value): i + 1 for i, value in enumerate(LEVELS)}
    stan_data = {
        "N": len(data), "J": len(ids), "K": len(LEVELS), "P": 3,
        "y": data.rpe.astype(int).map(lookup).to_numpy(int),
        "participant": participant,
        "X": np.column_stack([ft, emom, work]),
        "beta_prior_sd": [0.75, 0.75, 0.50],
        "person_prior_sd": 1.25 if specification == "relaxed" else 0.75,
        "prior_only": prior_only,
    }
    stan_data[power_name] = power
    return stan_data


def lactate_data(data: pd.DataFrame, specification: str, power_name: str,
                 power: float, prior_only: int) -> dict:
    ids, participant, ft, emom, work = common_data(data)
    log_pre = np.log(data.lactate_pre)
    baseline = ((log_pre - log_pre.mean()) / log_pre.std(ddof=1)).to_numpy()
    relaxed = specification == "relaxed"
    stan_data = {
        "N": len(data), "J": len(ids), "P": 4,
        "y_log": np.log(data.lactate_post).to_numpy(),
        "participant": participant,
        "X": np.column_stack([ft, emom, work, baseline]),
        "alpha_prior_mean": float(np.log(10)),
        "alpha_prior_sd": 0.50 if relaxed else 0.20,
        "beta_prior_sd": ([0.40, 0.40, 0.30, 0.30] if relaxed
                           else [0.15, 0.15, 0.10, 0.10]),
        "person_prior_sd": 0.40 if relaxed else 0.15,
        "residual_prior_sd": 0.50 if relaxed else 0.20,
        "prior_only": prior_only,
    }
    stan_data[power_name] = power
    return stan_data


def parameter_draws(fit, outcome: str) -> dict[str, np.ndarray]:
    beta = fit.stan_variable("beta")
    if outcome == "RPE":
        return {
            "sigma_person": fit.stan_variable("sigma_person"),
            "VPC": fit.stan_variable("vpc"),
            "FT_effect": beta[:, 0], "EMOM_effect": beta[:, 1],
            "work_effect": beta[:, 2],
        }
    return {
        "sigma_person": fit.stan_variable("sigma_person"),
        "sigma_residual": fit.stan_variable("sigma_residual"),
        "VPC": fit.stan_variable("vpc"),
        "log_intercept": fit.stan_variable("alpha"),
        "FT_effect": beta[:, 0], "EMOM_effect": beta[:, 1],
        "work_effect": beta[:, 2], "baseline_effect": beta[:, 3],
    }


def sample(model: CmdStanModel, stan_data: dict, name: str, quick: bool):
    output_dir = FIT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    iterations = 500 if quick else 2000
    return model.sample(
        data=stan_data, chains=4, parallel_chains=4,
        iter_warmup=iterations, iter_sampling=iterations,
        seed=SEED, adapt_delta=0.99, max_treedepth=15,
        inits=0, output_dir=str(output_dir), show_progress=False,
    )


def wide_summary(outcome: str, specification: str, power_name: str,
                 power: float, fit) -> dict:
    row = {
        "outcome": outcome, "specification": specification,
        power_name: power, **diagnostics(fit),
    }
    for parameter, values in parameter_draws(fit, outcome).items():
        for key, value in quantiles(values).items():
            row[f"{parameter.lower()}_{key}"] = value
    return row


def likelihood_contraction_rows(prior: dict[str, np.ndarray], posterior: dict[str, np.ndarray],
                                outcome: str, specification: str) -> list[dict]:
    labels = {
        "FT_effect": "FT effect", "EMOM_effect": "EMOM effect",
        "work_effect": "Work effect", "baseline_effect": "Baseline effect",
        "log_intercept": "log intercept",
    }
    rows = []
    for parameter in prior:
        prior_q = quantiles(prior[parameter])
        post_q = quantiles(posterior[parameter])
        prior_width = prior_q["q975"] - prior_q["q025"]
        post_width = post_q["q975"] - post_q["q025"]
        prior_sd = float(np.std(prior[parameter], ddof=1))
        posterior_sd = float(np.std(posterior[parameter], ddof=1))
        row = {
            "outcome": outcome, "specification": specification,
            "parameter": labels.get(parameter, parameter),
            "prior_q025": prior_q["q025"], "prior_median": prior_q["median"],
            "prior_q975": prior_q["q975"],
            "posterior_q025": post_q["q025"], "posterior_median": post_q["median"],
            "posterior_q975": post_q["q975"],
            "prior_sd": prior_sd, "posterior_sd": posterior_sd,
            "sd_ratio_posterior_to_prior": posterior_sd / prior_sd,
            "width_ratio_posterior_to_prior": post_width / prior_width,
        }
        rows.append(row)
    return rows


def prior_contraction_rows(prior: dict[str, np.ndarray], posterior: dict[str, np.ndarray],
                           outcome: str, specification: str, power: float) -> list[dict]:
    rows = []
    for parameter in prior:
        prior_q = quantiles(prior[parameter])
        post_q = quantiles(posterior[parameter])
        prior_width = prior_q["q975"] - prior_q["q025"]
        post_width = post_q["q975"] - post_q["q025"]
        rows.append({
            "outcome": outcome, "specification": specification,
            "prior_power": power, "parameter": parameter,
            "prior_median": prior_q["median"], "posterior_median": post_q["median"],
            "prior_width": prior_width, "posterior_width": post_width,
            "width_ratio_posterior_to_prior": post_width / prior_width,
        })
    return rows


def run_likelihood_scaling(data: pd.DataFrame, quick: bool) -> None:
    rpe_model = CmdStanModel(stan_file=str(MODEL_DIR / "ordinal_power_scaled.stan"))
    lactate_model = CmdStanModel(stan_file=str(MODEL_DIR / "log_lactate_power_scaled.stan"))
    rows, contractions = [], []
    for specification in ["calibrated", "relaxed"]:
        for outcome, model, data_function in [
            ("RPE", rpe_model, rpe_data),
            ("Lactate", lactate_model, lactate_data),
        ]:
            prior_fit = sample(
                model,
                data_function(data, specification, "likelihood_power", 0.0, 1),
                f"likelihood/{outcome.lower()}_{specification}_prior",
                quick,
            )
            prior = parameter_draws(prior_fit, outcome)
            for power in POWERS:
                fit = sample(
                    model,
                    data_function(data, specification, "likelihood_power", power, 0),
                    f"likelihood/{outcome.lower()}_{specification}_power_{power:.2f}",
                    quick,
                )
                rows.append(wide_summary(outcome, specification, "likelihood_power", power, fit))
                if power == 1.0:
                    contractions.extend(likelihood_contraction_rows(
                        prior, parameter_draws(fit, outcome), outcome, specification
                    ))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "tables" / "likelihood_power_scaling.csv", index=False)
    pd.DataFrame(contractions).to_csv(
        OUT / "tables" / "prior_posterior_contraction.csv", index=False
    )
    primary_rows = []
    parameter_labels = {
        "sigma_person": "sigma_person", "sigma_residual": "sigma_residual",
        "vpc": "VPC", "log_intercept": "log_intercept",
        "ft_effect": "FT_effect", "emom_effect": "EMOM_effect",
        "work_effect": "work_effect", "baseline_effect": "baseline_effect",
    }
    for _, row in summary[
        (summary.specification == "relaxed") & (summary.likelihood_power == 1.0)
    ].iterrows():
        for key, label in parameter_labels.items():
            if pd.isna(row.get(f"{key}_median")):
                continue
            primary_rows.append({
                "outcome": row.outcome, "parameter": label,
                "q025": row[f"{key}_q025"], "median": row[f"{key}_median"],
                "q975": row[f"{key}_q975"],
            })
    pd.DataFrame(primary_rows).to_csv(
        OUT / "tables" / "relaxed_primary_summary.csv", index=False
    )
    relaxed_diagnostics = {}
    for _, row in summary[
        (summary.specification == "relaxed") & (summary.likelihood_power == 1.0)
    ].iterrows():
        relaxed_diagnostics[row.outcome] = {
            "max_rhat": row.max_rhat,
            "min_ess_bulk": row.min_ess_bulk,
            "divergences": int(row.divergences),
        }
    relaxed_diagnostics.update({
        "n_participants": 25, "n_observations": 75, "likelihood_power": 1.0,
    })
    (OUT / "checks" / "relaxed_model_diagnostics.json").write_text(
        json.dumps(relaxed_diagnostics, indent=2) + "\n"
    )


def run_prior_scaling(data: pd.DataFrame, quick: bool) -> None:
    rpe_model = CmdStanModel(stan_file=str(MODEL_DIR / "ordinal_prior_power_scaled.stan"))
    lactate_model = CmdStanModel(stan_file=str(MODEL_DIR / "log_lactate_prior_power_scaled.stan"))
    rows, contractions = [], []
    for specification in ["calibrated", "relaxed"]:
        for outcome, model, data_function in [
            ("RPE", rpe_model, rpe_data),
            ("Lactate", lactate_model, lactate_data),
        ]:
            for power in POWERS:
                base = f"prior/{outcome.lower()}_{specification}_power_{power:.2f}"
                prior_fit = sample(
                    model, data_function(data, specification, "prior_power", power, 1),
                    base + "_prior", quick,
                )
                post_fit = sample(
                    model, data_function(data, specification, "prior_power", power, 0),
                    base + "_posterior", quick,
                )
                prior, posterior = parameter_draws(prior_fit, outcome), parameter_draws(post_fit, outcome)
                for distribution, fit, draws in [
                    ("prior", prior_fit, prior), ("posterior", post_fit, posterior)
                ]:
                    diagnostic = diagnostics(fit)
                    for parameter, values in draws.items():
                        rows.append({
                            "outcome": outcome, "specification": specification,
                            "prior_power": power, "distribution": distribution,
                            "parameter": parameter, **quantiles(values), **diagnostic,
                        })
                contractions.extend(prior_contraction_rows(
                    prior, posterior, outcome, specification, power
                ))
    pd.DataFrame(rows).to_csv(OUT / "tables" / "prior_power_scaling.csv", index=False)
    pd.DataFrame(contractions).to_csv(
        OUT / "tables" / "prior_power_contraction.csv", index=False
    )


def latest_fit(directory: Path, prefix: str):
    groups: dict[str, list[Path]] = {}
    for path in directory.glob(f"{prefix}-*.csv"):
        match = re.search(r"-(\d{14})_(\d)\.csv$", path.name)
        if match:
            groups.setdefault(match.group(1), []).append(path)
    complete = {stamp: paths for stamp, paths in groups.items() if len(paths) == 4}
    if not complete:
        raise FileNotFoundError(f"No complete four-chain fit in {directory}")
    files = sorted(complete[max(complete)], key=lambda p: int(re.search(r"_(\d)\.csv$", p.name).group(1)))
    return from_csv([str(path) for path in files])


def ordinal_probabilities(eta: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    z = np.clip(cutpoints[:, :, None] - eta[:, None, :], -40, 40)
    cdf = 1.0 / (1.0 + np.exp(-z))
    probabilities = np.concatenate(
        [cdf[:, 0:1, :], np.diff(cdf, axis=1), 1.0 - cdf[:, -1:, :]], axis=1
    )
    return probabilities.transpose(0, 2, 1)


def variance_components(expected: np.ndarray, conditional_variance: np.ndarray):
    between = np.var(expected, axis=1, ddof=0)
    within = conditional_variance.mean(axis=1)
    total = between + within
    return between, within, total, between / total


def run_observed_scale_vpc(data: pd.DataFrame) -> None:
    rows = []
    ids = sorted(data.id.astype(int).unique())
    for specification in ["calibrated", "relaxed"]:
        fit = latest_fit(
            FIT_ROOT / "likelihood" / f"rpe_{specification}_power_1.00",
            "ordinal_power_scaled",
        )
        beta = fit.stan_variable("beta")
        cutpoints = fit.stan_variable("cutpoints")
        person = fit.stan_variable("person_effect")
        format_components = {}
        for workout in FORMATS:
            fixed = np.zeros(len(beta)) if workout == "AMRAP" else beta[:, 0 if workout == "FT" else 1]
            eta = fixed[:, None] + person
            probabilities = ordinal_probabilities(eta, cutpoints)
            expected = probabilities @ LEVELS
            conditional_variance = probabilities @ (LEVELS ** 2) - expected ** 2
            format_components[workout] = variance_components(expected, conditional_variance)
        pooled_between = np.mean([format_components[x][0] for x in FORMATS], axis=0)
        pooled_within = np.mean([format_components[x][1] for x in FORMATS], axis=0)
        pooled_total = pooled_between + pooled_within
        format_components["Pooled across formats"] = (
            pooled_between, pooled_within, pooled_total, pooled_between / pooled_total
        )
        for workout, (between, within, total, vpc) in format_components.items():
            row = {"specification": specification, "scenario": "common_work", "format": workout}
            for key, value in quantiles(vpc).items():
                row[f"observed_scale_vpc_{key}"] = value
            row.update({
                "between_variance_median": float(np.median(between)),
                "within_variance_median": float(np.median(within)),
                "total_variance_median": float(np.median(total)),
            })
            rows.append(row)

        # Secondary descriptive calculation at the observed design. The
        # common-work result above remains the primary estimand because it does
        # not attribute observed workload differences to participants.
        participant_index = data.id.astype(int).map(
            {pid: i for i, pid in enumerate(ids)}
        ).to_numpy()
        ft = (data.workout == "FT").astype(float).to_numpy()
        emom = (data.workout == "EMOM").astype(float).to_numpy()
        work = data.work_repetitions_z.to_numpy(float)
        eta = (
            beta[:, 0, None] * ft
            + beta[:, 1, None] * emom
            + beta[:, 2, None] * work
            + person[:, participant_index]
        )
        probabilities = ordinal_probabilities(eta, cutpoints)
        expected = probabilities @ LEVELS
        conditional_variance = probabilities @ (LEVELS ** 2) - expected ** 2
        expected_by_person = np.column_stack([
            expected[:, data.id.astype(int).to_numpy() == pid].mean(axis=1)
            for pid in ids
        ])
        variance_by_person = np.column_stack([
            conditional_variance[:, data.id.astype(int).to_numpy() == pid].mean(axis=1)
            for pid in ids
        ])
        between, within, total, vpc = variance_components(
            expected_by_person, variance_by_person
        )
        row = {
            "specification": specification, "scenario": "observed_design",
            "format": "Pooled across formats",
        }
        for key, value in quantiles(vpc).items():
            row[f"observed_scale_vpc_{key}"] = value
        row.update({
            "between_variance_median": float(np.median(between)),
            "within_variance_median": float(np.median(within)),
            "total_variance_median": float(np.median(total)),
        })
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "tables" / "observed_scale_vpc.csv", index=False)
    (OUT / "checks" / "observed_scale_vpc_definition.json").write_text(json.dumps({
        "definition": "variance of posterior expected participant means divided by total posterior predictive variance on RPE categories 5-10",
        "participant_variance_ddof": 0,
        "common_work": "work_z=0; format fixed effects retained; pooled estimate averages between and within components before taking the ratio",
        "actual_design": "descriptive secondary calculation using observed workout and work_z; not the primary estimand",
        "n_participants": len(ids), "n_formats": len(FORMATS),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis", choices=["all", "likelihood", "prior", "observed-scale"],
        default="all",
    )
    parser.add_argument("--quick", action="store_true", help="Use 500 warmup and sampling draws")
    args = parser.parse_args()
    for directory in [OUT / "tables", OUT / "checks", FIT_ROOT]:
        directory.mkdir(parents=True, exist_ok=True)
    configure_cmdstan()
    data = pd.read_csv(DATA)
    if args.analysis in ["all", "likelihood"]:
        run_likelihood_scaling(data, args.quick)
    if args.analysis in ["all", "prior"]:
        run_prior_scaling(data, args.quick)
    if args.analysis in ["all", "observed-scale"]:
        run_observed_scale_vpc(data)
    print(f"Wrote numerical revision outputs to {OUT}")


if __name__ == "__main__":
    main()
