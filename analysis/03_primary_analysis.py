#!/usr/bin/env python3
"""Physiologically calibrated priors, individual predictions, and targeted PPCs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, cmdstan_path, set_cmdstan_path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
FIT_ROOT = Path(os.environ.get("HIFT_FIT_ROOT", str(OUT / "fits")))
MODEL_DIR = Path(os.environ.get("HIFT_MODEL_DIR", str(ROOT / "analysis" / "models")))
SEED = 20260926
LEVELS = np.arange(5, 11)
WORKOUTS = ["AMRAP", "FT", "EMOM"]


def q(x: np.ndarray) -> dict[str, float]:
    z = np.quantile(x, [0.025, 0.5, 0.975])
    return {"q025": float(z[0]), "median": float(z[1]), "q975": float(z[2])}


def probs(eta: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    # eta: draws, cuts: draws x (K-1)
    cum = 1 / (1 + np.exp(-(cuts - eta[:, None])))
    return np.column_stack([cum[:, 0], np.diff(cum, axis=1), 1 - cum[:, -1]])


def common_data(d: pd.DataFrame):
    ids = sorted(d.id.astype(int).unique().tolist())
    idx = {pid: i + 1 for i, pid in enumerate(ids)}
    participant = d.id.map(idx).to_numpy(int)
    ft = (d.workout == "FT").astype(float).to_numpy()
    emom = (d.workout == "EMOM").astype(float).to_numpy()
    work = d.work_repetitions_z.to_numpy(float)
    return ids, participant, ft, emom, work


def diagnostics(fit) -> dict:
    s = fit.summary()
    mv = fit.method_variables()
    return {
        "max_rhat": float(s.R_hat.max()), "min_ess_bulk": float(s.ESS_bulk.min()),
        "min_ess_tail": float(s.ESS_tail.min()),
        "divergences": int(np.asarray(mv["divergent__"]).sum()),
        "max_treedepth_observed": int(np.asarray(mv["treedepth__"]).max()),
    }


def sample(model, data: dict, name: str, prior: bool = False):
    path = FIT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return model.sample(
        data=data, chains=4, parallel_chains=4,
        iter_warmup=1000 if prior else 2000,
        iter_sampling=1000 if prior else 2000,
        seed=SEED, adapt_delta=0.99, max_treedepth=15,
        inits=0, output_dir=str(path), show_progress=False,
    )


def rpe_data(d: pd.DataFrame, person_sd: float, prior_only: int) -> dict:
    ids, participant, ft, emom, work = common_data(d)
    lookup = {v: i + 1 for i, v in enumerate(LEVELS)}
    return {
        "N": len(d), "J": len(ids), "K": len(LEVELS), "P": 3,
        "y": d.rpe.astype(int).map(lookup).to_numpy(int),
        "participant": participant, "X": np.column_stack([ft, emom, work]),
        "beta_prior_sd": [0.75, 0.75, 0.50],
        "person_prior_sd": person_sd, "prior_only": prior_only,
    }


def lactate_data(d: pd.DataFrame, relaxed: bool, prior_only: int) -> dict:
    ids, participant, ft, emom, work = common_data(d)
    log_pre = np.log(d.lactate_pre)
    baseline = ((log_pre - log_pre.mean()) / log_pre.std(ddof=1)).to_numpy()
    return {
        "N": len(d), "J": len(ids), "P": 4,
        "y_log": np.log(d.lactate_post).to_numpy(), "participant": participant,
        "X": np.column_stack([ft, emom, work, baseline]),
        "alpha_prior_mean": float(np.log(10)),
        "alpha_prior_sd": 0.50 if relaxed else 0.20,
        "beta_prior_sd": [0.40, 0.40, 0.30, 0.30] if relaxed else [0.15, 0.15, 0.10, 0.10],
        "person_prior_sd": 0.40 if relaxed else 0.15,
        "residual_prior_sd": 0.50 if relaxed else 0.20,
        "prior_only": prior_only,
    }


def summarize_rpe(fit, name: str) -> dict:
    beta = fit.stan_variable("beta")
    return {
        "model": name, "sigma_person": q(fit.stan_variable("sigma_person")),
        "vpc": q(fit.stan_variable("vpc")), "diagnostics": diagnostics(fit),
        "coefficients": {
            label: q(beta[:, i]) for i, label in enumerate(["workout_FT", "workout_EMOM", "work_z"])
        },
    }


def summarize_lactate(fit, name: str) -> dict:
    beta = fit.stan_variable("beta")
    return {
        "model": name, "sigma_person_log_mmol": q(fit.stan_variable("sigma_person")),
        "sigma_residual_log_mmol": q(fit.stan_variable("sigma_residual")),
        "vpc": q(fit.stan_variable("vpc")), "diagnostics": diagnostics(fit),
        "alpha": q(fit.stan_variable("alpha")),
        "coefficients": {
            label: q(beta[:, i]) for i, label in enumerate(["workout_FT", "workout_EMOM", "work_z", "baseline_z"])
        },
    }


def prior_checks(rpe_fit, lac_fit, d: pd.DataFrame) -> dict:
    ry = rpe_fit.stan_variable("y_rep").astype(int)
    ly = lac_fit.stan_variable("y_rep_mmol")
    cuts = rpe_fit.stan_variable("cutpoints")
    sig = rpe_fit.stan_variable("sigma_person")
    low = probs(-sig, cuts) @ LEVELS
    high = probs(sig, cuts) @ LEVELS
    return {
        "rpe": {
            "replicated_mean": q(LEVELS[ry - 1].mean(axis=1)),
            "expected_difference_plus_vs_minus_one_person_sd": q(high - low),
            "probability_any_category": {
                str(level): float(np.mean(np.any(ry == i + 1, axis=1)))
                for i, level in enumerate(LEVELS)
            },
        },
        "lactate": {
            "replicated_geometric_mean_mmol": q(np.exp(np.log(ly).mean(axis=1))),
            "replicated_max_mmol": q(ly.max(axis=1)),
            "replicated_min_mmol": q(ly.min(axis=1)),
            "share_of_values_between_0_5_and_30_mmol": q(((ly >= 0.5) & (ly <= 30)).mean(axis=1)),
            "share_of_values_above_30_mmol": q((ly > 30).mean(axis=1)),
        },
        "physiological_reference": {
            "typical_postexercise_range_mmol": "approximately 5–15",
            "broad_plausibility_envelope_mmol": "0.5–30",
            "rationale": "The envelope deliberately exceeds typical HIFT values and includes unusual high-intensity observations near 20–26 mmol/L.",
        },
    }


def individual_predictions(rpe_fit, lac_fit, d: pd.DataFrame) -> dict:
    ids = sorted(d.id.astype(int).unique().tolist())
    r_beta = rpe_fit.stan_variable("beta")
    cuts = rpe_fit.stan_variable("cutpoints")
    r_u = rpe_fit.stan_variable("person_effect")
    l_alpha = lac_fit.stan_variable("alpha")
    l_beta = lac_fit.stan_variable("beta")
    l_u = lac_fit.stan_variable("person_effect")
    l_resid = lac_fit.stan_variable("sigma_residual")
    rng = np.random.default_rng(SEED)
    r_rows, l_rows = [], []
    for j, workout in enumerate(WORKOUTS):
        r_fixed = np.zeros(len(r_beta)) if workout == "AMRAP" else r_beta[:, j - 1]
        l_fixed = l_alpha.copy() if workout == "AMRAP" else l_alpha + l_beta[:, j - 1]
        for k, pid in enumerate(ids):
            pr = probs(r_fixed + r_u[:, k], cuts)
            expected = pr @ LEVELS
            rrow = {"id": pid, "workout": workout, **{f"expected_rpe_{a}": b for a, b in q(expected).items()}}
            for ci, level in enumerate(LEVELS):
                pq = q(pr[:, ci])
                for key, val in pq.items(): rrow[f"p_rpe_{level}_{key}"] = val
            r_rows.append(rrow)

            mu = l_fixed + l_u[:, k]
            median = np.exp(mu)
            arithmetic_mean = np.exp(mu + 0.5 * l_resid**2)
            future = np.exp(rng.normal(mu, l_resid))
            l_rows.append({
                "id": pid, "workout": workout,
                **{f"conditional_median_mmol_{a}": b for a, b in q(median).items()},
                **{f"conditional_mean_mmol_{a}": b for a, b in q(arithmetic_mean).items()},
                **{f"future_session_mmol_{a}": b for a, b in q(future).items()},
            })
    r_df, l_df = pd.DataFrame(r_rows), pd.DataFrame(l_rows)
    r_df.to_csv(OUT / "tables" / "individual_rpe_predictions.csv", index=False)
    l_df.to_csv(OUT / "tables" / "individual_lactate_predictions.csv", index=False)

    r_sigma = rpe_fit.stan_variable("sigma_person")
    l_sigma = lac_fit.stan_variable("sigma_person")
    hypothetical = {}
    for j, workout in enumerate(WORKOUTS):
        r_fixed = np.zeros(len(r_beta)) if workout == "AMRAP" else r_beta[:, j - 1]
        r_low = probs(r_fixed - r_sigma, cuts) @ LEVELS
        r_high = probs(r_fixed + r_sigma, cuts) @ LEVELS
        l_fixed = l_alpha.copy() if workout == "AMRAP" else l_alpha + l_beta[:, j - 1]
        l_low, l_high = np.exp(l_fixed - l_sigma), np.exp(l_fixed + l_sigma)
        hypothetical[workout] = {
            "rpe_expected_low_minus_one_sd": q(r_low),
            "rpe_expected_high_plus_one_sd": q(r_high),
            "rpe_expected_difference": q(r_high - r_low),
            "lactate_median_low_minus_one_sd": q(l_low),
            "lactate_median_high_plus_one_sd": q(l_high),
            "lactate_high_to_low_ratio": q(l_high / l_low),
        }
    return {"hypothetical_participants": hypothetical}


def posterior_predictive(rpe_fit, lac_fit, d: pd.DataFrame) -> dict:
    ry = rpe_fit.stan_variable("y_rep").astype(int)
    ly = lac_fit.stan_variable("y_rep_mmol")
    result = {"rpe_by_workout": {}, "lactate_by_workout": {}}
    for workout in WORKOUTS:
        mask = (d.workout == workout).to_numpy()
        result["rpe_by_workout"][workout] = {
            str(level): {
                "observed": int((d.loc[mask, "rpe"] == level).sum()),
                "replicated_count": q((ry[:, mask] == i + 1).sum(axis=1)),
            } for i, level in enumerate(LEVELS)
        }
        obs = d.loc[mask, "lactate_post"].to_numpy()
        rep = ly[:, mask]
        result["lactate_by_workout"][workout] = {
            "observed_mean": float(obs.mean()), "replicated_mean": q(rep.mean(axis=1)),
            "observed_sd": float(obs.std(ddof=1)), "replicated_sd": q(rep.std(axis=1, ddof=1)),
            "observed_min": float(obs.min()), "replicated_min": q(rep.min(axis=1)),
            "observed_max": float(obs.max()), "replicated_max": q(rep.max(axis=1)),
        }
    obs_rpe_between = d.groupby("id").rpe.mean().std(ddof=1)
    obs_lac_between = d.groupby("id").lactate_post.mean().std(ddof=1)
    rmat = ry.reshape(ry.shape[0], len(d.id.unique()), 3)
    lmat = ly.reshape(ly.shape[0], len(d.id.unique()), 3)
    result["between_participant"] = {
        "rpe_sd_of_participant_means_observed": float(obs_rpe_between),
        "rpe_sd_of_participant_means_replicated": q(LEVELS[rmat - 1].mean(axis=2).std(axis=1, ddof=1)),
        "lactate_sd_of_participant_means_observed": float(obs_lac_between),
        "lactate_sd_of_participant_means_replicated": q(lmat.mean(axis=2).std(axis=1, ddof=1)),
    }
    return result


def main() -> None:
    for x in [OUT / "summaries", OUT / "tables", OUT / "checks"]:
        x.mkdir(parents=True, exist_ok=True)
    if os.environ.get("HIFT_CMDSTAN"):
        set_cmdstan_path(os.environ["HIFT_CMDSTAN"])
    else:
        cmdstan_path()
    d = pd.read_csv(ROOT / "data" / "hift_long_complete_case.csv")
    r_model = CmdStanModel(stan_file=str(MODEL_DIR / "ordinal_calibrated.stan"))
    l_model = CmdStanModel(stan_file=str(MODEL_DIR / "log_lactate_calibrated.stan"))

    r_prior = sample(r_model, rpe_data(d, 0.75, 1), "rpe_calibrated_prior", True)
    l_prior = sample(l_model, lactate_data(d, False, 1), "lactate_calibrated_prior", True)
    prior = prior_checks(r_prior, l_prior, d)
    (OUT / "checks" / "calibrated_prior_predictive.json").write_text(json.dumps(prior, indent=2) + "\n")

    r_fit = sample(r_model, rpe_data(d, 0.75, 0), "rpe_calibrated")
    r_relaxed = sample(r_model, rpe_data(d, 1.25, 0), "rpe_calibrated_relaxed")
    l_fit = sample(l_model, lactate_data(d, False, 0), "lactate_calibrated")
    l_relaxed = sample(l_model, lactate_data(d, True, 0), "lactate_calibrated_relaxed")

    results = {
        "rpe_calibrated": summarize_rpe(r_fit, "rpe_calibrated"),
        "rpe_calibrated_relaxed": summarize_rpe(r_relaxed, "rpe_calibrated_relaxed"),
        "lactate_calibrated": summarize_lactate(l_fit, "lactate_calibrated"),
        "lactate_calibrated_relaxed": summarize_lactate(l_relaxed, "lactate_calibrated_relaxed"),
    }
    for name, obj in results.items():
        (OUT / "summaries" / f"{name}.json").write_text(json.dumps(obj, indent=2) + "\n")

    individual = individual_predictions(r_fit, l_fit, d)
    (OUT / "summaries" / "hypothetical_participants.json").write_text(json.dumps(individual, indent=2) + "\n")
    ppc = posterior_predictive(r_fit, l_fit, d)
    (OUT / "checks" / "targeted_posterior_predictive_checks.json").write_text(json.dumps(ppc, indent=2) + "\n")

    original_rpe = json.loads((ROOT / "results" / "original" / "summaries" / "rpe_ordinal_complete.json").read_text())
    original_lac = json.loads((ROOT / "results" / "original" / "summaries" / "lactate_gaussian_complete.json").read_text())
    compare = pd.DataFrame([
        {"outcome": "RPE", "prior": "original half-Student-t(3,0,1)", **original_rpe["vpc"]},
        {"outcome": "RPE", "prior": "calibrated half-Normal(0,0.75)", **results["rpe_calibrated"]["vpc"]},
        {"outcome": "RPE", "prior": "relaxed half-Normal(0,1.25)", **results["rpe_calibrated_relaxed"]["vpc"]},
        {"outcome": "Lactate", "prior": "original standardized half-Student-t(3,0,1)", **original_lac["vpc"]},
        {"outcome": "Lactate", "prior": "calibrated log-scale half-Normal(0,0.15/0.20)", **results["lactate_calibrated"]["vpc"]},
        {"outcome": "Lactate", "prior": "relaxed log-scale half-Normal(0,0.40/0.50)", **results["lactate_calibrated_relaxed"]["vpc"]},
    ])
    compare.to_csv(OUT / "tables" / "prior_sensitivity_comparison.csv", index=False)
    print(json.dumps({"prior_predictive": prior, "posterior": results}, indent=2))


if __name__ == "__main__":
    main()
