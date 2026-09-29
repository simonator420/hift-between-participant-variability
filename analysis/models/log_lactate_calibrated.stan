data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> P;
  vector[N] y_log;
  array[N] int<lower=1, upper=J> participant;
  matrix[N, P] X;
  real alpha_prior_mean;
  real<lower=0> alpha_prior_sd;
  vector<lower=0>[P] beta_prior_sd;
  real<lower=0> person_prior_sd;
  real<lower=0> residual_prior_sd;
  int<lower=0, upper=1> prior_only;
}
parameters {
  real alpha;
  vector[P] beta;
  real<lower=0> sigma_person;
  real<lower=0> sigma_residual;
  vector[J] z_person;
}
transformed parameters {
  vector[J] person_effect = sigma_person * z_person;
  vector[N] mu_log = alpha + X * beta;
  for (n in 1:N) mu_log[n] += person_effect[participant[n]];
}
model {
  alpha ~ normal(alpha_prior_mean, alpha_prior_sd);
  beta ~ normal(0, beta_prior_sd);
  sigma_person ~ normal(0, person_prior_sd);
  sigma_residual ~ normal(0, residual_prior_sd);
  z_person ~ std_normal();
  if (!prior_only) y_log ~ normal(mu_log, sigma_residual);
}
generated quantities {
  real vpc = square(sigma_person) /
    (square(sigma_person) + square(sigma_residual));
  vector[N] y_rep_log;
  vector[N] y_rep_mmol;
  for (n in 1:N) {
    y_rep_log[n] = normal_rng(mu_log[n], sigma_residual);
    y_rep_mmol[n] = exp(y_rep_log[n]);
  }
}
