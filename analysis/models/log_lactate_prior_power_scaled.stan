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
  real<lower=0> prior_power;
  int<lower=0, upper=1> prior_only;
}
parameters {
  real alpha;
  vector[P] beta;
  real<lower=1e-6> sigma_person;
  real<lower=1e-6> sigma_residual;
  vector[J] z_person;
}
transformed parameters {
  vector[J] person_effect = sigma_person * z_person;
  vector[N] mu_log = alpha + X * beta;
  for (n in 1:N) mu_log[n] += person_effect[participant[n]];
}
model {
  target += prior_power * normal_lpdf(alpha | alpha_prior_mean, alpha_prior_sd);
  target += prior_power * normal_lpdf(beta | rep_vector(0, P), beta_prior_sd);
  target += prior_power * normal_lpdf(sigma_person | 0, person_prior_sd);
  target += prior_power * normal_lpdf(sigma_residual | 0, residual_prior_sd);
  target += prior_power * normal_lpdf(z_person | 0, 1);
  if (!prior_only) y_log ~ normal(mu_log, sigma_residual);
}
generated quantities {
  real vpc = square(sigma_person) /
    (square(sigma_person) + square(sigma_residual));
}
