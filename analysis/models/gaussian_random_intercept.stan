data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> P;
  vector[N] y;
  array[N] int<lower=1, upper=J> participant;
  matrix[N, P] X;
  real<lower=0> prior_scale;
  int<lower=0, upper=1> prior_only;
}
parameters {
  vector[P] beta;
  real<lower=0> sigma_person;
  real<lower=0> sigma_residual;
  vector[J] z_person;
}
transformed parameters {
  vector[J] person_effect = sigma_person * z_person;
  vector[N] mu = X * beta;
  for (n in 1:N) mu[n] += person_effect[participant[n]];
}
model {
  beta ~ normal(0, prior_scale);
  sigma_person ~ student_t(3, 0, prior_scale);
  sigma_residual ~ student_t(3, 0, prior_scale);
  z_person ~ std_normal();
  if (!prior_only) y ~ normal(mu, sigma_residual);
}
generated quantities {
  real vpc = square(sigma_person) /
    (square(sigma_person) + square(sigma_residual));
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma_residual);
    y_rep[n] = normal_rng(mu[n], sigma_residual);
  }
}
