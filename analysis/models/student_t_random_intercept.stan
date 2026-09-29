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
  real<lower=0> nu_minus_two;
  vector[J] z_person;
}
transformed parameters {
  real<lower=2> nu = 2 + nu_minus_two;
  vector[J] person_effect = sigma_person * z_person;
  vector[N] mu = X * beta;
  for (n in 1:N) mu[n] += person_effect[participant[n]];
}
model {
  beta ~ normal(0, prior_scale);
  sigma_person ~ student_t(3, 0, prior_scale);
  sigma_residual ~ student_t(3, 0, prior_scale);
  nu_minus_two ~ exponential(0.1);
  z_person ~ std_normal();
  if (!prior_only) y ~ student_t(nu, mu, sigma_residual);
}
generated quantities {
  real residual_variance = square(sigma_residual) * nu / (nu - 2);
  real vpc = square(sigma_person) /
    (square(sigma_person) + residual_variance);
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    log_lik[n] = student_t_lpdf(y[n] | nu, mu[n], sigma_residual);
    y_rep[n] = student_t_rng(nu, mu[n], sigma_residual);
  }
}
