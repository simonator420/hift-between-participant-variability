data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=2> K;
  int<lower=1> P;
  array[N] int<lower=1, upper=K> y;
  array[N] int<lower=1, upper=J> participant;
  matrix[N, P] X;
  vector<lower=0>[P] beta_prior_sd;
  real<lower=0> person_prior_sd;
  real<lower=0> prior_power;
  int<lower=0, upper=1> prior_only;
}
parameters {
  vector[P] beta;
  ordered[K - 1] cutpoints;
  real<lower=1e-6> sigma_person;
  vector[J] z_person;
}
transformed parameters {
  vector[J] person_effect = sigma_person * z_person;
  vector[N] eta = X * beta;
  for (n in 1:N) eta[n] += person_effect[participant[n]];
}
model {
  target += prior_power * normal_lpdf(beta | rep_vector(0, P), beta_prior_sd);
  target += prior_power * normal_lpdf(cutpoints | 0, 2.5);
  target += prior_power * normal_lpdf(sigma_person | 0, person_prior_sd);
  target += prior_power * normal_lpdf(z_person | 0, 1);
  if (!prior_only) y ~ ordered_logistic(eta, cutpoints);
}
generated quantities {
  real vpc = square(sigma_person) /
    (square(sigma_person) + square(pi()) / 3);
}
