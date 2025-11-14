
data {
  int<lower=1> N;
  array[N] int<lower=0> y;                 // was: int<lower=0> y[N];
  vector<lower=0>[N] py;
  int<lower=1> J_country;
  int<lower=1> R_region;
  array[N] int<lower=1> country_id;        // was: int<lower=1> country_id[N];
  array[J_country] int<lower=1> region_id_country; // was: int<lower=1> region_id_country[J_country];
  int<lower=1> K_age;
  matrix[N, K_age] B_age;
  array[N] int<lower=0, upper=1> male;     // was: int<lower=0,upper=1> male[N];
  vector[N] year_c;
}
parameters {
  real alpha;
  vector[R_region] z_region;
  real<lower=0> sigma_region;
  vector[J_country] z_country;
  real<lower=0> sigma_country;
  vector[K_age] beta_age;
  real beta_male;
  real beta_year;
  real<lower=0> phi;
}
transformed parameters {
  vector[R_region] region_eff = z_region * sigma_region;
  vector[J_country] country_eff;
  for (j in 1:J_country)
    country_eff[j] = region_eff[ region_id_country[j] ] + z_country[j] * sigma_country;
}
model {
  // Priors
  alpha        ~ normal(0, 2);
  z_region     ~ normal(0, 1);
  sigma_region ~ exponential(1);
  z_country    ~ normal(0, 1);
  sigma_country~ exponential(1);
  beta_age     ~ normal(0, 1);
  beta_male    ~ normal(0, 1);
  beta_year    ~ normal(0, 0.5);
  phi          ~ exponential(1);

  // Likelihood
  for (n in 1:N) {
    real eta = alpha
               + country_eff[country_id[n]]
               + B_age[n] * beta_age
               + beta_male * male[n]
               + beta_year * year_c[n]
               + log(py[n]);
    y[n] ~ neg_binomial_2_log(eta, phi);
  }
}
generated quantities {
  vector[N] log_lambda;
  int y_rep[N];
  for (n in 1:N) {
    real eta = alpha
               + country_eff[country_id[n]]
               + B_age[n] * beta_age
               + beta_male * male[n]
               + beta_year * year_c[n]
               + log(py[n]);
    log_lambda[n] = eta;
    y_rep[n] = neg_binomial_2_log_rng(eta, phi);
  }
}
