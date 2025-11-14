functions {
  real partial_sum(array[] int y_slice,
                   int start, int end,
                   vector py,
                   array[] int country_id,
                   matrix B_age,
                   array[] int male,
                   vector year_c,
                   real alpha,
                   vector country_eff,
                   vector beta_age,
                   real beta_male,
                   real beta_year,
                   real phi) {
    real lp = 0;
    for (n in start:end) {
      int k = n - start + 1;
      real eta = alpha
                 + country_eff[country_id[n]]
                 + B_age[n] * beta_age
                 + beta_male * male[n]
                 + beta_year * year_c[n]
                 + log(py[n]);
      lp += neg_binomial_2_log_lpmf(y_slice[k] | eta, phi);
    }
    return lp;
  }
}
data {
  int<lower=1> N;
  array[N] int<lower=0> y;
  // Exposure must be strictly positive to avoid log(0) -> -inf
  vector<lower=1e-12>[N] py;
  int<lower=1> J_country;
  int<lower=1> R_region;
  array[N] int<lower=1> country_id;
  array[J_country] int<lower=1> region_id_country;
  int<lower=1> K_age;
  matrix[N, K_age] B_age;
  array[N] int<lower=0, upper=1> male;
  vector[N] year_c;
  int<lower=1> grainsize;
  int<lower=0, upper=1> prior_only;        // if 1, skip likelihood for prior predictive draws
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
  // Precision/overdispersion must be strictly positive; avoid 0 for stability
  real<lower=1e-8> phi;
}
transformed parameters {
  vector[R_region] region_eff = z_region * sigma_region;
  vector[J_country] country_eff;
  for (j in 1:J_country)
    country_eff[j] = region_eff[ region_id_country[j] ] + z_country[j] * sigma_country;
}
model {
  // Priors
  alpha         ~ normal(0, 2);
  z_region      ~ normal(0, 1);
  sigma_region  ~ normal(0, 0.5);      // half-normal via <lower=0>
  z_country     ~ normal(0, 1);
  sigma_country ~ normal(0, 0.5);      // half-normal via <lower=0>
  beta_age      ~ normal(0, 1);
  beta_male     ~ normal(0, 1);
  beta_year     ~ normal(0, 0.2);      // per-decade scaling in data
  phi           ~ gamma(2, 0.1);

  // Likelihood
  if (prior_only == 0) {
    target += reduce_sum(partial_sum, y, grainsize,
                         py, country_id, B_age, male, year_c,
                         alpha, country_eff, beta_age, beta_male, beta_year, phi);
  }
}
generated quantities {
  vector[N] log_lambda;
  for (n in 1:N) {
    real eta = alpha
               + country_eff[country_id[n]]
               + B_age[n] * beta_age
               + beta_male * male[n]
               + beta_year * year_c[n]
               + log(py[n]);
    log_lambda[n] = eta;
  }
}
