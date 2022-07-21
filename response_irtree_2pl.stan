data {
  int<lower=1> I;               // # items
  int<lower=1> J;               // # persons
  int<lower=1> N;               // # total responses
  int<lower=1> C;               // # categories
  int<lower=1> P;               // # pseudo categories
  int<lower=1, upper=I> ii[N];  // item for n
  int<lower=1, upper=J> jj[N];  // person for n
  int y[N];   // correctness for n
  int mapping[C,P];                // mapping pseudo items
}
parameters {
  vector[P] theta[J];              // latent traits matrix
  cholesky_factor_corr[P] Lcorr; // cholesky factor (L_u matrix for P)
  vector<lower=0>[P] sigma;     // variance
  real<lower=0>  alpha[I,P];      //discrimination parameters of pseudo items
  vector[P] beta[I];               //difficulty parameters of pseudo items
}
transformed parameters {
  corr_matrix[P] R; // correlation matrix
  cov_matrix[P] Sigma; // VCV matrix
  R = multiply_lower_tri_self_transpose(Lcorr); // Lcorr * Lcorr'
  Sigma = quad_form_diag(R, sigma); // diag_matrix(sig) * R * diag_matrix(sig)
}
model {
  beta ~ multi_normal(rep_vector(0,P),diag_matrix(rep_vector(1,P)));
  for (i in 1:I){
    for (p in 1:P){
      alpha[i,p] ~ lognormal(0,1);
    }
  }
  to_vector(sigma) ~ cauchy(0, 5); // prior for variancre
  Lcorr ~ lkj_corr_cholesky(2); // prior for cholesky factor of a correlation matrix
  theta ~ multi_normal(rep_vector(0,P), Sigma);

  for (n in 1:N){
    for (p in 1:P){
        if(mapping[y[n],p] != -1){
          mapping[y[n],p]~bernoulli_logit(alpha[ii[n],p]*(theta[jj[n],p]-beta[ii[n],p]));
        }
    }
  }
  
}

generated quantities {
  matrix[N,P] log_lik;
  for (n in 1: N){
      for (p in 1:P){
        if(mapping[y[n],p] != -1){
          log_lik[n,p]=bernoulli_logit_lpmf(mapping[y[n],p]|alpha[ii[n],p]*(theta[jj[n],p]-beta[ii[n],p]));
        }else{
          log_lik[n,p]=0;
        }
    }
      }
}