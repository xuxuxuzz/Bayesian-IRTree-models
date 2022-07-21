data {
  int<lower=1> I;               // # of items
  int<lower=1> J;               // # of persons
  int<lower=1> N;               // # of total responses
  int<lower=1> C;               // # of categories
  int<lower=1> P;               // # of pseudo categories
  int<lower=1, upper=I> ii[N];  // item for n
  int<lower=1, upper=J> jj[N];  // person for n
  int y[N];   // correctness for n
  int mapping[C,P];                // mapping pseudo items
  
  int<lower=1> IC;               // # of item covariates
  int<lower=1> PC;               // # of person covariates
  matrix[I,IC] itemcov;            // covariates for items
  matrix[J,PC] personcov;          // covariates for persons
}
parameters {
  vector[P] theta[J];              // latent traits matrix
  cholesky_factor_corr[P] Lcorr; // cholesky factor (L_u matrix for P)
  vector<lower=0>[P] sigma;     // variance
  real<lower=0>  alpha[I,P];      //discrimination parameters of pseudo items
  vector[P] beta[I];               //difficulty parameters of pseudo items
  matrix[IC,P] gamma;             //coefficients for item properties
  matrix[PC,P] lambda;             //coefficients for person properties
}
transformed parameters {
  corr_matrix[P] R; // correlation matrix
  cov_matrix[P] Sigma; // VCV matrix
  R = multiply_lower_tri_self_transpose(Lcorr); // Lcorr * Lcorr'
  Sigma = quad_form_diag(R, sigma); // diag_matrix(sig) * R * diag_matrix(sig)
}
model {
  
  for (ic in 1:IC){
    gamma[ic,] ~ normal(0,1);
  }

  for (i in 1:I){
    beta[i,] ~ multi_normal(itemcov[i,]*gamma, diag_matrix(rep_vector(1,P)));
    for (p in 1:P){
      alpha[i,p] ~ lognormal(0,1);
    }
  }
  to_vector(sigma) ~ cauchy(0, 5); // prior for variancre
  Lcorr ~ lkj_corr_cholesky(2); // prior for cholesky factor of a correlation matrix
  
  for (pc in 1:PC){
    lambda[pc,] ~ normal(0,1);
  }
  for (j in 1:J){
    theta[j,] ~ multi_normal(personcov[j,]*lambda, Sigma);
  }


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