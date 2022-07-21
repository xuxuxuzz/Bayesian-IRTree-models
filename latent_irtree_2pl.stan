data {
  int<lower=1> I;               // # of items
  int<lower=1> J;               // # of persons
  int<lower=1> N;               // # observations
  int<lower=1> C;               // # of categories
  int<lower=1> L;               // # of latent traits
  int<lower=1> D;               // # of nodes
  int<lower=1, upper=I> ii[N];  // item for n
  int<lower=1, upper=J> jj[N];  // person for n
  int<lower=1, upper=D> dd[N];  // nodes for n
  int y[N];                     // correctness for n
  int mapping[D,L];             // mapping pseudo items
  

}
parameters {
  vector[L] theta[J];              // abilities
  vector<lower=0>[L] sigma;       // Variance
  vector<lower=0>[L] alpha[I];      //discrimination parameters of pseudo items
  ordered[C-1] beta[I];             //category difficulty
}

model {
  for (i in 1:I){
    for (c in 1:(C-1)){
        beta[i,c] ~ normal(0,3);
    }
  }

  for (i in 1:I){
    for (l in 1:L){
      alpha[i,l] ~ lognormal(0,1);
    }
  }
  sigma ~ cauchy(0, 5); // prior for variancre
  theta ~ multi_normal(rep_vector(0,L), diag_matrix(sigma));

  for (n in 1:N){
  y[n] ~ ordered_logistic(sum(to_vector(alpha[ii[n],]).*to_vector(mapping[dd[n],]).*to_vector(theta[jj[n],])),beta[ii[n],]);
  }
  
}
