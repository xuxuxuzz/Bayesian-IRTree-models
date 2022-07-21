library(rstan)
library(bayesplot)
library(reshape2)

##########################################
###### CHANGE YOUR PATH ACCORDINGLY#######
##########################################

### data can be found in osf (https://osf.io/4b85w/)
oxf <- read.csv('Documents/ARC_raw_data.CSV')
rse <- na.omit(oxf[,c('record_id',paste0('rse_',1:10))])

### reverse negatively worded item
rse[,paste0('rse_',c(2,5,6,8,9))] <- 5-rse[,paste0('rse_',c(2,5,6,8,9))]

### drop duplication
rse_u <- rse[!duplicated(rse$record_id),]

### transform data into long format
dt_rse <- as.data.frame(melt(rse_u,id.vars = "record_id"))
colnames(dt_rse) <- c('person','item','response')

########################################
####explanatory response tree model#####
########################################

### mapping scheme for targeted self-esteem and extreme response preferences
map_rse <- rbind(c(0,1,-1),
                 c(0,0,-1),
                 c(1,-1,0),
                 c(1,-1,1))

### negatively-worded = 0
itemcov <- rep(1,10)
itemcov[c(2,5,6,8,9)] <- 0

### child=0; parent/carer=1
personcov <- oxf[,c('record_id','group')]
personcov <- personcov[personcov$record_id %in% rse_u$record_id,]
personcov <- personcov[!duplicated(personcov$record_id),]-1

data_rse_ex <- list(I=length(unique(dt_rse$item)),J=length(unique(dt_rse$person)),N=nrow(dt_rse),
                    C=4,P=3,mapping=map_rse,
                    ii=as.integer(factor(x=dt_rse$item)),jj=as.integer(factor(x=dt_rse$person)),y=dt_rse$response,
                    IC=1,PC=1,itemcov=matrix(itemcov), personcov=matrix(personcov$group)
)

### run stan model
fit_rse_ex <- stan(file='Documents/explanatory_irtree_2pl.stan',data=data_rse_ex,chain=3, iter = 5000,cores=3)

### summarize results
fit_rse_ex_summary <- summary(fit_rse_ex)$summary

traceplot(fit_rse_ex,c('gamma','lambda'))

### visualize results
mcmc_areas(as.matrix(fit_rse_ex),pars=c('gamma[1,1]','gamma[1,2]','gamma[1,3]'),prob=0.95)
mcmc_areas(as.matrix(fit_rse_ex),pars=c('lambda[1,1]','lambda[1,2]','lambda[1,3]'),prob=0.95)


####################################
####linear latent-variable tree#####
####################################

### only keep duplicated cases
rse_d <- rse[rse$record_id %in% rse$record_id[duplicated(rse$record_id)],]

rse_d$node <- NA

### obtain the number of waves where each respondent participated in

for (each in unique(rse_d$record_id)){
  rse_d[rse_d$record_id==each,'node'] <- 1:sum(rse_d$record_id==each)
}



map_latent <- rbind(c(1,0,0),
                    c(1,1,0),
                    c(1,1,1))

dt_tree <- as.data.frame(melt(rse_d,id.vars = c('record_id','node')))
colnames(dt_tree) <- c('person','node','item','response')

data_latent_tree <- list(I=length(unique(dt_tree$item)),J=length(unique(dt_tree$person)),
                         C=4,L=3,D=3, mapping=map_latent,N=nrow(dt_tree),
                         dd=as.integer(factor(x=dt_tree$node)),
                         ii=as.integer(factor(x=dt_tree$item)),
                         jj=as.integer(factor(x=dt_tree$person)),
                         y=dt_tree$response
)

fit_rse_latent_tree <- stan(file='Documents/latent_irtree_2pl.stan',data=data_latent_tree,chain=3, iter = 500,cores=3)
fit_rse_latent_tree_summary <- summary(fit_rse_latent_tree )$summary

######## trace plot examples

traceplot(fit_rse_latent_tree ,c('theta[1,1]','theta[1,3]','theta[20,1]','theta[31,2]','theta[101,1]','theta[221,3]'))

traceplot(fit_rse_latent_tree ,c('theta[1,1]','theta[1,3]','theta[20,1]','theta[31,2]','theta[101,1]','theta[221,3]'))

### visualize results
parents <- na.omit(oxf[,c('record_id','group')])
growth <- data.frame(record_id=unique(dt_tree$person),
                     theta2=fit_rse_latent_tree_summary[grepl(pattern = 'theta.*2]$',rownames(fit_rse_latent_tree_summary)),'mean'],
                     theta3=fit_rse_latent_tree_summary[grepl(pattern = 'theta.*3]$',rownames(fit_rse_latent_tree_summary)),'mean'])

growth <- merge(x=parents,y=growth,by='record_id')

growth$group <- ifelse(growth$group==1,'adolescent','parent')

ggplot(data=growth,aes(x=theta2,group=group,fill=as.factor(group)))+
  geom_density(alpha=0.2)+
  guides(fill=guide_legend(title="group"))+theme_bw()

ggplot(data=growth,aes(x=theta3,group=group,fill=as.factor(group)))+
  geom_density(alpha=0.2)+
  guides(fill=guide_legend(title="group"))+theme_bw()