library("randomForest")
library(grf)
library(data.table)
library(treeshap)

source("./R/cf_to_rf.R")

# n <- 500
# p <- 30
# X <- matrix(rnorm(n * p), n, p)
# 
# colnames(X) <- c("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
#                  "one1", "two2", "three3", "four4", "five5", "six6", "seven7", "eight8", "nine9", "ten10",
#                  "one11", "two12", "three13", "four14", "five15", "six16", "seven17", "eight18", "nine19", "ten110")
# 
# # Train a causal forest.
# W <- rbinom(n, 1, 0.4 + 0.2 * (X[, 1] > 0))
# Y <- pmax(X[, 1], 0) * W + X[, 2] + X[, 5] + X[, 7] - X[, 18] + pmin(X[, 3], 0) + rnorm(n)
# tau.forest <- causal_forest(X, Y, W, num.trees = 500, honesty = T)
# 
# tau.forest.ht <- causal_forest(X, Y, W, num.trees = 500, honesty = T)
# tau.forest.hf <- causal_forest(X, Y, W, num.trees = 500, honesty = F)

cf.ht <- cf_to_rf(tau.forest.ht, X, W, Y, rf)
cf.hf <- cf_to_rf(tau.forest.hf, X, W, Y, rf)

dim(cf.ht$forest$bestvar)
dim(cf.hf$forest$bestvar)

plot(get_tree(tau.forest.ht, 1))
plot(get_tree(tau.forest.hf, 1))

a.ht <- randomForest.unify(cf.ht, X, is_grf = F)
treeshap_cf_ht <- treeshap(a.ht, X)

a.hf <- randomForest.unify(cf.hf, X, is_grf = F)
treeshap_cf_hf <- treeshap(a.hf, X)

ht <- a.ht$model
hf <- a.hf$model

# test for unify
# cf_unify <- randomForest.unify(tau.forest, data = X, W, Y, is_grf = T)
# treeshap1 <- treeshap(cf_unify, as.data.frame(X))

# a simple rf version for me to identify where the problem located.
outcome <- predict(tau.forest, X)
dat_tmp_source <- as.data.frame(X)
dat_tmp <- as.data.frame(X)
dat_tmp$outcome <- outcome$predictions
rf <- randomForest(outcome ~ ., data = dat_tmp, ntree = 100, nodesize = 100)
# rf_unify <- randomForest.unify(rf, dat_tmp_source)
# treeshap2 <- treeshap(rf_unify, dat_tmp_source)

# convert cf to rf
cf <- cf_to_rf(tau.forest, X, W, Y, rf)

# test for rv
a <- cf$forest$bestvar
b <- rf$forest$bestvar
dim(a)
dim(b)

a <- randomForest.unify(cf, X, is_grf = F)
treeshap_cf <- treeshap(a, X)

b <- randomForest.unify(rf, X, is_grf = F)
treeshap_rf <- treeshap(b, X)

data <- X
rf_model <- cf #rf
n <- rf_model$ntree
ret <- data.table()
x <- lapply(1:n, function(tree){
  tree_data <- as.data.table(randomForest::getTree(rf_model, k = tree, labelVar = TRUE))
  tree_data[, c("left daughter", "right daughter", "split var", "split point", "prediction")]
})
times_vec <- sapply(x, nrow)

x_cf <- x
times_vec_cf <- times_vec 

rf_model <- rf
n <- rf_model$ntree
ret <- data.table()
x <- lapply(1:n, function(tree){
  tree_data <- as.data.table(randomForest::getTree(rf_model, k = tree, labelVar = TRUE))
  tree_data[, c("left daughter", "right daughter", "split var", "split point", "prediction")]
})
times_vec <- sapply(x, nrow)

x_rf <- x
times_vec_rf <- times_vec 

times_vec_cf
times_vec_rf

x <- x_cf
times_vec <- times_vec_cf
y <- rbindlist(x)
y[, Tree := rep(0:(n - 1), times = times_vec)]
y[, Node := unlist(lapply(times_vec, function(x) 0:(x - 1)))]
setnames(y, c("Yes", "No", "Feature", "Split",  "Prediction", "Tree", "Node"))
y[, Feature := as.character(Feature)]
y[, Yes := Yes - 1]
y[, No := No - 1]
y[y$Yes < 0, "Yes"] <- NA
y[y$No < 0, "No"] <- NA
y[, Missing := NA]
y[, Missing := as.integer(Missing)] # seems not, but needed

ID <- paste0(y$Node, "-", y$Tree)
y$Yes <- match(paste0(y$Yes, "-", y$Tree), ID)
y$No <- match(paste0(y$No, "-", y$Tree), ID)

y$Cover <- 0

y$Decision.type <- factor(x = rep("<=", times = nrow(y)), levels = c("<=", "<"))
y[is.na(Feature), Decision.type := NA]

# Here we lose "Quality" information
y[!is.na(Feature), Prediction := NA]

# treeSHAP assumes, that [prediction = sum of predictions of the trees]
# in random forest [prediction = mean of predictions of the trees]
# so here we correct it by adjusting leaf prediction values
y[is.na(Feature), Prediction := Prediction / n]


setcolorder(y, c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Prediction", "Cover"))

y_rf <- y
y_cf <- y

sapply(y_rf, class)
sapply(y_cf, class)

ret <- list(model = as.data.frame(y_cf), data = as.data.frame(data))
class(ret) <- "model_unified"
attr(ret, "missing_support") <- FALSE
attr(ret, "model") <- "randomForest"

a_rf <- set_reference_dataset(ret, as.data.frame(data))
a_cf <- set_reference_dataset(ret, as.data.frame(data))



# check for treeshap of two different algorithm
# cf a
unified_model <- a
model_a <- unified_model$model

roots_a <- which(model_a$Node == 0) - 1
yes_a <- model_a$Yes - 1
no_a <- model_a$No - 1
missing_a <- model_a$Missing - 1
feature_a <- match(model_a$Feature, colnames(x)) - 1
split_a <- model_a$Split
decision_type_a <- unclass(model_a$Decision.type)
is_leaf_a <- is.na(model_a$Feature)
value_a <- model_a$Prediction
cover_a <- model_a$Cover

x2_a <- as.data.frame(t(as.matrix(x))) # transposed to be able to pick a observation with [] operator in Rcpp
is_na_a <- is.na(x2_a) # needed, because dataframe passed to cpp somehow replaces missing values with random values

# rf b
unified_model <- b
model_b <- unified_model$model

roots_b <- which(model_b$Node == 0) - 1
yes_b <- model_b$Yes - 1
no_b <- model_b$No - 1
missing_b <- model_b$Missing - 1
feature_b <- match(model_b$Feature, colnames(x)) - 1
split_b <- model_b$Split
decision_type_b <- unclass(model_b$Decision.type)
is_leaf_b <- is.na(model_b$Feature)
value_b <- model_b$Prediction
cover_b <- model_b$Cover

x2_b <- as.data.frame(t(as.matrix(x))) # transposed to be able to pick a observation with [] operator in Rcpp
is_na_b <- is.na(x2_b) # needed, because dataframe passed to cpp somehow replaces missing values with random values







