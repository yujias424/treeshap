#' Unify randomForest model
#'
#' Convert your randomForest model into a standarised representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param rf_model An object of \code{randomForest} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
#'
#' @return a unified model representation - a \code{\link{model_unified.object}} object
#'
#' @import data.table
#'
#' @export
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{\link[lightgbm:lightgbm]{LightGBM models}}
#'
#' \code{\link{gbm.unify}} for \code{\link[gbm:gbm]{GBM models}}
#'
#' \code{\link{catboost.unify}} for  \code{\link[catboost:catboost.train]{Catboost models}}
#'
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{ranger.unify}} for \code{\link[ranger:ranger]{ranger models}}
#'
#' @examples
#'
#' library(randomForest)
#' data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
#'                            c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
#'                              'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
#' data <- na.omit(cbind(data_fifa, target = fifa20$target))
#'
#' rf <- randomForest::randomForest(target~., data = data, maxnodes = 10, ntree = 10)
#' unified_model <- randomForest.unify(rf, data)
#' shaps <- treeshap(unified_model, data[1:2,])
#' # plot_contribution(shaps, obs = 1)
#'
randomForest.unify <- function(rf_model, data, W = NULL, Y = NULL, is_grf = FALSE, numCores = 8) {
  if (is_grf == FALSE){
    if(!'randomForest' %in% class(rf_model)){stop('Object rf_model was not of class "randomForest"')}
    if(any(attr(rf_model$terms, "dataClasses") != "numeric")) {
      stop('Models built on data with categorical features are not supported - please encode them before training.')
    }
    n <- rf_model$ntree
    ret <- data.table()
    x <- lapply(1:n, function(tree){
      tree_data <- as.data.table(getTree(rf_model, k = tree, labelVar = TRUE)) # lv = T
      tree_data[, c("left daughter", "right daughter", "split var", "split point", "prediction")]
    })
    times_vec <- sapply(x, nrow)
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

    ret <- list(model = as.data.frame(y), data = as.data.frame(data))
    class(ret) <- "model_unified"
    attr(ret, "missing_support") <- FALSE
    attr(ret, "model") <- "randomForest"
    return(set_reference_dataset(ret, as.data.frame(data)))
  } else {
    if(!'causal_forest' %in% class(rf_model)){stop('Object rf_model was not of class "causal_forest"')}
    # if(any(attr(rf_model$terms, "dataClasses") != "numeric")) {
    #   stop('Models built on data with categorical features are not supported - please encode them before training.')
    # }

    # convert cf to rf
    rf_model_tmp <- cf_to_rf(rf_model, X = data, W = W, Y = Y, numCores = numCores)
    rf_model <- rf_model_tmp

    print("convert successfully")

    # n <- rf_model$ntree #'_num_trees'
    # ret <- data.table()
    # x <- lapply(1:n, function(tree){
    #   cf_tree <- grf::get_tree(rf_model, index = tree)
    #   tree_data <- get_cftree(cf_tree, data, W, Y)
    # })

    # x <- x[-which(sapply(x, is.null))]
    # cf$ntree <- length(x)
    
    # na_idx <- c()
    # for (i in 1:length(x)){
    #   if (sum( is.na( x[[i]]$prediction ) ) > 0){
    #     na_idx <- c(na_idx, i)
    #   }
    # }
    
    # for (i in na_idx){
    #   x[[i]] <- NULL
    # }
    # n <- length(x)
    
    n <- rf_model$ntree
    ret <- data.table()
    x <- lapply(1:n, function(tree){
      tree_data <- as.data.table(getTree(rf_model, k = tree, labelVar = TRUE)) # lv = T
      tree_data[, c("left daughter", "right daughter", "split var", "split point", "prediction")]
    })
    times_vec <- sapply(x, nrow)
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
    
    ret <- list(model = as.data.frame(y), data = as.data.frame(data))
    class(ret) <- "model_unified"
    attr(ret, "missing_support") <- FALSE
    attr(ret, "model") <- "randomForest"
    return(set_reference_dataset(ret, as.data.frame(data)))
  }
  
}

get_cftree <- function(cf_tree, X, W, Y){
  
  # get drawn dat
  drawn_sample_dat <- as.data.frame(X[sort(cf_tree$drawn_samples), ])
  drawn_sample_dat$id <- sort(cf_tree$drawn_samples)
  drawn_sample_w <- data.frame("tx" = W[sort(cf_tree$drawn_samples)])
  drawn_sample_w$id <- sort(cf_tree$drawn_samples)
  drawn_sample_y <- data.frame("outcome" = Y[sort(cf_tree$drawn_samples)])
  drawn_sample_y$id <- sort(cf_tree$drawn_samples)
  
  # add node index in nodes
  for (i in 1:length(cf_tree$nodes)){
    cf_tree$nodes[[i]]$index <- i
  }
  
  # add father node index in nodes
  for (i in 1:length(cf_tree$nodes)){
    if(cf_tree$nodes[[i]]$index == 1){
      cf_tree$nodes[[i]]$father <- 0
    } else {
      for (j in 1:length(cf_tree$nodes)){
        # if (!cf_tree$nodes[[i]]$is_leaf){
        if (cf_tree$nodes[[i]]$index %in% c(cf_tree$nodes[[j]]$left_child, cf_tree$nodes[[j]]$right_child)){
          cf_tree$nodes[[i]]$father <- j
        } 
        # }
      }
    }
  }
  
  # add samples and corresponding tx assignment in the nodes
  for (i in 1:length(cf_tree$nodes)){
    if (cf_tree$nodes[[i]]$father != 0){
      if (!cf_tree$nodes[[i]]$is_leaf){
        father_index <- cf_tree$nodes[[i]]$father
        father_dat <- drawn_sample_dat[drawn_sample_dat$id %in% cf_tree$nodes[[father_index]]$sample, ]
        ld_sample <- father_dat[father_dat[, cf_tree$nodes[[father_index]]$split_variable] <= cf_tree$nodes[[father_index]]$split_value,]$id
        rd_sample <- father_dat[father_dat[, cf_tree$nodes[[father_index]]$split_variable] > cf_tree$nodes[[father_index]]$split_value,]$id
        
        if (cf_tree$nodes[[i]]$index == cf_tree$nodes[[father_index]]$left_child){
          cf_tree$nodes[[i]]$samples <- ld_sample
          cf_tree$nodes[[i]]$tx_assign <- drawn_sample_w[drawn_sample_w$id %in% ld_sample, ]$tx
          cf_tree$nodes[[i]]$outcome <- drawn_sample_y[drawn_sample_y$id %in% ld_sample, ]$outcome
        } else if (cf_tree$nodes[[i]]$index == cf_tree$nodes[[father_index]]$right_child){
          cf_tree$nodes[[i]]$samples <- rd_sample
          cf_tree$nodes[[i]]$tx_assign <- drawn_sample_w[drawn_sample_w$id %in% rd_sample, ]$tx
          cf_tree$nodes[[i]]$outcome <- drawn_sample_y[drawn_sample_y$id %in% rd_sample, ]$outcome
        }
      } else {
        cf_tree$nodes[[i]]$tx_assign <- drawn_sample_w[drawn_sample_w$id %in% cf_tree$nodes[[i]]$samples, ]$tx
        cf_tree$nodes[[i]]$outcome <- drawn_sample_y[drawn_sample_y$id %in% cf_tree$nodes[[i]]$samples, ]$outcome
      }
    } else {
      cf_tree$nodes[[i]]$samples <- sort(cf_tree$drawn_samples)
      cf_tree$nodes[[i]]$tx_assign <- drawn_sample_w[drawn_sample_w$id %in% cf_tree$nodes[[i]]$samples, ]$tx
      cf_tree$nodes[[i]]$outcome <- drawn_sample_y[drawn_sample_y$id %in% cf_tree$nodes[[i]]$samples, ]$outcome
    }
  }
  
  # estimate the prediction tau for the node
  for (i in 1:length(cf_tree$nodes)){
    tmp_df <- data.frame("id" = cf_tree$nodes[[i]]$samples, "tx" = cf_tree$nodes[[i]]$tx_assign, "outcome" = cf_tree$nodes[[i]]$outcome)
    # print(tmp_df)
    # print(tmp_df[tmp_df$tx == 1, ]$outcome)
    # print(tmp_df[tmp_df$tx == 0, ]$outcome)
    mean_tret <- mean(tmp_df[tmp_df$tx == 1, ]$outcome)
    mean_ctrl <- mean(tmp_df[tmp_df$tx == 0, ]$outcome)
    # print(mean_tret)
    # print(mean_ctrl)
    tau_prediction <- mean_tret - mean_ctrl
    # print(tau_prediction)
    cf_tree$nodes[[i]]$tau_pred <- tau_prediction

    # Any NaN prediction will cause failure in shap calculation.
    if (is.nan(mean(tmp_df[tmp_df$tx == 1, ]$outcome)) | is.nan(mean(tmp_df[tmp_df$tx == 0, ]$outcome))){
      # print("drop it!")
      drop <- T
      break
    } else {
      drop <- F
    }
  }
  
  # build the data table
  ld <- c() # left daughter
  rd <- c() # right daughter
  sv <- c() # split variable
  sp <- c() # split point
  st <- c() # status
  pred <- c() # tau prediction
  for (i in 1:length(cf_tree$nodes)){
    if (!cf_tree$nodes[[i]]$is_leaf){
      ld <- c(ld, cf_tree$nodes[[i]]$left_child)
      rd <- c(rd, cf_tree$nodes[[i]]$right_child)
      sv <- c(sv, colnames(X)[cf_tree$nodes[[i]]$split_variable]) #cf_tree$nodes[[i]]$split_variable
      sp <- c(sp, cf_tree$nodes[[i]]$split_value)
    } else {
      ld <- c(ld, 0)
      rd <- c(rd, 0)
      sv <- c(sv, NA)
      sp <- c(sp, 0.00)
    }
    
    pred <- c(pred, cf_tree$nodes[[i]]$tau_pred)
  }
  
  # print(pred)
  tree_data <- data.table("left daughter" = ld, "right daughter" = rd, "split var" = sv, "split point" = sp, "prediction" = pred)
  # tree_data
  if (drop == F){
    return(tree_data)
  } else {
    return(NULL)
  }
}

cf_to_rf <- function(cf, X, W, Y, numCores = 8){
  cf$type <- "regression"
  cf$ntree <- cf$'_num_trees'
  cf$forest <- list()
  n <- cf$ntree
  
  print("start mc step 1")
  start.time <- Sys.time()
  if (numCores > 1){

    # cl <- makeCluster(numCores)
    # registerDoParallel(cl)

    # x <- foreach(tree = 1:n) %dopar% {
    #   cf_tree <- grf::get_tree(cf, index = tree)
    #   tree_data <- get_cftree(cf_tree, X, W, Y)
    #   tree_data
    # }

    # stopCluster(cl)

    # x <- lapply(1:n, function(tree){
    #   cf_tree <- grf::get_tree(cf, index = tree)
    #   tree_data <- get_cftree(cf_tree, X, W, Y)
    # })

    x <- mclapply(1:n, function(tree){
      cf_tree <- grf::get_tree(cf, index = tree)
      tree_data <- get_cftree(cf_tree, X, W, Y)
    }, mc.cores = numCores)

  } else {
    x <- lapply(1:n, function(tree){
      cf_tree <- grf::get_tree(cf, index = tree)
      tree_data <- get_cftree(cf_tree, X, W, Y)
    })
  }
  end.time <- Sys.time()
  print("end mc step 1")
  time.taken <- end.time - start.time
  print(time.taken)
  
  # There may exist some NULL object in the list
  x <- x[-which(sapply(x, is.null))]
  cf$ntree <- length(x)
  
  print("start mc step 2")
  start.time <- Sys.time()
  if(numCores <= 1){
    # join left daughter
    ld <- x[[1]]$`left daughter`
    rd <- x[[1]]$`right daughter`
    ndbt <- c(dim(x[[1]])[1])
    np <- x[[1]]$prediction
    xbs <- x[[1]]$`split point`
    bv <- x[[1]]$`split var`
    for (i in 2:length(x)){
      ld <- rowr::cbind.fill(ld, x[[i]]$`left daughter`, fill = 0)
      rd <- rowr::cbind.fill(rd, x[[i]]$`right daughter`, fill = 0)
      ndbt <- c(ndbt, dim(x[[i]])[1])
      np <- rowr::cbind.fill(np, x[[i]]$prediction, fill = 0)
      xbs <- rowr::cbind.fill(xbs, x[[i]]$`split point`, fill = 0)
      bv <- rowr::cbind.fill(bv, x[[i]]$`split var`, fill = NA)
    }
  } else {

    # get the max length of target var
    ld_len_max <- max(unlist(lapply(x, function(x) length(x$`left daughter`))))
    rd_len_max <- max(unlist(lapply(x, function(x) length(x$`right daughter`))))
    np_len_max <- max(unlist(lapply(x, function(x) length(x$prediction))))
    xbs_len_max <- max(unlist(lapply(x, function(x) length(x$`split point`))))
    bv_len_max <- max(unlist(lapply(x, function(x) length(x$`split var`))))

    ld_list <- lapply(x, function(x) {
        tmp <- x$`left daughter`
        length(tmp) <- ld_len_max
        tmp
      })
    
    ld_list <- rapply(ld_list, f=function(x) ifelse(is.na(x), 0, x), how="replace")

    rd_list <- lapply(x, function(x) {
        tmp <- x$`right daughter`
        length(tmp) <- rd_len_max
        tmp
      })
    
    rd_list <- rapply(rd_list, f=function(x) ifelse(is.na(x), 0, x), how="replace")

    np_list <- lapply(x, function(x) {
        tmp <- x$prediction
        length(tmp) <- np_len_max
        tmp
      })

    np_list <- rapply(np_list, f=function(x) ifelse(is.na(x), 0, x), how="replace")

    xbs_list <- lapply(x, function(x) {
        tmp <- x$`split point`
        length(tmp) <- xbs_len_max
        tmp
      })

    xbs_list <- rapply(xbs_list, f=function(x) ifelse(is.na(x), 0, x), how="replace")

    bv_list <- lapply(x, function(x) {
        tmp <- x$`split var`
        length(tmp) <- bv_len_max
        tmp
      })

    ndbt <- c()
    for (i in 1:length(x)){
      ndbt <- c(ndbt, dim(x[[i]])[1])
    }

    # cl <- makeCluster(numCores)
    # registerDoParallel(cl)

    # ndbt <- foreach(i = 1:length(x), .combine=c) %dopar% dim(x[[i]])[1]

    # stopCluster(cl)

    # list to dataframe
    ld <- data.frame(matrix(unlist(ld_list), ncol = length(x)))
    rd <- data.frame(matrix(unlist(rd_list), ncol = length(x)))
    np <- data.frame(matrix(unlist(np_list), ncol = length(x)))
    xbs <- data.frame(matrix(unlist(xbs_list), ncol = length(x)))
    bv <- data.frame(matrix(unlist(bv_list), ncol = length(x)))

    # # a mc version
    # ld <- foreach(i = 1:length(x), .combine=function(x, y)cbind.fill(x, y, fill = 0)) %dopar% x[[i]]$`left daughter`
    # rd <- foreach(i = 1:length(x), .combine=function(x, y)cbind.fill(x, y, fill = 0)) %dopar% x[[i]]$`right daughter`
    # ndbt <- foreach(i = 1:length(x), .combine=c) %dopar% dim(x[[i]])[1]
    # np <- foreach(i = 1:length(x), .combine=function(x, y)cbind.fill(x, y, fill = 0)) %dopar% x[[i]]$prediction
    # xbs <- foreach(i = 1:length(x), .combine=function(x, y)cbind.fill(x, y, fill = 0)) %dopar% x[[i]]$`split point`
    # bv <- foreach(i = 1:length(x), .combine=function(x, y)cbind.fill(x, y, fill = NA)) %dopar% x[[i]]$`split var`

  }

  end.time <- Sys.time()
  print("end mc step 2")
  time.taken <- end.time - start.time
  print(time.taken)

  colnames(ld) <- seq(1, dim(ld)[2])
  colnames(rd) <- seq(1, dim(rd)[2])
  colnames(np) <- seq(1, dim(np)[2])
  colnames(xbs) <- seq(1, dim(xbs)[2])
  colnames(bv) <- seq(1, dim(bv)[2])

  ld <- as.matrix(ld)
  mode(ld) <- "integer"
  
  rd <- as.matrix(rd)
  mode(rd) <- "integer"
  
  bv <- as.data.frame(lapply(bv, as.factor))

  # bv <- as.matrix(bv)
  # mode(bv) <- "character"
  # bv[is.na(bv)] <- 0
  
  
  
  colnames(ld) <- NULL
  colnames(rd) <- NULL
  
  xbs <- as.matrix(xbs)
  colnames(xbs) <- NULL
  
  bv_tmp <- matrix(, nrow = dim(bv)[1], ncol = dim(bv)[2])
  
  for (i in 1:dim(bv)[1]){
    for (j in 1:dim(bv)[2]){
      if (!is.na(bv[i, j])){
        bv_tmp[i,j] <- which(colnames(X) == bv[i, j])
      } else {
        bv_tmp[i,j] <- 0
      }
    }
  }
  
  np <- as.matrix(np)
  colnames(np) <- NULL
  
  cf$forest$leftDaughter <- ld
  cf$forest$rightDaughter <- rd
  cf$forest$bestvar <- bv_tmp
  cf$forest$xbestsplit <- xbs
  cf$forest$ndbigtree <- ndbt
  cf$forest$nodepred <- np
  
  cf$forest$nodestatus <- ld
  
  cf$importance <- data.frame("IncNodePurity" = rep(0, dim(X)[2]))
  row.names(cf$importance) <- colnames(X)
  class(cf) <- "causal_forest"
  return(cf)
}

# getTree <- function (rfobj, k = 1, labelVar = FALSE) 
# {
#   if (is.null(rfobj$forest)) {
#     stop("No forest component in ", deparse(substitute(rfobj)))
#   }
#   if (k > rfobj$ntree) {
#     stop("There are fewer than ", k, "trees in the forest")
#   }
#   if (rfobj$type == "regression") {
#     tree <- cbind(rfobj$forest$leftDaughter[, k], rfobj$forest$rightDaughter[, 
#                                                                              k], rfobj$forest$bestvar[, k], rfobj$forest$xbestsplit[, 
#                                                                                                                                     k], rfobj$forest$nodestatus[, k], rfobj$forest$nodepred[, 
#                                                                                                                                                                                             k])[1:rfobj$forest$ndbigtree[k], ]
#   }
#   else {
#     tree <- cbind(rfobj$forest$treemap[, , k], rfobj$forest$bestvar[, 
#                                                                     k], rfobj$forest$xbestsplit[, k], rfobj$forest$nodestatus[, 
#                                                                                                                               k], rfobj$forest$nodepred[, k])[1:rfobj$forest$ndbigtree[k], 
#                                                                                                                               ]
#   }
#   dimnames(tree) <- list(1:nrow(tree), c("left daughter", "right daughter", 
#                                          "split var", "split point", "status", "prediction"))
#   if (labelVar) {
#     tree <- as.data.frame(tree)
#     v <- tree[[3]]
#     v[v == 0] <- NA
#     tree[[3]] <- factor(colnames(tau.forest_rf$X.orig)[v])
#     if (rfobj$type == "classification") {
#       v <- tree[[6]]
#       v[!v %in% 1:nlevels(rfobj$y)] <- NA
#       tree[[6]] <- levels(rfobj$y)[v]
#     }
#   }
#   tree
# }