cf_to_rf <- function(cf, X, W, Y, rf){
  cf$type <- "regression"
  cf$ntree <- tau.forest$'_num_trees'
  cf$forest <- list()
  n <- cf$ntree
  
  x <- lapply(1:n, function(tree){
    cf_tree <- grf::get_tree(cf, index = tree)
    tree_data <- get_cftree(cf_tree, X, W, Y)
  })
  
  x <- x[-which(sapply(x, is.null))]
  cf$ntree <- length(x)
  
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
  
  ld <- as.matrix(ld)
  mode(ld) <- "integer"
  
  rd <- as.matrix(rd)
  mode(rd) <- "integer"
  
  bv <- as.data.frame(lapply(bv, as.factor))
  # bv <- as.matrix(bv)
  # mode(bv) <- "character"
  # bv[is.na(bv)] <- 0
  
  colnames(ld) <- seq(1, dim(ld)[2])
  colnames(rd) <- seq(1, dim(rd)[2])
  colnames(np) <- seq(1, dim(np)[2])
  colnames(xbs) <- seq(1, dim(xbs)[2])
  colnames(bv) <- seq(1, dim(bv)[2])
  
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
  cf$terms <- rf$terms
  class(cf) <- class(rf) # "causal_forest" # 
  return(cf)
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
    
    # print(is.nan(tau_prediction))
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

