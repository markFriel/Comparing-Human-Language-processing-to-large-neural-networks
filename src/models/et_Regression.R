library(data.table)
library(ggplot2)
library(caret)
library(car)

#Function to compute the Adjusted R squared values of a set of modelsxs
create_model <- function(target, predictors, df){
    data_ctrl <- trainControl(method = "cv", number = 5)
    fl <- as.formula(paste(target, paste(predictors, collapse = " + "), sep = " ~ "))
    model_caret <- train(fl, data = df, trControl = data_ctrl, method = 'lm', na.action = na.pass)
    return(model_caret)
}

create_log_model <- function(target, predictors, df){
  data_ctrl <- trainControl(method = "cv", number = 5)
  fl <- as.formula(paste(target, paste(predictors, collapse = " + "), sep = " ~ "))
  model_caret <- train(fl, data = df, trControl = data_ctrl, method = 'glm',family='binomial', na.action = na.pass)
  return(model_caret)
}

computeRsquared <- function(df, predictors, targets){
    list <- c()
    for (i in targets) {
        val_list <- c()
        model <- create_model(i, predictors, df)
        adjRsquared <- summary(model)$adj.r.squared
        val_list <- append(val_list, adjRsquared)
        list <- append(list, val_list)
    }
    return(list)
}


significance_testing <- function(df){
    sample_number = length(df)
    for (i in 1 : sample_number) {
        x1 <- df[c(i)]
        x1_name = colnames(x1)
        for (j in 1 : sample_number) {
            x2 <- df[c(j)]
            x2_name <- colnames(x2)
            results <- t.test(x1, x2)$p.value
            if (results < 0.05) {
                print(paste('The difference between', x1_name, ' and ', x2_name, ' is significant'))
            }
        }
    }
}


cbindlist <- function(list) {
    n <- length(list)
    res <- NULL
    for (i in seq(n))res <- cbind(res, list[[i]])
    return(res)
}

transpose_df <- function(df, names){
    df <- as.data.frame(t(as.matrix(df)))
    names(df) <- names
    names(df) <- gsub("_", " ", names)
    return(df)
}


flatten_df <- function(df){
    cat_frame <- data.frame()
    for (i in names(df)) {
        col <- df[[i]]
        frame <- data.frame(metric = i, Adjusted_R_Squared = col)
        cat_frame <- rbind(cat_frame, frame)
    }
    return(cat_frame)
}


reading_measure <- function(Subject_Path, feature_path, control_model_variables, independVariables, dependVariable){
    files <- list.files(path = Subject_Path)
    word_feature_df = read.csv(feature_path)
    column_names = append(independVariables, 'Base_Model', 0)
    list <- data.frame()

    for (j in files) {
        reading_df = read.csv(paste(Subject_Path, j, sep = ''))
        full_df <- cbind(reading_df, word_feature_df)
        full_df = subset(full_df, WORD_FIRST_FIXATION_DURATION > 100)
        full_df$WORD_SKIP <- as.factor(full_df$WORD_SKIP)
        R_squared_Values <- c()

        #Base model
        model <- create_log_model(dependVariable, control_model_variables, full_df)
        adjRsquared <- model$results$Accuracy
        R_squared_Values <- append(R_squared_Values, adjRsquared)

        # Base model with surprisal values
        for (k in independVariables) {
            predictors <- append(control_model_variables, k)
            model <- create_log_model(dependVariable, predictors, full_df)
            adjRsquared <- model$results$Accuracy
            R_squared_Values <- append(R_squared_Values, adjRsquared)
        }

        df = data.frame(dependVariable = R_squared_Values)
        df <- transpose_df(df, column_names)
        list <- rbind(list, df)
    }
    return(list)
}


aggregated_data <- function(clean_df, control_model_variables, independVariables, dependVariables){
    list <- c()
    Row_names <- append(independVariables, 'Base', 0)

    #Base Model
    R_squared_Values <- computeRsquared(clean_df, control_model_variables, dependVariables)
    df = data.frame(i = R_squared_Values)
    list <- append(list, df)

    for (i in independVariables) {
        predictors = append(control_model_variables, i)
        R_squared_Values <- computeRsquared(clean_df, predictors, dependVariables)
        df = data.frame(i = R_squared_Values)
        list <- append(list, df)
    }

    result <- cbindlist(list)
    result <- transpose_df(result, dependVariables)
    row.names(result) <- Row_names
    return(result)
}





















