---
title: "Weight Lifting Exercise Dataset"
subtitle: "Coursera - Practical Machine Learning Assignment"
date: "24 June 2018"
output: 
  html_document:
    keep_md: yes
  md_document:
    variant: markdown_github
  pdf_document: default
---



# Executive Summary

The Weight Lifting Exercise Dataset will be used to build a predictive model to predict the manner in which 6 people executed exercises wearing accelerometers on the belt, forearm, arm, and dumbel while performing barbell lifts, which is given by the "classe" variable.

Building the predictive model followed these steps:

* Data cleaning and preparation,
* Exploratory data analysis,
* Pre-processing,
* Model training, examination and testing,
* Predicting the classification in the test set.


# Data source
The data for this project come originally from this source: http://groupware.les.inf.puc-rio.br/har. [1] 

The training data for this project were prepared by Coursera and are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data which will be used to answer 20 cases are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# Data cleaning and preparation


```r
train_orig <- read.csv("pml-training.csv", stringsAsFactors = FALSE, na.strings=c("", "NA", "#DIV/0!"))
test_orig <- read.csv("pml-testing.csv", stringsAsFactors = FALSE, na.strings=c("", "NA", "#DIV/0!"))
dim(train_orig) #Dimensions
```

```
## [1] 19622   160
```

```r
dim(test_orig) #Dimensions
```

```
## [1]  20 160
```

```r
#head(train_orig) #excluded from knitr execution due to output size
#str(train_orig) #excluded from knitr execution due to output size
#summary(train_orig) #excluded from knitr execution due to output size
```
A lot of variables contain a great number of NA. Remove these columns first:

```r
rows_count <- nrow(train_orig)
max_na <- rows_count * 0.1 #90% of the data should not be NA
cols_to_remove <- colSums(is.na(train_orig)) > max_na
train_clean <- train_orig[,!cols_to_remove]
test_clean <- test_orig[,!cols_to_remove]
```

Remove other columns that we expect not to use:

```r
timestamps <- grep("timestamp", names(train_clean))
train_clean <- train_clean[,-c(1, timestamps)]
test_clean <- test_clean[,-c(1, timestamps)]
```

Convert character vectors to factors:

```r
train_clean$classe <- factor(train_clean$classe)
train_clean$user_name <- factor(train_clean$user_name)
user_name_lvls <- levels(train_clean$user_name)
train_clean$new_window <- factor(train_clean$new_window)
new_window_lvls <- levels(train_clean$new_window)
test_clean$user_name <- factor(test_clean$user_name, levels=user_name_lvls)
test_clean$new_window <- factor(test_clean$new_window, levels=new_window_lvls)
```

# Exploratory data analyses 

The test set provided by Coursera is the test set to be used for the ultimate evaluation of the model. Therefore we will only use the trainset for training and testing during model building proces.


```r
set.seed(11223344)
classe_col_idx <- which(names(train_clean) == "classe")
in_train <- createDataPartition(y=train_clean$classe, p=0.75, list=FALSE)
mod_train <- train_clean[in_train, ]
mod_test <- train_clean[-in_train, ]
```

Trying to find variables which are correlated to 'classe'?

```r
classe_corr <- cor(data.matrix(mod_train[, -classe_col_idx]), as.numeric(mod_train$classe))
classe_corr[abs(classe_corr) > 0.3,]
```

```
## pitch_forearm 
##     0.3458645
```

A correlation matrix might help to identify highly correlated features. During model selection they might be excluded.

```r
cm <- cor(data.matrix(mod_train[, -classe_col_idx])) #calculate correlation matrix
corrplot(cm, type="lower", tl.cex=0.70, tl.col="black", tl.srt = 45, diag = FALSE, method="ellipse")
```

![](Practical_machine_learning_project_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

Some features appear highly correlated.

# Pre-processing

We will use the `preProcess` function from the caret packet to apply pre-processing transformations and variable selection:


```r
# exclude variables that are highly correlated and
# center and scale the variables
ppm <-preProcess(mod_train[,-classe_col_idx],method=c("corr", "center", "scale"))
mod_train_pp <- predict(ppm, mod_train[,-classe_col_idx])
mod_train_pp$classe <- mod_train$classe

mod_test_pp <- predict(ppm, mod_test[,-classe_col_idx])
mod_test_pp$classe <- mod_test$classe
```

# Model training 
## Random Forest
A random forest will be trained. Applying 5-fold cross-validation should avoid overfitting. We limit the number of trees to 100 since earlier runs showed that the error rate decreases only very with more than that.

```r
start <- proc.time()
rf_train <- train(classe ~.,
                  method="rf", #random forest
                  ntree=100, #limit the number of trees to 100
                  data=mod_train_pp, #pre-processed train data
                  trControl=trainControl(method='cv',
                                         number=5), #5-fold cross validation
                  allowParallel=TRUE, importance=TRUE)
proc.time() - start
```

```
##    user  system elapsed 
##  237.22    0.97  239.47
```

```r
rf_train
```

```
## Random Forest 
## 
## 14718 samples
##    48 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11774, 11775, 11774, 11775, 11774 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9915071  0.9892560
##   27    0.9970105  0.9962187
##   52    0.9942249  0.9926955
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

## Examination of the model

```r
varImpPlot(rf_train$finalModel, cex=0.7, pch=16, main='Variable importance')
```

![](Practical_machine_learning_project_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

```r
plot(rf_train$finalModel, cex=0.7, main='Error vs. number of trees')
```

![](Practical_machine_learning_project_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

## Model testing and out-of-sample error estimate
Applying the fitted model on the test data set, to check its accuracy and estimated out of sample error.

```r
rf_test <- predict(rf_train, mod_test_pp)
conf_mat <- confusionMatrix(mod_test_pp$classe, rf_test)
conf_mat$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    1    0    0    0
##          B    1  946    2    0    0
##          C    0    0  855    0    0
##          D    0    0    2  802    0
##          E    0    0    0    2  899
```
A limited amount of measures from the test set fall out of the model predictions.


```r
# model accuracy
acc <- postResample(mod_test_pp$classe, rf_test)
mod_acc <- acc[[1]]
mod_acc
```

```
## [1] 0.9983687
```

```r
#out of sample error
oos_err <- 1 - mod_acc 
oos_err
```

```
## [1] 0.001631321
```
The estimated accuracy of the model is 99.8% and the estimated out-of-sample error based on our fitted model applied to the test dataset is 0.02%.

# Apply the model
We now can apply the model to the testing data set to answer the 20 problems from the Coursera quiz:

```r
test_clean_pp <- predict(ppm, test_clean)
rf_quiz <- predict(rf_train, test_clean_pp)
rf_quiz
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

# References

[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. URL: http://groupware.les.inf.puc-rio.br/har