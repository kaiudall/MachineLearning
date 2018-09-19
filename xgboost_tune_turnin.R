library(devtools)
install_github("berndbischl/ParamHelpers")
install_github("mlr-org/mlr")
install_github("mlr-org/mlrMBO")
install.packages("parallelMap")
install.packages("FeatureHashing")

library(ParamHelpers)
library(parallelMap)
library(mlr)
library(mlrMBO)
library(FeatureHashing)
library(data.table)
library(xgboost)
library(mice)
library(tidyverse)
library(randomForest)
# We used xgboost to get our results. 
# We used the mlr package to hyptertune our xgboost
# We used randomForest's importance for variable slection

##################### Part 1 #####################
# Read in Data
train <- fread("/Users/joshyoung/Documents/BigData/train.csv")
test <- fread("/Users/joshyoung/Documents/BigData/test.csv")
y_train <- train$SalePrice

train <- train[, -81]

# Mark for whether or not test set
test$isTest <- rep(1,nrow(test))
train$isTest <- rep(0,nrow(train))

# Make all one dataset
train_test <- rbind(train, test)

train_test[is.na(train_test$Alley), 7] = 'None'; train_test[is.na(train_test$BsmtQual), 31] = 'None'
train_test[is.na(train_test$BsmtCond), 32] = 'None'; train_test[is.na(train_test$BsmtExposure), 33] = 'None'
train_test[is.na(train_test$BsmtExposure), 33] = 'None'; train_test[is.na(train_test$BsmtFinType1), 34] = 'None'
train_test[is.na(train_test$BsmtFinType2), 36] = 'None'; train_test[is.na(train_test$FireplaceQu), 58] = 'None'
train_test[is.na(train_test$GarageType), 59] = 'None'; train_test[is.na(train_test$GarageFinish), 61] = 'None'
train_test[is.na(train_test$GarageQual), 64] = 'None'; train_test[is.na(train_test$GarageCond), 65] = 'None'
train_test[is.na(train_test$PoolQC), 73] = 'None'; train_test[is.na(train_test$Fence), 74] = 'None'
train_test[is.na(train_test$MiscFeature), 75] = 'None'

# Turn to factors
cat_var <- names(train_test)[which(sapply(train_test, is.character))]
cat_var <- append(cat_var, c('MSSubClass', 'OverallQual', 'OverallCond'))
train_test[,(cat_var):=lapply(.SD, as.factor),.SDcols=cat_var]

train_test <- train_test[, -1]

######################### Part 2 ########################
# Impute missing values
set.seed(3827)
train_mice <- mice(train_test, method = 'pmm')
train_test <- mice::complete(train_mice)
# write.csv(train_test, '/Users/joshyoung/Documents/BigData/train_imp.csv', row.names = FALSE)

train_test <- read.csv('/Users/joshyoung/Documents/BigData/train_imp.csv')
test <- train_test %>%
  dplyr::filter(isTest == 1) %>%
  dplyr::select(-isTest)
train <- train_test %>%
  dplyr::filter(isTest == 0) %>%
  dplyr::select(-isTest)

# Get important variables
set.seed(3278)
rf <- randomForest(train, y_train, importance = TRUE)
sort(importance(rf)[, 1])

# Get variables to drop
features_to_drop <- c("MoSold", "Condition2", "PoolQC", "Electrical", "Street", "LowQualFinSF", "RoofMatl", 
                      "YrSold", "Utilities", "Heating", "PoolArea", "MiscFeature") 

train <- data.table(train); test <- data.table(test)
train = train[, -features_to_drop,with= FALSE]
test = test[, -features_to_drop,with= FALSE]

################# Part 3 ###################
train$SalePrice <- y_train
train <- data.frame(train)
tsk = makeRegrTask(data = train, target = "SalePrice")
tsk = createDummyFeatures(tsk)

# Starting tune / eta, nrounds normally start eta predictions 0.01 - 0.15
lrn = makeLearner("regr.xgboost", nthread = 4)
lrn = setHyperPars(lrn,
                   max_depth = 5,
                   min_child_weight = 1,
                   gamma = 0,
                   subsample = .8,
                   colsample_bytree = .8,
                   objective = 'reg:linear'
)
res = makeResampleDesc("CV", iters = 4L)
par = makeParamSet(
  makeIntegerParam(id = "nrounds", lower = 40,upper = 100),
  makeNumericParam(id = "eta", lower = 0.01, upper = 0.15)
)

mbo.ctrl = makeMBOControl()
mbo.ctrl = setMBOControlInfill(mbo.ctrl, crit = crit.ei)
mbo.ctrl = setMBOControlTermination(mbo.ctrl, max.evals = 25L)

design.mat = generateRandomDesign(n = 200, par.set = par)
ctrl = makeTuneControlMBO(mbo.control = mbo.ctrl, mbo.design = design.mat)
parallelStartMulticore(cpus = 4L)
tune.pars = tuneParams(learner = lrn, task = tsk, resampling = res,
                       measures = rmsle, par.set = par, control = ctrl)
parallelStop()
tune.pars$x
tune.pars$y
# I got nrounds = 63, eta = 0.1477969, Yours will be different.  Plug these into second tuning

# Second Tuning / max_depth, min_child_weight
lrn = makeLearner("regr.xgboost", nthread = 4)
lrn = setHyperPars(lrn,
                   nrounds = 63,
                   eta =  0.1477969,
                   gamma = 0,
                   subsample = .8,
                   colsample_bytree = .8,
                   objective = 'reg:linear'
)
res = makeResampleDesc("CV", iters = 5L)
par = makeParamSet(
  makeIntegerParam(id = "max_depth", lower = 3, upper = 11),
  makeIntegerParam(id = "min_child_weight", lower = 1, upper = 7)
)

mbo.ctrl = makeMBOControl()
mbo.ctrl = setMBOControlInfill(mbo.ctrl, crit = crit.ei)
mbo.ctrl = setMBOControlTermination(mbo.ctrl, max.evals = 25L)

design.mat = generateRandomDesign(n = 200, par.set = par)
ctrl = makeTuneControlMBO(mbo.control = mbo.ctrl, mbo.design = design.mat)
parallelStartMulticore(cpus = 4L)
tune.pars = tuneParams(learner = lrn, task = tsk, resampling = res, measures = rmsle, par.set = par, control = ctrl)
parallelStop()
tune.pars$x
tune.pars$y
# I got max_depth = 4, min_child_weight = 1 plug in your eta, nrounds, max_depth, and min_child_weight below

# Third Tuning / Gamma
lrn = makeLearner("regr.xgboost", nthread = 4)
lrn = setHyperPars(lrn,
                   nrounds = 63,
                   eta =  0.1477969,
                   max_depth = 5,
                   min_child_weight = 1,
                   subsample = .8,
                   colsample_bytree = .8,
                   objective = 'reg:linear'
)
res = makeResampleDesc("CV", iters = 4L)
par = makeParamSet(
  makeNumericParam(id = "gamma", lower = 0, upper = 0.5)
)

mbo.ctrl = makeMBOControl()
mbo.ctrl = setMBOControlInfill(mbo.ctrl, crit = crit.ei)
mbo.ctrl = setMBOControlTermination(mbo.ctrl, max.evals = 25L)

design.mat = generateRandomDesign(n = 200, par.set = par)
ctrl = makeTuneControlMBO(mbo.control = mbo.ctrl, mbo.design = design.mat)
parallelStartMulticore(cpus = 4L)
tune.pars = tuneParams(learner = lrn, task = tsk, resampling = res,
                       measures = rmsle, par.set = par, control = ctrl)
parallelStop()
tune.pars$x
tune.pars$y
# I got gamm = 0.401946. Again, move everything you've tuned down to the next one.

# Fourth Tuning / subsample, colsample_bytree. Normally start between 0.6 and 0.9.
lrn = makeLearner("regr.xgboost", nthread = 4)
lrn = setHyperPars(lrn,
                   nrounds = 63,
                   eta =  0.1477969,
                   max_depth = 5,
                   min_child_weight = 1,
                   gamma = 0.401946,
                   objective = 'reg:linear'
)
res = makeResampleDesc("CV", iters = 5L)
par = makeParamSet(
  makeNumericParam(id = "subsample", lower = 0.5, upper = 0.7),
  makeNumericParam(id = "colsample_bytree", lower = 0.4, upper = 0.7)
)

mbo.ctrl = makeMBOControl()
mbo.ctrl = setMBOControlInfill(mbo.ctrl, crit = crit.ei)
mbo.ctrl = setMBOControlTermination(mbo.ctrl, max.evals = 25L)

design.mat = generateRandomDesign(n = 200, par.set = par)
ctrl = makeTuneControlMBO(mbo.control = mbo.ctrl, mbo.design = design.mat)
parallelStartMulticore(cpus = 4L)
tune.pars = tuneParams(learner = lrn, task = tsk, resampling = res,
                       measures = rmsle, par.set = par, control = ctrl)
parallelStop()
tune.pars$x
tune.pars$y
# subsample = 0.6011694, colsample_bytree = 0.3831516

train <- data.frame(train); train <- createDummyFeatures(train)
test <- data.frame(test); test <- createDummyFeatures(test)
train <- train %>% select(-SalePrice);

# Train and test final model / make nrounds real big and make eta somewhere between .005 and .01
train[] <- lapply(train, as.numeric)
test[]<-lapply(test, as.numeric)
dtrain=xgb.DMatrix(as.matrix(train),label= y_train)
dtest=xgb.DMatrix(as.matrix(test))

# xgboost parameters
xgb_params = list(
  seed = 0,
  eta = 0.01,
  max_depth = 5,
  min_child_weight = 1,
  gamma = 0.401946,
  subsample = 0.6011694,
  colsample_bytree = 0.3831516,
  objective = 'reg:linear'
)

# Cross-Validate
res = xgb.cv(xgb_params,
             dtrain,
             nrounds=10000,
             nfold=5,
             early_stopping_rounds=40,
             print_every_n = 10,
             verbose= 1,
             metric = list('rmse'),
             maximize=FALSE)

# Train and predict

best_n_rounds = which.min(res$evaluation_log$test_rmse_mean)
gb_dt=xgb.train(xgb_params, dtrain, nrounds = as.integer(best_n_rounds))
submission=fread("sample_submission.csv",colClasses = c("integer","numeric"))
submission$SalePrice=predict(gb_dt,dtest)
write.csv(submission,"xgb.csv",row.names = FALSE)

# 0.12292 on Kaggle with 1743 nrounds and .01 eta was our best score