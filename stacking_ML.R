##-------------------GARY HUTSON -STACKED ML MODELS - NHS R COMMUNITY ----------------------------
#-------------------------- Please seek permission from code owner before reuse ------------------

#Stacking function
library(mlbench)
library(caret)
library(caretEnsemble)

# Load the dataset
data(Ionosphere)
dataset <- Ionosphere
dataset <- dataset[,-2]
dataset$V1 <- as.numeric(as.character(dataset$V1))
typeof(dataset)
class(dataset)

#Generate stacking

summary(dataset)

#Stacking model

# Example of Stacking algorithms

seed <- 123
train_ctl <- trainControl(method="repeatedcv", number=10, repeats=3,
                             savePredictions = TRUE, classProbs=TRUE)
algo_list <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial', 'nb')
set.seed(seed)
class_name <- "Class"
models <- caretList(as.formula(paste(class_name, "~ .")), 
                    data=dataset, trControl=train_ctl, methodList=algo_list)
models
results <- resamples(models)
summary(results)
dotplot(results)


#See how the models perform on their own

# stack using random forest as the second order model
stack_ctl <- trainControl(method="repeatedcv", number=10, repeats=3,
                             savePredictions=TRUE, classProbs=TRUE)
#Methods are boot, boot 362, optimism boot, boot_all, cv, repeatedcv, LOOCV, LGOCV


set.seed(seed)
rand_forest.stack <- caretStack(models, method="rf", metric="Accuracy", trControl=stack_ctl)

#Inspect elements of model
rand_forest.stack$models
rand_forest.stack$ens_model$finalModel$confusion
rand_forest.stack$ens_model$results$mtry
rand_forest.stack$ens_model$results$Accuracy
rand_forest.stack$ens_model$bestTune$mtry
rand_forest.stack$ens_model$resample
rand_forest.stack$ens_model$times$everything # Performance information re the random forest model utilised
rand_forest.stack$ens_model$pred # Returns top level predictions from the random forest 
rand_forest.stack$ens_model$perfNames
rand_forest.stack$ens_model$control
rand_forest.stack$ens_model$preProcess
rand_forest.stack$ens_model$maximize




      
