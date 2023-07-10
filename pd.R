library(readxl)
library(ggplot2)
library(boot)
library(GGally)
library(car)
library(glmnet)
library(tictoc)
library(caret)
library(dplyr)
library(broom)
library(MASS)
library(class)
library(klaR)
library(tree)
library(randomForest)
library(gbm)
library(pls)
library(datasets)
library(caTools)
library(party)
library(dplyr)
library(magrittr)

# EDA
data=read_excel("C:/hello/parkinson disease/Parkinsson disease_xl.xls")
head(data)
tail(data)
summary(data)
print (data)

# DATA CLEANING
which(is.na(data)==TRUE)
data <- data[,-1]
print(data)

str(head(data, n=0))

# DATA VISUALIZATION --- PLOTTING GRAPH 
tbl <- as.vector(c(with(data, table(status)), as.numeric(nrow(data))))
bp <- barplot(tbl, beside = TRUE, names.arg = c("Healthy", "PDP", "Total"),
              ylab="Frequency", xlab="Status", ylim= c(0,max(tbl)+50),
              main="Response-Target Distribution" ,col=c("green","red","grey"))
text(x = bp, y = tbl, label = tbl,
     pos = 3, cex = 0.8, col = c("green","red","black"))

set.seed(2645)

data$status = as.factor(data$status)
# for (x in seq(from=1, to=ncol(data[,-17])-4, by=4)){
#   ggp = ggpairs(data[,-17][,c(seq(from=x, to=x+4))],
#                 aes(color=data$status), progress=ggmatrix_progress())
#   print(ggp)
# }

set.seed(2645)

# SPLITTING DATA INTO TRAINING AND TESTING DATA SET
training_indices <- createDataPartition(data$status, p=0.8, list=FALSE)

x <- model.matrix(status~.,data)[,-1]
y <- data$status

x.train <- data[training_indices,]
y.train <- data$status[training_indices]

x.test  <- data[-training_indices,]
g <-  data[-training_indices,]
y.test  <- data$status[-training_indices]


print("MULTIPLE LOGISTIC REGRESSION")
set.seed(2645)
# Attach the DataSet:

attach(data)
# Performing Multiple Logistic Regression, Family = Binomial to indicate classification

glm.fit <- glm(status~., data=data, subset=training_indices, family="binomial")


glm.prediction <- predict(glm.fit, newdata = x.test, type="response")
glm.prediction
table_mat <- table(data$status[-training_indices], glm.prediction > 0.5)
colnames(table_mat) <- c("0","1")
table_mat

predic <- ifelse(glm.prediction>0.5,1,0)
predic
vsa.test.error.glm <- mean(predic!=data$status[-training_indices])
vsa.test.error.glm


set.seed(2645)
cv.error <- cv.glm(x.train, glm.fit, K=5)$delta

cv.error

acc <- 100 - (cv.error[1]*100)
acc

# # LOGISTIC DISCRIMINANT ANALYSIS(LDA) : 82%
# set.seed(2645)
# folds <- createFolds(data$status, k=5, returnTrain = TRUE)
# d.train <- data[training_indices,]
# ce_vector <- vector()
# for (fold in folds){
#   train = fold
#   lda.fit <- lda(d.train$status~., data=d.train[,-5], subset=train, family=binomial)
#   lda.prediction <- predict(lda.fit, newdata=d.train[-train,-5])
#   fit.pred <- ifelse(lda.prediction>0.5, 1, 0)
#   c.e = mean(fit.pred != y.test)
#   ce_vector <- c(ce_vector,c.e)
# }
# mce <- mean(ce_vector)
# mce
# acc_lda <- 100 - mce
# print(paste('Accuracy for test is found to be', acc_lda))

# # QUADRATIC DISCRIMINANT ANALYSIS
# set.seed(2645)
# qda.fit <- qda(data$status~., data=data, subset=training_indices, family="binomial", cv=TRUE)
# qda.prediction <- predict(qda.fit, newdata=d.train[-training_indices,])
# fit.pred <- ifelse(qda.prediction>0.5, 1, 0)
# c.e = mean(fit.pred != y.test)
# c.e
# acc_qda <- 100 - mce
# print(paste('Accuracy for test is found to be', acc_qda))

# DECISION TREE
set.seed(230)
print("DECISON TREE ALGORITHM")
model<- ctree(status ~ .,data = x.test)
# plot(model)

predict_model<-predict(model, x.test)

m_at <- table(x.test$status, predict_model)
confusionMatrix(table(predict_model, x.test$status))

ac_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test is found to be', ac_Test))

# SUPPORT VECTOR MACHINE
ctrl <- trainControl(method = "cv", verboseIter = FALSE, number = 5)

grid_svm <- expand.grid(C = c(0.01, 0.1, 1, 10, 20))

tic(msg= "Total time for SVM is ")
svm_fit <- train(status ~ .,data = x.train,
                 method = "svmLinear", preProcess = c("center","scale"),
                 tuneGrid = grid_svm, trControl = ctrl)


s=plot(svm_fit)

print(s)
toc()
svm_predict <- predict(svm_fit, newdata = x.test)
svm_predict
svm_results <- table_mat <- table(data$status[-training_indices], svm_predict > 0.5)

print(svm_results)
# 
# y <- 100*mean(svm_predict==g[,1])
# print(paste('Accuracy for test is found to be ',y ))
confusionMatrix(table(svm_predict, x.test$status))


# RANDOM FOREST
set.seed(120)
classifier_RF = randomForest(x = x.train,
                             y = x.train$status,
                             ntree = 500)

classifier_RF

y_pred = predict(classifier_RF, newdata = x.test)

# CONFUSION MATRIX
table_mat <- table(data$status[-training_indices], y_pred > 0.5)
colnames(table_mat) <- c("0","1")
table_mat