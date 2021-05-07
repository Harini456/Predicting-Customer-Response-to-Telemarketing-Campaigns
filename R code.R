#Main Objective: increase the effectiveness of the bank's telemarketing campaign
#This project will enable the bank to develop a more granular understanding of its customer base, predict customers' response to its telemarketing campaign and establish a target customer profile for future marketing plans.
#By analyzing customer features, such as demographics and transaction history, the bank will be able to predict customer saving behaviours and identify which type of customers is more likely to make term deposits. 
#The bank can then focus its marketing efforts on those customers. 
#This will not only allow the bank to secure deposits more effectively but also increase customer satisfaction by reducing undesirable advertisements for certain customers.

#The dataset contains numerical and categorical varibale.
#Categorical Variable :
#* Marital - (Married , Single , Divorced)",
#* Job - (Management,BlueCollar,Technician,entrepreneur,retired,admin.,services,selfemployed,housemaid,student,unemployed,unknown)
#* Contact - (Telephone,Cellular,Unknown)
#* Education - (Primary,Secondary,Tertiary,Unknown)
#* Month - (Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec)
#* Poutcome - (Success,Failure,Other,Unknown)
#* Housing - (Yes/No)
#* Loan - (Yes/No)
#* deposit - (Yes/No)
#* Default - (Yes/No)
#Numerical Variable:
#*  Age
#* Balance
#* Day
#* Duration
#* Campaign
#* Pdays
#* Previous
#install.packages("kableExtra")
library(Hmisc)      #data description
library(tidyverse)  # data manipulation
library(data.table) # fast file reading
library(caret)      # rocr analysis
library(ROCR)       # rocr analysis
library(kableExtra) # nice table html formating 
library(gridExtra)  # arranging ggplot in grid
library(rpart)      # decision tree
library(rpart.plot) # decision tree plotting
library(caTools)    # split 
library(car)

#Read Data
df=read.csv("D:/Harini(christ unniversity)/2nd sem subjects/Machine Learning/bank.csv")
head(df)
class(df)
typeof(df)
dim(df)#Dimension of the dataset
summary(df)#Statistical Summary

#Data Validation
sum(duplicated(df))#Check for Duplicate Rows
sum(!complete.cases(df))#Check for Missing Data

str(df)#structure of the data
describe(df)#data description

sapply(df, class)#datatypes of each variable.
df=transform(df, deposit=as.factor(deposit),job=as.factor(job),marital=as.factor(marital),education=as.factor(education),default=as.factor(default),housing=as.factor(housing),loan=as.factor(loan),contact=as.factor(contact),month=as.factor(month),poutcome=as.factor(poutcome))#chaning the datatype for required columns to factor.
prop.table(table(df$deposit))#Outcome Imbalance

#Data Exploration
boxplot(df, main = "BoxPlot",notch = TRUE, col = 1:17)

#Age Distribution vs Marital Status That Subscribes Term Deposit
ggplot(df, aes(x=age, fill=marital)) + geom_histogram(binwidth = 2, alpha=0.7) +facet_grid(cols = vars(deposit)) +expand_limits(x=c(0,100)) +scale_x_continuous(breaks = seq(0,100,10)) +ggtitle("Age Distribution by Marital Status")
#The bulk of clients are married or divorced. Sharp drop of clients above age 60 with marital status ‘divorced’ and ‘married’. *Single clients drop in numbers above age 40.

#Education vs Subscription
ggplot(data = df, aes(x=education, fill=deposit)) +geom_bar() +ggtitle("Term Deposit Subscription based on Education Level") +xlab(" Education Level") +guides(fill=guide_legend(title="Subscription of Term Deposit"))
#Having higher education is seen to contribute to higher subscription of term deposit. Most clients who subscribe are from ‘secondary’ and ‘tertiary’ education levels. Tertiary educated clients have higher rate of subscription (15%) from total clients called.

#Subscription based on Number of Contact during Campaign
ggplot(data=df, aes(x=campaign, fill=deposit))+geom_histogram()+ggtitle("Subscription based on Number of Contact during the Campaign")+xlab("Number of Contact during the Campaign")+xlim(c(min=1,max=30)) +guides(fill=guide_legend(title="Subscription of Term Deposit"))
#It can be observed from barchart that there will be no subscription beyond 7 contact during the campaign. Future campaign could improve resource utilization by setting limits to contacts during a campaign. Future campaigns can focus on first 3 contacts as it will have higher subscription rate.

#Scatterplot of Duration by Age
ggplot(df,aes(age, duration)) +geom_point() +facet_grid(cols = vars(deposit)) +scale_x_continuous(breaks = seq(0,100,10)) +ggtitle("Scatterplot of Duration vs Age for Subscription of Term Deposit")
#Less clients after age of 60. Duration during call looks similar.

ggplot (df, aes(x=balance)) + 
  geom_histogram(color = "blue", fill = "blue") +
  facet_grid(cols=vars(deposit)) + 
  ggtitle('Balance Histogram') + ylab('Count') + xlab('Balance') +
  geom_vline(data=df, aes(xintercept=mean(balance)), color="red", linetype="dashed")
ggplot (df, aes(x=age)) + 
  geom_histogram(color = "blue", fill = "blue", binwidth = 5) +
  facet_grid(cols=vars(y)) + 
  ggtitle('Age Distribution by Subscription') + ylab('Count') + xlab('Age') +
  scale_x_continuous(breaks = seq(0,100,5)) +
  geom_vline(data=df, aes(xintercept=mean(age)), color="red", linetype="dashed")


#Scatterplot Matrix
pairs(~duration+month+day+balance,data=df)
pairs(~balance+housing+loan+campaign,data=df)
#Due to large number of attributes (17 total), 8 was chosen for correlation. No clear correlation pattern can be observed as most attributes are categorical.
df
#Split the Training / Testing data and Scale
# split into training and testing
set.seed(123)
split = sample.split(df$deposit,SplitRatio = 0.70)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)
dim(training_set)
dim(test_set)


#Model 1 - random forest
library("randomForest")
varNames = names(df)
varNames
varNames = varNames[!varNames %in% c("deposit")]
varNames
varNames1 = paste(varNames, collapse = "+")
varNames1
rf.form = as.formula(paste("deposit", varNames1, sep = " ~ "))
rf.form
BankMarketing.rf = randomForest(rf.form,training_set,ntree=500,importance=T)
BankMarketing.rf
plot(BankMarketing.rf, main="Random Forest")
varImpPlot(BankMarketing.rf,sort = T,main = "Variable Importance",n.var = 5)
var.imp = data.frame(importance(BankMarketing.rf,type = 2))
var.imp$Variables = row.names(var.imp)
var.imp[order(var.imp$MeanDecreaseGini, decreasing = T),]
predicted.response = predict(BankMarketing.rf, test_set)
confusionMatrix(data = test_set$predicted.response,reference = test_set$deposit,positive = 'yes')


#SVM
library(e1071)
classifier = svm(formula = deposit ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')
y_pred = predict(classifier, newdata = test_set[-17])
cm = table(test_set[, 17], y_pred)
cm
bestmodal.tune<-classifier$best.model
summary(bestmodal.tune)
train.svm<-svm(deposit~.,training_set,kernel="polynomial",cost=0.01,scale=TRUE,degree=3,gamma=1)
summary(train.svm)
plot(train.svm,training_set) 
test.svm<-predict(train.svm,test_set)
table(predict=test.svm,truth=test_set$deposit)
confusionMatrix(data = test.svm,reference = test_set$deposit,positive = 'yes')

#Model3: Logistic regression
classifier.lm = glm(formula = deposit ~ .,
                    family = binomial,
                    data = training_set)
pred_lm = predict(classifier.lm, type='response', newdata=test_set[-17])

# plot the prediction distribution
predictions_LR <- data.frame(y = test_set$deposit, pred = NA)
predictions_LR$pred <- pred_lm
plot_pred_type_distribution(predictions_LR,0.30)

# choose the best threshold as 0.30
test.eval.LR = binclass_eval(test_set[, 17], pred_lm > 0.30)

# Making the Confusion Matrix
test.eval.LR$cm




