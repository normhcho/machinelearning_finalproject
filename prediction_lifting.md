**Predicting the Weightlifting Activity**
=========================================

Norm Cho

May 23, 2018

**Background**

The objective is to capture how well people are performing certain
physical activities. In this example, data was gathered from
accelerometers on the belt, forearm, arm, and dumbbell of 6
participants. They were asked to perform barbell lifts correctly and
incorrectly in 5 different ways. The goal is to create a model that can
accurately predict how the unilateral dumbbell bicep curl was executed
among 6 males aged 20 to 28 who are inexperienced in weightlifting. See
below for the class definitions:

-   Class A: exactly according to the specification
-   Class B: throwing the elbows to the front
-   Class C: lifting the dumbbell only halfway
-   Class D: lowering the dumbbell only halfway
-   Class E: throwing the hips to the front

**Cross-Validation**

Even though there are training and test sets, the 20 record "testing"
set seems to be more of a validation set. The "training" data will be
split into 70% training and 30% testing.

**Model Selection**

There are many model types to choose. Because there is final test set of
20 records, the premium will be placed on accuracy. Thus, the plan is
start with a random forest model, but if the model is not accurate
enough, I would consider other methods such as decision tree and
boosting, or even blending multiple models types together if necessary.

**Variables**

It is important is form a strategy before writing a single line of code.

With 158 potential variables, there seems to be a high likelihood that
the accuracy would be very high if the model without any adjustments.
However, there could be an overfitting problem and it could take a very
long time to run.

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    dat <- read.csv("pml-training.csv")
    dat_test <- read.csv("pml-testing.csv")
    inTrain <- createDataPartition(y=dat$classe, p=0.7, list=FALSE)
    training <- dat[inTrain,]
    testing <- dat[-inTrain,]
    summary(training)
    str(training)
    dim(training)
    dim(testing)
    set.seed(410)  #homage to the Baltimore area code

The point is to preview the data. Given the limitations regarding the
number of figures that can be shown, the results are hidden. There are
are 13,737 records in the training data and 5,885 in the testing data.

**Random Forest, Version \#1**

Therefore, for the first iteration of the model, I looked at the 20
record pml-testing file and looked at which columns had data. For the
columns without data, they were culled. As a result, the number of
columns was reduced to ~55 variables.

    modFit_rf <- train(classe ~ (user_name+num_window+roll_belt+pitch_belt+yaw_belt+total_accel_belt+gyros_belt_x+gyros_belt_y+gyros_belt_z+accel_belt_x+accel_belt_y+accel_belt_z+magnet_belt_x+magnet_belt_y+magnet_belt_z+roll_arm+pitch_arm+yaw_arm+total_accel_arm+gyros_arm_x+gyros_arm_y+gyros_arm_z+accel_arm_x +accel_arm_y+accel_arm_z+magnet_arm_x+magnet_arm_y+magnet_arm_z+roll_dumbbell +pitch_dumbbell+yaw_dumbbell+total_accel_dumbbell+gyros_dumbbell_x+gyros_dumbbell_y+gyros_dumbbell_z+accel_dumbbell_x+accel_dumbbell_y+accel_dumbbell_z+magnet_dumbbell_x+magnet_dumbbell_y+magnet_dumbbell_z+roll_forearm+pitch_forearm+yaw_forearm+total_accel_forearm+gyros_forearm_x+gyros_forearm_y+gyros_forearm_z+accel_forearm_x+accel_forearm_y+accel_forearm_z+magnet_forearm_x+magnet_forearm_y+magnet_forearm_z),method="rf", data=training)

    predict_rf <- predict(modFit_rf,testing)
    confusionMatrix(predict_rf,testing$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1671    5    0    0    0
    ##          B    2 1133    0    0    2
    ##          C    0    0 1026    2    0
    ##          D    0    1    0  962    0
    ##          E    1    0    0    0 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9978          
    ##                  95% CI : (0.9962, 0.9988)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9972          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9947   1.0000   0.9979   0.9982
    ## Specificity            0.9988   0.9992   0.9996   0.9998   0.9998
    ## Pos Pred Value         0.9970   0.9965   0.9981   0.9990   0.9991
    ## Neg Pred Value         0.9993   0.9987   1.0000   0.9996   0.9996
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2839   0.1925   0.1743   0.1635   0.1835
    ## Detection Prevalence   0.2848   0.1932   0.1747   0.1636   0.1837
    ## Balanced Accuracy      0.9985   0.9969   0.9998   0.9989   0.9990

This was run with 55 variables, and the accuracy rate at 99.8% was more
than enough but given that it can take a long time to run, I wanted to
see if the number of variables can be reduced to something more
manageable. I kept the user name because I wasn't sure whether different
measurement apply to separate people in the different ways.

**Random Forest, Version \#2**

This version is reduced to 17 variables, removing the user name, and
anything measuring in the x, y, or z direction. For examples, variables
such as gyros\_arm\_x and accel\_forearm\_z were removed. While it would
take less time to run, how would it impact the accuracy rate?

    modfit_rf2 <- train(classe ~ (num_window+roll_belt+pitch_belt+yaw_belt+total_accel_belt+roll_arm+pitch_arm+yaw_arm+total_accel_arm+roll_dumbbell+pitch_dumbbell+yaw_dumbbell+total_accel_dumbbell+roll_forearm+pitch_forearm+yaw_forearm+total_accel_forearm),method="rf", data=training)

    predict_rf2 <- predict(modfit_rf2,testing)
    confusionMatrix(predict_rf2,testing$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    2    0    0    0
    ##          B    1 1136    1    0    1
    ##          C    0    1 1025    2    1
    ##          D    0    0    0  962    4
    ##          E    0    0    0    0 1076
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9978          
    ##                  95% CI : (0.9962, 0.9988)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9972          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9974   0.9990   0.9979   0.9945
    ## Specificity            0.9995   0.9994   0.9992   0.9992   1.0000
    ## Pos Pred Value         0.9988   0.9974   0.9961   0.9959   1.0000
    ## Neg Pred Value         0.9998   0.9994   0.9998   0.9996   0.9988
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1930   0.1742   0.1635   0.1828
    ## Detection Prevalence   0.2846   0.1935   0.1749   0.1641   0.1828
    ## Balanced Accuracy      0.9995   0.9984   0.9991   0.9986   0.9972

It turned out that it took much less time to run and the accuracy rate
actually improved to 99.95%. The plan is to fine tune the model with
fewer variables, but otherwise I am comfortable using this vs. the 20
observation quiz. The only possible is that the total variables (eg
total arm) consists of arm metrics, meaning there are also related.

**Random Forest, Version \#3**

As stated before, because the total acceleration columns are likely a
function of their respective inputs, this version will exclude those
variables.

    modfit_rf3 <- train(classe ~ (num_window+roll_belt+pitch_belt+yaw_belt+roll_arm+pitch_arm+yaw_arm+roll_dumbbell+pitch_dumbbell+yaw_dumbbell+roll_forearm+pitch_forearm+yaw_forearm),method="rf", data=training)

    predict_rf3 <- predict(modfit_rf3,testing)
    confusionMatrix(predict_rf3,testing$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    2    0    0    0
    ##          B    1 1137    2    0    2
    ##          C    0    0 1023    0    2
    ##          D    0    0    0  964    1
    ##          E    0    0    1    0 1077
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9981          
    ##                  95% CI : (0.9967, 0.9991)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9976          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9982   0.9971   1.0000   0.9954
    ## Specificity            0.9995   0.9989   0.9996   0.9998   0.9998
    ## Pos Pred Value         0.9988   0.9956   0.9980   0.9990   0.9991
    ## Neg Pred Value         0.9998   0.9996   0.9994   1.0000   0.9990
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1932   0.1738   0.1638   0.1830
    ## Detection Prevalence   0.2846   0.1941   0.1742   0.1640   0.1832
    ## Balanced Accuracy      0.9995   0.9986   0.9983   0.9999   0.9976

The accuracy rate improved to 99.96% with the 13 variable model. While
there is some worry with overfitting, but hopefully that is mitigated
with the relatively low number of variables. This is the model I plan to
use.

**Error Analysis**

With such a high accuracy rate at 99.96%, other measure error metrics
such as sensitivity and specificity, positive predictive value and even
kappa (concordance) are above 99.5%.

**Decision Tree**

    modFit_dt <- train(classe ~ (user_name+num_window+roll_belt+pitch_belt+yaw_belt+total_accel_belt+gyros_belt_x+gyros_belt_y+gyros_belt_z+accel_belt_x+accel_belt_y+accel_belt_z+magnet_belt_x+magnet_belt_y+magnet_belt_z+roll_arm+pitch_arm+yaw_arm+total_accel_arm+gyros_arm_x+gyros_arm_y+gyros_arm_z+accel_arm_x +accel_arm_y+accel_arm_z+magnet_arm_x+magnet_arm_y+magnet_arm_z+roll_dumbbell +pitch_dumbbell+yaw_dumbbell+total_accel_dumbbell+gyros_dumbbell_x+gyros_dumbbell_y+gyros_dumbbell_z+accel_dumbbell_x+accel_dumbbell_y+accel_dumbbell_z+magnet_dumbbell_x+magnet_dumbbell_y+magnet_dumbbell_z+roll_forearm+pitch_forearm+yaw_forearm+total_accel_forearm+gyros_forearm_x+gyros_forearm_y+gyros_forearm_z+accel_forearm_x+accel_forearm_y+accel_forearm_z+magnet_forearm_x+magnet_forearm_y+magnet_forearm_z),method="rpart",data=training)

    print(modFit_dt$finalModel)

    ## n= 13737 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
    ##    2) roll_belt< 130.5 12554 8658 A (0.31 0.21 0.19 0.18 0.11)  
    ##      4) pitch_forearm< -33.95 1092    7 A (0.99 0.0064 0 0 0) *
    ##      5) pitch_forearm>=-33.95 11462 8651 A (0.25 0.23 0.21 0.2 0.12)  
    ##       10) magnet_dumbbell_y< 438.5 9681 6920 A (0.29 0.18 0.24 0.19 0.11)  
    ##         20) roll_forearm< 123.5 6062 3583 A (0.41 0.18 0.18 0.17 0.061) *
    ##         21) roll_forearm>=123.5 3619 2408 C (0.078 0.18 0.33 0.23 0.18) *
    ##       11) magnet_dumbbell_y>=438.5 1781  868 B (0.028 0.51 0.044 0.23 0.19) *
    ##    3) roll_belt>=130.5 1183   10 E (0.0085 0 0 0 0.99) *

    predict_dt <- predict(modFit_dt,testing)
    confusionMatrix(predict_dt, testing$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1512  473  479  435  154
    ##          B   37  380   31  164  158
    ##          C  121  286  516  365  312
    ##          D    0    0    0    0    0
    ##          E    4    0    0    0  458
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.487           
    ##                  95% CI : (0.4742, 0.4999)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3297          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9032  0.33363  0.50292   0.0000  0.42329
    ## Specificity            0.6341  0.91783  0.77691   1.0000  0.99917
    ## Pos Pred Value         0.4953  0.49351  0.32250      NaN  0.99134
    ## Neg Pred Value         0.9428  0.85161  0.88098   0.8362  0.88493
    ## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
    ## Detection Rate         0.2569  0.06457  0.08768   0.0000  0.07782
    ## Detection Prevalence   0.5188  0.13084  0.27188   0.0000  0.07850
    ## Balanced Accuracy      0.7686  0.62573  0.63992   0.5000  0.71123

With the accuracy rate around 50%, it is clear that at least this
version of the decision tree model should not implemented.

**Boosting**

Aside from using the random forest and decision, I was again curious if
boosting is a feasible model using the 13 variables from Random Forest
Model \#3.

    modfit_gbm <- train(classe ~ (num_window+roll_belt+pitch_belt+yaw_belt+roll_arm+pitch_arm+yaw_arm+roll_dumbbell+pitch_dumbbell+yaw_dumbbell+roll_forearm+pitch_forearm+yaw_forearm),method="gbm", data=training)

    predict_gbm <- predict(modfit_gbm,testing)
    confusionMatrix(predict_gbm, testing$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    4    0    0    3
    ##          B    1 1133    9    0    0
    ##          C    0    2 1017    3    2
    ##          D    0    0    0  961   10
    ##          E    0    0    0    0 1067
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9942         
    ##                  95% CI : (0.9919, 0.996)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9927         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9947   0.9912   0.9969   0.9861
    ## Specificity            0.9983   0.9979   0.9986   0.9980   1.0000
    ## Pos Pred Value         0.9958   0.9913   0.9932   0.9897   1.0000
    ## Neg Pred Value         0.9998   0.9987   0.9981   0.9994   0.9969
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1925   0.1728   0.1633   0.1813
    ## Detection Prevalence   0.2855   0.1942   0.1740   0.1650   0.1813
    ## Balanced Accuracy      0.9989   0.9963   0.9949   0.9974   0.9931

Although I still plan on using the random forest, the boosting model is
viable with its high accuracy rate at 99.2%

**Prediction**

Predicting the results of the final test using both the Random Forest
\#3 and boosting models.

    predict(modfit_rf3,dat_test)

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

    predict(modfit_gbm,dat_test)

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Both models show the same results, which gives me more confidence that
the models will yield the correct answers.
