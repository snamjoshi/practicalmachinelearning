# Coursera Data Science Specialization Course 8 Course Project
Sanjeev V Namjoshi  
December 21, 2015  

## Introduction

In this assignment, we examine data from various personal activity trackers ("wearable devices"). The data comes from a study where participants used these devices to track their movements. According to the study authors, "Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions". The following specification classes were used:

- Exactly according to the specification (Class A)
- Throwing the elbows to the front (Class B)
- Lifting the dumbbell only halfway (Class C)
- Lowering the dumbbell only halfway (Class D)
- Throwing the hips to the front (Class E)

Only Class A has been performed correctly. The aim of this project is to use the accelerometer data for each of the six participants to see if it can be used to predict which of the following exercise classes they were asked to perform. If the model can successfully predict the correct class based on the data, we can conclude with reasonable certainty that the wearable devices are providing an accurate measure of movement tracking.

For more information on the data and the study please see: http://groupware.les.inf.puc-rio.br/har

## Load packages and set seed


```r
library(caret)

set.seed(59493)
```

## Examine and load data

The testing and training sets can be accessed at the following URLs:


```r
testingURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainingURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
```

A first look at the data reveals that there are numerous blanks in certain columns and cells filled with the text "#DIV/0!". To transform this data into a shape that R is better equipped at handling, we use the `read.csv()` function to ensure that the cells without date are coded as `NA`.


```r
testing <- read.csv(url(testingURL), header = TRUE, na.string = c("", "NA", "#DIV/0!"))
training <- read.csv(url(trainingURL), header = TRUE, na.string = c("", "NA", "#DIV/0!"))
```

## Slice the data

We would like to performed supervised learning for our model so we need to slice our training set into a training set and a testing set. This will allow us to perform cross-validation later to verify the accuracy of our model before we test in on the actual test set. We will split the training set into 60% training and 40% testing.


```r
inTrain <- createDataPartition(y = training$classe, p = 0.6, list = FALSE)

splitTest <- training[-inTrain, ]
splitTrain <- training[inTrain, ]
```

## Transform the data

First let's take a look at the structure of the sliced training data.


```r
str(splitTrain)
```

```
## 'data.frame':	11776 obs. of  160 variables:
##  $ X                       : int  1 3 6 7 12 14 15 17 18 20 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 820366 304277 368296 528316 576390 604281 692324 732306 788335 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 12 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.42 1.45 1.42 1.43 1.42 1.45 1.51 1.55 1.59 ...
##  $ pitch_belt              : num  8.07 8.07 8.06 8.09 8.18 8.21 8.2 8.12 8.08 8.07 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0 0.02 0.02 0.02 0.02 0 0 0 0.02 ...
##  $ gyros_belt_y            : num  0 0 0 0 0 0 0 0 0.02 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 0 -0.02 0 -0.02 ...
##  $ accel_belt_x            : int  -21 -20 -21 -22 -22 -22 -21 -21 -21 -22 ...
##  $ accel_belt_y            : int  4 5 4 3 2 4 2 4 5 5 ...
##  $ accel_belt_z            : int  22 23 21 21 23 21 22 22 21 22 ...
##  $ magnet_belt_x           : int  -3 -2 0 -4 -2 -8 -1 -6 1 -1 ...
##  $ magnet_belt_y           : int  599 600 603 599 602 598 597 598 600 604 ...
##  $ magnet_belt_z           : int  -313 -305 -312 -311 -319 -310 -310 -317 -316 -314 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -129 -129 -129 -129 ...
##  $ pitch_arm               : num  22.5 22.5 22 21.9 21.5 21.4 21.4 21.3 21.2 21.1 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.03 -0.03 -0.03 0 0 0 -0.02 -0.02 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 0 0 0 -0.03 -0.03 -0.02 -0.03 -0.02 ...
##  $ accel_arm_x             : int  -288 -289 -289 -289 -288 -288 -289 -289 -288 -289 ...
##  $ accel_arm_y             : int  109 110 111 111 111 111 111 110 108 109 ...
##  $ accel_arm_z             : int  -123 -126 -122 -125 -123 -124 -124 -122 -124 -125 ...
##  $ magnet_arm_x            : int  -368 -368 -369 -373 -363 -371 -374 -371 -373 -373 ...
##  $ magnet_arm_y            : int  337 344 342 336 343 331 342 337 336 335 ...
##  $ magnet_arm_z            : int  516 513 513 509 520 523 510 512 510 514 ...
##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 12.9 13.4 13.1 13.1 ...
##  $ pitch_dumbbell          : num  -70.5 -70.3 -70.8 -70.2 -70.5 ...
##  $ yaw_dumbbell            : num  -84.9 -85.1 -84.5 -85.1 -84.9 ...
##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

We can draw two conclusions from this:

1. The first seven columns can be removed because they are unlikely to contribute to our predictive model. These columns include the user's name and timestamps.
2. Many columns appear to be filled with `NA`. This requires further investigation.

First, let's remove the first seven columns.


```r
splitTrain <- splitTrain[ ,-c(1:7)]   
```

### Process columns containing `NA`

After determining the number and proportion of columns containing `NA`, we will omit all columns that contain at least 50% `NA`. As we can see from the data below, all the `NA` columns contain at least 90% `NA`. It is unlikely that any of these columns will contain enough useful information to contribute to the model when compared with the many other measurements for which we do have values.


```r
# Determine the number of NA per column
naPerColumn <- sapply(splitTrain, function(x) sum(length(which(is.na(x)))))

# Determine proportion of NA per total observations
naPerColumn <- data.frame(naPerColumn) / nrow(splitTrain)
print(naPerColumn)
```

```
##                          naPerColumn
## roll_belt                  0.0000000
## pitch_belt                 0.0000000
## yaw_belt                   0.0000000
## total_accel_belt           0.0000000
## kurtosis_roll_belt         0.9802989
## kurtosis_picth_belt        0.9814029
## kurtosis_yaw_belt          1.0000000
## skewness_roll_belt         0.9802140
## skewness_roll_belt.1       0.9814029
## skewness_yaw_belt          1.0000000
## max_roll_belt              0.9797894
## max_picth_belt             0.9797894
## max_yaw_belt               0.9802989
## min_roll_belt              0.9797894
## min_pitch_belt             0.9797894
## min_yaw_belt               0.9802989
## amplitude_roll_belt        0.9797894
## amplitude_pitch_belt       0.9797894
## amplitude_yaw_belt         0.9802989
## var_total_accel_belt       0.9797894
## avg_roll_belt              0.9797894
## stddev_roll_belt           0.9797894
## var_roll_belt              0.9797894
## avg_pitch_belt             0.9797894
## stddev_pitch_belt          0.9797894
## var_pitch_belt             0.9797894
## avg_yaw_belt               0.9797894
## stddev_yaw_belt            0.9797894
## var_yaw_belt               0.9797894
## gyros_belt_x               0.0000000
## gyros_belt_y               0.0000000
## gyros_belt_z               0.0000000
## accel_belt_x               0.0000000
## accel_belt_y               0.0000000
## accel_belt_z               0.0000000
## magnet_belt_x              0.0000000
## magnet_belt_y              0.0000000
## magnet_belt_z              0.0000000
## roll_arm                   0.0000000
## pitch_arm                  0.0000000
## yaw_arm                    0.0000000
## total_accel_arm            0.0000000
## var_accel_arm              0.9797894
## avg_roll_arm               0.9797894
## stddev_roll_arm            0.9797894
## var_roll_arm               0.9797894
## avg_pitch_arm              0.9797894
## stddev_pitch_arm           0.9797894
## var_pitch_arm              0.9797894
## avg_yaw_arm                0.9797894
## stddev_yaw_arm             0.9797894
## var_yaw_arm                0.9797894
## gyros_arm_x                0.0000000
## gyros_arm_y                0.0000000
## gyros_arm_z                0.0000000
## accel_arm_x                0.0000000
## accel_arm_y                0.0000000
## accel_arm_z                0.0000000
## magnet_arm_x               0.0000000
## magnet_arm_y               0.0000000
## magnet_arm_z               0.0000000
## kurtosis_roll_arm          0.9836107
## kurtosis_picth_arm         0.9836957
## kurtosis_yaw_arm           0.9804688
## skewness_roll_arm          0.9835258
## skewness_pitch_arm         0.9836957
## skewness_yaw_arm           0.9804688
## max_roll_arm               0.9797894
## max_picth_arm              0.9797894
## max_yaw_arm                0.9797894
## min_roll_arm               0.9797894
## min_pitch_arm              0.9797894
## min_yaw_arm                0.9797894
## amplitude_roll_arm         0.9797894
## amplitude_pitch_arm        0.9797894
## amplitude_yaw_arm          0.9797894
## roll_dumbbell              0.0000000
## pitch_dumbbell             0.0000000
## yaw_dumbbell               0.0000000
## kurtosis_roll_dumbbell     0.9801291
## kurtosis_picth_dumbbell    0.9799592
## kurtosis_yaw_dumbbell      1.0000000
## skewness_roll_dumbbell     0.9800442
## skewness_pitch_dumbbell    0.9798743
## skewness_yaw_dumbbell      1.0000000
## max_roll_dumbbell          0.9797894
## max_picth_dumbbell         0.9797894
## max_yaw_dumbbell           0.9801291
## min_roll_dumbbell          0.9797894
## min_pitch_dumbbell         0.9797894
## min_yaw_dumbbell           0.9801291
## amplitude_roll_dumbbell    0.9797894
## amplitude_pitch_dumbbell   0.9797894
## amplitude_yaw_dumbbell     0.9801291
## total_accel_dumbbell       0.0000000
## var_accel_dumbbell         0.9797894
## avg_roll_dumbbell          0.9797894
## stddev_roll_dumbbell       0.9797894
## var_roll_dumbbell          0.9797894
## avg_pitch_dumbbell         0.9797894
## stddev_pitch_dumbbell      0.9797894
## var_pitch_dumbbell         0.9797894
## avg_yaw_dumbbell           0.9797894
## stddev_yaw_dumbbell        0.9797894
## var_yaw_dumbbell           0.9797894
## gyros_dumbbell_x           0.0000000
## gyros_dumbbell_y           0.0000000
## gyros_dumbbell_z           0.0000000
## accel_dumbbell_x           0.0000000
## accel_dumbbell_y           0.0000000
## accel_dumbbell_z           0.0000000
## magnet_dumbbell_x          0.0000000
## magnet_dumbbell_y          0.0000000
## magnet_dumbbell_z          0.0000000
## roll_forearm               0.0000000
## pitch_forearm              0.0000000
## yaw_forearm                0.0000000
## kurtosis_roll_forearm      0.9831861
## kurtosis_picth_forearm     0.9832711
## kurtosis_yaw_forearm       1.0000000
## skewness_roll_forearm      0.9831012
## skewness_pitch_forearm     0.9832711
## skewness_yaw_forearm       1.0000000
## max_roll_forearm           0.9797894
## max_picth_forearm          0.9797894
## max_yaw_forearm            0.9831861
## min_roll_forearm           0.9797894
## min_pitch_forearm          0.9797894
## min_yaw_forearm            0.9831861
## amplitude_roll_forearm     0.9797894
## amplitude_pitch_forearm    0.9797894
## amplitude_yaw_forearm      0.9831861
## total_accel_forearm        0.0000000
## var_accel_forearm          0.9797894
## avg_roll_forearm           0.9797894
## stddev_roll_forearm        0.9797894
## var_roll_forearm           0.9797894
## avg_pitch_forearm          0.9797894
## stddev_pitch_forearm       0.9797894
## var_pitch_forearm          0.9797894
## avg_yaw_forearm            0.9797894
## stddev_yaw_forearm         0.9797894
## var_yaw_forearm            0.9797894
## gyros_forearm_x            0.0000000
## gyros_forearm_y            0.0000000
## gyros_forearm_z            0.0000000
## accel_forearm_x            0.0000000
## accel_forearm_y            0.0000000
## accel_forearm_z            0.0000000
## magnet_forearm_x           0.0000000
## magnet_forearm_y           0.0000000
## magnet_forearm_z           0.0000000
## classe                     0.0000000
```

```r
# Remove columns with at least 50% NA
naColumns <- rownames(subset(naPerColumn, naPerColumn > 0.5, naPerColumn)) 
```

Now that we have identified the columns containing at least 50% `NA`, we can store this column in the variable `naColumns`. Next, we remove these columns from all of our data sets so they are not used in the predictions. Additionally, we process the sliced testing set and the original testing test to remove columns that do not contribute to the prediction just as we did for the sliced training set.


```r
# Remove these columns from the split training set
splitTrain <- splitTrain[ ,-which(names(splitTrain) %in% naColumns)]

# Remove these columns from the split testing set
splitTest <- splitTest[ ,-c(which(names(splitTest) %in% naColumns), 1:7)]

# Remove these columns from the testing set
testing <- testing[ ,-c(which(names(testing) %in% naColumns), 1)]
```

## Build the model

We will used the Generalized Boosting Model to analyze the data. The algorithm for this model is included as a part of the `caret` package.


```r
model <- train(classe ~ ., data = splitTrain, method = "gbm", verbose = FALSE)
```

## Perform cross-validation

Now that we have the model, we can test it on our sliced testing data with the `predict()` function. Then we display the Confusion Matrix. The Confusion Matrix compares the output from our predictions against the `classe` column from the testing set which indicates the actual class the data came from. Since this information is already provided to us in the data, we can determine the accuracy of our prediction by checking it against the actual class.


```r
prediction <- predict(model, splitTest)
confusionMatrix(prediction, splitTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2179   63    0    1    5
##          B   33 1413   33    4   18
##          C   10   42 1309   40   13
##          D    6    0   24 1229   19
##          E    4    0    2   12 1387
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9581          
##                  95% CI : (0.9534, 0.9624)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.947           
##  Mcnemar's Test P-Value : 3.98e-09        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9763   0.9308   0.9569   0.9557   0.9619
## Specificity            0.9877   0.9861   0.9838   0.9925   0.9972
## Pos Pred Value         0.9693   0.9414   0.9257   0.9617   0.9872
## Neg Pred Value         0.9905   0.9835   0.9908   0.9913   0.9915
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2777   0.1801   0.1668   0.1566   0.1768
## Detection Prevalence   0.2865   0.1913   0.1802   0.1629   0.1791
## Balanced Accuracy      0.9820   0.9585   0.9703   0.9741   0.9795
```

From the Confusion Matrix we can see that we have predicted with 95.8% accuracy. Our out-of-sample error is as follows:


```r
paste("Out-of-sample error:", round((1 - confusionMatrix(prediction, splitTest$classe)$overall[1]) * 100,2),"%")
```

```
## [1] "Out-of-sample error: 4.19 %"
```

## Prediction

Finally, we can perform our prediction on the testing set to see if we can accurately classify each of the participants into the exercise class they were asked to perform.


```r
predict(model, testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusions

The data from the Confusion Matrix indicates that the data was sufficient to predict the exercise class performed by the participants with high accuracy. Other models may provide even higher accuracy of classification. The conclusion from this assignments that, within the context of this experiment, the wearable devices are giving accurate measurements of the various performed tasks.
