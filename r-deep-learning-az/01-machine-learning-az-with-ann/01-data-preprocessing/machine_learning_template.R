# STEP 1: Set Up
# Importing the data set
dataset = read.csv('Data.csv')
# dataset = dataset[,2:3] 


# STEP 2: Take Care of Missing Data
# Impute Age column
dataset$Age <-  ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x)mean(x, na.rm = TRUE)),
                     dataset$Age)

# Impute Salary Column
dataset$Salary <-  ifelse(is.na(dataset$Salary),
                       ave(dataset$Salary, FUN = function(x)mean(x, na.rm = TRUE)),
                       dataset$Salary)

# STEP 3: Encode Categorical variables

# Encode categorical data in features column
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),
                         labels = c(1,2,3))

# Encode categorical data purchased column
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No','Yes'),
                           labels = c(0,1))

# sTEP 4: Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split <-  sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set = subset(dataset, split == TRUE)

# STEP 5: Feature Scaling 
training_set[,2:3] <- scale(training_set[ ,2:3])
test_set[,2:3] <- scale(test_set[ ,2:3])
