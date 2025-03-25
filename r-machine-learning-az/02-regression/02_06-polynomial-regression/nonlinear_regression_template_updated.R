 # Regression Template

######### Step 1 ##########

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

######### Step 2 ##########
# Fitting the Regression Model to the dataset
# Create your regressor here 

########## Step 4 ##########
# Predicting a new result 
# Predict for a single observation point 
# Polynomial Regression Model learned the correlations in dataset containing Level1 column and also the polynomial columns (Level2 - Level4) 
# So when we create new dataframe containing the 6.5 observation, have to add the additional levels too Levels2 - Level4

y_pred = predict(regressor, data.frame(Level = 6.5))

########## Step 3 ##########
# Visualising the Regression Model results (Basic Curve)
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model Results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)

# Instead of only predicting 10 salaries of 10 levels, predict 90 salaries of 90 levels  
# Build vector/sequence of imaginary levels between 1-10 incremented by 0.1  
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, 
				newdata = data.frame(Level = x_grid))), # create new column of levels containing all the levels in x_grid
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +