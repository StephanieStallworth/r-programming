# Market Basket Optimization Associated Rule Learning - Eclat
# People who bought also bought
# Simplified apriori model: has just one parameter - support parameter
# Step 1: Set a min support (support = # transactions containing set of items divided by total number of transactions)
# Step 2: Take all the subset in transactions having higher support than minimum support
# Step 3: Sort these subsets by decreasing support
# For getting simple info like the sets of products most frequently purchased together


# Import dataset
dataset = read.csv('Market_Basket_Optimisation.csv',header = FALSE)

# Transform dataset into sparse matrix (matrix that contains alot of zeros)
#install.packages('arules')
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE) 
summary(dataset)
itemFrequencyPlot(dataset,topN = 10) #n most purchased products

# Training Eclat on the dataset - products that are bought at least 4 times a day
rules = eclat(data = dataset, parameter = list(support = 0.004,# products that are bought at least 4 times a day.min support is the number of transactions containing this product divided by total number of transactions
                                               minlen = 2)) # set of at least two items most frequently purchased together
                                               
# Visualizing the results - products that are bought at least 4 times a day
inspect(sort(rules,by = 'support')[1:10]) # Different sets of items most frequently purchased together

