# Association Rule Learning - Apriori

# Import dataset
dataset = read.csv('Market_Basket_Optimisation.csv',header = FALSE)

# Transform dataset into sparse matrix (matrix that contains alot of zeros)
#install.packages('arules')
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE) 
summary(dataset)
itemFrequencyPlot(dataset,topN = 10) #n most purchased products

# Training Apriori on the dataset - products that are bought at least 3 times a day
rules = apriori(data = dataset, parameter = list(support = 0.003,# products that are bought at least 3 times a day.min support is the number of transactions containing this product divided by total number of transactions
                                                 confidence =0.20 )) # each rule should be correct for this % of transactions. Strt with default then decrease

# Visualizing the results -products that are bought at least 3 times a day
inspect(sort(rules,by = 'lift')[1:10]) # Sort by the highest rules then grab the first 10 rows


# Training Apriori on the dataset -products that are bought at least 4 times a day
rules = apriori(data = dataset, parameter = list(support = 0.004,# products that are bought at least  4 times a day. support is the number of transactions containing this product over the total number of products
                                                 confidence =0.20 )) # each rule should be correct for this % of transactions. Strt with default then decrease

# Visualizing the results - products that are bought at least 3 times a day
inspect(sort(rules,by = 'lift')[1:10]) # Sort by the highest rules then grab the first 10 rows
