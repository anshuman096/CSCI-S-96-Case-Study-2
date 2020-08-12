#
# @author: Anshuman Dikhit
#


suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(fastDummies))
suppressPackageStartupMessages(library(chron))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(FNN))


axiom_data = read.csv('./householdAxiomData.csv', na.strings = c('', NA))
credit_data = read.csv('./householdCreditData.csv', na.strings = c('', NA))
vehicle_data = read.csv('./householdVehicleData.csv', na.strings = c('', NA))
curr_customers = read.csv('./CurrentCustomerMktgResults.csv', na.strings = c('', NA))

prosp_customers = read.csv('./ProspectiveCustomers.csv', na.strings = c('', NA))

df = left_join(curr_customers, axiom_data, by = 'HHuniqueID')
df = left_join(df, credit_data, by = 'HHuniqueID')
df = left_join(df, vehicle_data, by = 'HHuniqueID')

test_df = left_join(prosp_customers, axiom_data, by = 'HHuniqueID')
test_df = left_join(test_df, credit_data, by = 'HHuniqueID')
test_df = left_join(test_df, vehicle_data, by = 'HHuniqueID')


# remove bad columns
df = df[, !(names(df) %in% c('dataID', 'past_Outcome', 'EstRace', 'annualDonations', 'X', 'carModel'))]

# convert categorical variables into dummies for easier calculation of Euclidean Distance
# Change all logical columns (T/F) into numericals (1/0)
logic_idx = sapply(df, is.logical)
df[, logic_idx] = lapply(df[, logic_idx], as.numeric)

# Convert LastContactMonth from factor to numeric
df$LastContactMonth = sapply(df$LastContactMonth, function(x) grep(paste("(?i)", x, sep=""), month.abb))


# Feature Engineering
hour = floor(as.numeric(times(df$CallStart)) * 24)
df$TimeOfDay = as.factor(ifelse(hour >= 9 & hour < 12, 'Morning', 
                                ifelse(hour >= 12 & hour < 15, 'Early_Afternoon', 'Late_Afternoon')))
df$CallDuration = as.numeric(times(df$CallEnd)) * 24 * 60 - as.numeric(times(df$CallStart)) * 24 * 60
df = df[, !(names(df) %in% c('CallStart', 'CallEnd'))]

# convert all categorical variables to dummies
# columns to convert: Communication, headOfhouseholdGender (binary), Job, Marital, Education, carMake, carModel, TimeOfDay
# if we wish to take missing variable values into account, then change ignore_na back to FALSE
df = dummy_cols(df, select_columns = 'headOfhouseholdGender', remove_first_dummy = TRUE, ignore_na = TRUE)
df = df[, !(names(df) %in% c('headOfhouseholdGender'))]

cat_cols = c('Communication', 'Job', 'Marital', 'Education', 'carMake', 'TimeOfDay')
df = dummy_cols(df, select_columns = cat_cols, ignore_na = TRUE) 
df = df[ ,!(names(df) %in% cat_cols)]
df = df[complete.cases(df), ]


# Generate and standardize Test Train partitions
set.seed(1234)
train_idx = sample(1:nrow(df), size = floor(0.65 * nrow(df)))

norm_train = train_df = df[train_idx, ]
norm_valid = valid_df = df[-train_idx, ]

norm_values = preProcess(train_df[, c(2:6, 8:ncol(train_df))], method = c('center', 'scale'))
norm_train[, c(2:6, 8:ncol(norm_train))] = predict(norm_values, train_df[, c(2:6, 8:ncol(train_df))])
norm_valid[, c(2:6, 8:ncol(norm_valid))] = predict(norm_values, valid_df[, c(2:6, 8:ncol(valid_df))])


# Run k-nearest neighbors testing accuracies for k = 1:750
k_accuracies = data.frame(k = 1:750, accuracy = rep(0, 25))
for(i in 1:750) {
    nn = knn(train = norm_train[, c(2:6, 8:ncol(norm_train))], 
             test = norm_valid[, c(2:6, 8:ncol(norm_valid))], 
             cl = norm_train[, 7],  k = i)
    confusion_matrix = confusionMatrix(nn, norm_valid$Y_AcceptedOffer)
    k_accuracies[i, 2] = confusion_matrix$overall[1]
}

# top 3 values for k
head(k_accuracies[order(-k_accuracies$accuracy), ], 3)

