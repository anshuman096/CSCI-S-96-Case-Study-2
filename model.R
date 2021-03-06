#
# @author: Anshuman Dikhit
#


suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(fastDummies))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(FNN))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(data.table))


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
df = df[, !(names(df) %in% c('dataID', 'past_Outcome', 'EstRace', 'annualDonations', 
                             'X', 'carMake', 'carModel', 'CallStart', 'CallEnd', 'PetsPurchases', 'DigitalHabits_5_AlwaysOn'))]

# convert categorical variables into dummies for easier calculation of Euclidean Distance
# Change all logical columns (T/F) into numericals (1/0)
logic_idx = sapply(df, is.logical)
df[, logic_idx] = lapply(df[, logic_idx], as.numeric)

# Convert LastContactMonth from factor to numeric
# used following stackoverflow: https://stackoverflow.com/questions/6549239/convert-months-mmm-to-numeric
df$LastContactMonth = sapply(df$LastContactMonth, function(x) grep(paste("(?i)", x, sep=""), month.abb))

# EDA + Dimension Reduction


# Job Categories
job_df = data.frame(Job = df$Job, accepted = df$Y_AcceptedOffer)
ggplot(job_df, aes(x = accepted)) + 
    geom_bar() + 
    facet_wrap(~ Job)
table(job_df)

# People in the working class are less likely to accept than non-workers
wk_class = c('admin.', 'blue-collar', 'entrepreneur', 'housemaid','management', 
             'retired', 'self-employed', 'services', 'student', 'technician')
df$WorkingClass = ifelse(df$Job %in% wk_class, 1, 0)
df = df[, !(names(df) %in% c('Job'))]


# Education
educ_df = data.frame(Education = df$Education, Accepted = df$Y_AcceptedOffer)
ggplot(educ_df, aes(x = Accepted)) + 
    geom_bar() + 
    facet_wrap(~ Education)

table(educ_df)[, 'Accepted']/(table(educ_df)[, 'Accepted'] + table(educ_df)[, 'DidNotAccept'])


# Marital Status
marriage_df = data.frame(Married = df$Marital, Accepted = df$Y_AcceptedOffer)
ggplot(marriage_df, aes(x = Accepted)) + 
    geom_bar() + 
    facet_wrap(~ Married)

# People who currently have a life partner are much more likely to decline than those who live on their own. This does not yield any significant change to our model however.
table(marriage_df)[, 'Accepted']/(table(marriage_df)[, 'Accepted'] + table(marriage_df)[, 'DidNotAccept'])


# Car Year
carYr_df = data.frame(carYr = as.factor(df$carYr), Accepted = df$Y_AcceptedOffer)
carYr_df = table(carYr_df)

carYr_df = data.frame(carYr_df[, 'Accepted']/(carYr_df[, 'Accepted'] + carYr_df[, 'DidNotAccept']))
carYr_df = setDT(carYr_df, keep.rownames = TRUE)[]
colnames(carYr_df) = c('carYr', 'Pct_Accepted')
ggplot(carYr_df, aes(x = carYr, y = Pct_Accepted)) + 
    geom_bar(stat = 'identity') + 
    theme(axis.text.x = element_text(angle = 90, hjust = 1))


# Age

summary(df$Age)

# Generational age ranges determined via this article: https://www.kasasa.com/articles/generations/gen-x-gen-y-gen-z
generation = ifelse(df$Age >= 18 & df$Age <= 25, 'GenZ',
                    ifelse(df$Age >= 26 & df$Age <= 40, 'Millennial',
                           ifelse(df$Age >= 41 & df$Age <= 55, 'GenX',
                                  ifelse(df$Age >= 56 & df$Age <= 76, 'Baby.Boomers', 'Silent/GreatestGen'))))
generation = as.factor(generation)
df$NonWorkingGen = ifelse(generation %in% c('GenZ', 'Baby.Boomers', 'Silent/GreatestGen'), 1, 0)

gen_df = data.frame(Gen = df$NonWorkingGen, Accepted = df$Y_AcceptedOffer)
ggplot(gen_df, aes(x = Accepted)) + 
    geom_bar() + 
    facet_wrap(~ Gen)

df = df[, !(names(df) %in% c('Age'))]


# convert all categorical variables to dummies
# columns to convert: Communication, headOfhouseholdGender (binary), Job, Marital, Education, carMake, carModel, TimeOfDay
# if we wish to take missing variable values into account, then change ignore_na back to FALSE
df = dummy_cols(df, select_columns = c('Communication', 'headOfhouseholdGender'), remove_first_dummy = TRUE, ignore_na = TRUE)
df = df[, !(names(df) %in% c('Communication', 'headOfhouseholdGender'))]

cat_cols = c('Education', 'Marital')
df = dummy_cols(df, select_columns = cat_cols, ignore_na = TRUE) 
df = df[ ,!(names(df) %in% cat_cols)]
df = df[complete.cases(df), ]


train_idx = sample(1:nrow(df), size = floor(0.70 * nrow(df)))

norm_train = train_df = df[train_idx, ]
norm_valid = valid_df = df[-train_idx, ]

norm_values = preProcess(train_df[, c(2:6, 8:ncol(train_df))], method = c('center', 'scale'))
norm_train[, c(2:6, 8:ncol(norm_train))] = predict(norm_values, train_df[, c(2:6, 8:ncol(train_df))])
norm_valid[, c(2:6, 8:ncol(norm_valid))] = predict(norm_values, valid_df[, c(2:6, 8:ncol(valid_df))])



# kNN


k_accuracies = data.frame(k = 1:150, accuracy = rep(0, 150))
for(i in 1:150) {
    nn = knn(train = norm_train[, c(2:6, 8:ncol(norm_train))], 
             test = norm_valid[, c(2:6, 8:ncol(norm_valid))], 
             cl = norm_train[, 7],  k = i)
    confusion_matrix = confusionMatrix(nn, norm_valid$Y_AcceptedOffer)
    k_accuracies[i, 2] = confusion_matrix$overall[1]
}



head(k_accuracies[order(-k_accuracies$accuracy), ], 3)



ggplot(k_accuracies, aes(x = k, y = accuracy)) + 
    geom_line() + ggtitle('kNN accuracy for k = 1:150')





