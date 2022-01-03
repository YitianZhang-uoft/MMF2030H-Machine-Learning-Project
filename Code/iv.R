# Information Value
library(Information)

# Model 1
training <- read.csv(file = '/Users/zackwang/Desktop/Logistic Regression/training_new.csv')
IV <- create_infotables(data=training, y="TARGET", bins=10, parallel=FALSE)
IV_Value = data.frame(IV$Summary)

# Top IV: 0.3
plot_infotables(IV, "EXT_SOURCE_2")

write.csv(IV_Value,"/Users/zackwang/Desktop/Logistic Regression/IV_R.csv", row.names = FALSE)

# Model 2
data <- read.csv(file = '/Users/zackwang/Desktop/Logistic Regression/training.csv')
data <- subset(data, select = -c(SK_ID_CURR) )
IV <- create_infotables(data=data, y="TARGET", bins=10, parallel=FALSE)
IV_Value = data.frame(IV$Summary)
IV_Tables = IV$Tables

# Top IV: 0.33
plot_infotables(IV, "EXT_SOURCE_3")

write.csv(IV_Value,"/Users/zackwang/Desktop/Logistic Regression/IV_R_2.csv", row.names = FALSE)

library(feather)
sapply(seq_along(1:length(IV_Tables)), function(i) write_feather(IV_Tables[[i]], paste0("/Users/zackwang/Desktop/iv_tables/","DF",i,".feather")))
