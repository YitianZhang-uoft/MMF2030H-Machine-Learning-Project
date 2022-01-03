# Model Fitting 1
training <- read.csv(file = '/Users/zackwang/Desktop/Logistic Regression/training_new.csv')

# all features without collinearty
fullmod <- glm(formula = TARGET ~ TOTALAREA_MODE+APARTMENTS_AVG+DAYS_EMPLOYED+
                 ORGANIZATION_TYPE_XNA+FLOORSMIN_AVG+YEARS_BUILD_AVG+AMT_GOODS_PRICE+
                 FLOORSMAX_AVG+FLOORSMAX_MEDI+AMT_DRAWINGS_POS_CURRENT+AMT_DRAWINGS_CURRENT+
                 NONLIVINGAPARTMENTS_MEDI+CODE_GENDER_M+NONLIVINGAREA_AVG+NONLIVINGAREA_MEDI+
                 FLAG_OWN_REALTY_N+EXT_SOURCE_2+REGION_RATING_CLIENT+OCCUPATION_TYPE_Core_staff+
                 ORGANIZATION_TYPE_Kindergarten+NAME_CONTRACT_TYPE_Cash_loans+
                 NAME_EDUCATION_TYPE_Higher_education+EMERGENCYSTATE_MODE_No+HOUSETYPE_MODE_block_of_flats+
                 REG_REGION_NOT_WORK_REGION+OBS_60_CNT_SOCIAL_CIRCLE+OBS_30_CNT_SOCIAL_CIRCLE+COMMONAREA_AVG+
                 COMMONAREA_MEDI+YEARS_BEGINEXPLUATATION_MEDI+YEARS_BEGINEXPLUATATION_AVG+
                 DEF_30_CNT_SOCIAL_CIRCLE+DEF_60_CNT_SOCIAL_CIRCLE+NAME_FAMILY_STATUS_Widow+CNT_FAM_MEMBERS+
                 LANDAREA_MEDI+NAME_TYPE_SUITE_Unaccompanied+NAME_TYPE_SUITE_Family+
                 REG_CITY_NOT_WORK_CITY+LIVE_CITY_NOT_WORK_CITY+AMT_CREDIT_SUM+AMT_CREDIT_SUM_DEBT+
                 ORGANIZATION_TYPE_Medicine+EXT_SOURCE_3+OCCUPATION_TYPE_Security_staff+
                 FONDKAPREMONT_MODE_org_spec_account+NAME_HOUSING_TYPE_House_apartment+
                 NAME_HOUSING_TYPE_With_parents+OWN_CAR_AGE+FLAG_OWN_CAR_N+CREDIT_ACTIVE+
                 CREDIT_TYPE_CREDIT_CARD+FLOORSMIN_MEDI+FLOORSMIN_MODE+ORGANIZATION_TYPE_Services+
                 OCCUPATION_TYPE_Private_service_staff+ENTRANCES_AVG+ENTRANCES_MEDI+
                 NAME_FAMILY_STATUS_Married+NAME_FAMILY_STATUS_Separated+CNT_INSTALMENT_FUTURE+
                 AMT_REQ_CREDIT_BUREAU_WEEK+AMT_REQ_CREDIT_BUREAU_DAY+ORGANIZATION_TYPE_Self_employed+
                 HOUR_APPR_PROCESS_START+NAME_EDUCATION_TYPE_Lower_secondary+
                 WEEKDAY_APPR_PROCESS_START_MONDAY+WEEKDAY_APPR_PROCESS_START_WEDNESDAY+
                 ORGANIZATION_TYPE_Military+WEEKDAY_APPR_PROCESS_START_SUNDAY+WEEKDAY_APPR_PROCESS_START_SATURDAY+
                 OCCUPATION_TYPE_Accountants+ORGANIZATION_TYPE_Bank+BASEMENTAREA_AVG+BASEMENTAREA_MEDI+
                 REG_CITY_NOT_LIVE_CITY+NAME_HOUSING_TYPE_Rented_apartment+WALLSMATERIAL_MODE_Panel+
                 WALLSMATERIAL_MODE_Stone_brick+ORGANIZATION_TYPE_Industry_type_10+
                 ORGANIZATION_TYPE_Industry_type_9+ORGANIZATION_TYPE_Religion+
                 OCCUPATION_TYPE_IT_staff+ORGANIZATION_TYPE_Transport_type_3+EXT_SOURCE_1+
                 FLAG_EMAIL+DAYS_LAST_PHONE_CHANGE+NAME_INCOME_TYPE_Commercial_associate+
                 ORGANIZATION_TYPE_Trade_type_3+ORGANIZATION_TYPE_Culture+
                 ORGANIZATION_TYPE_Emergency+ORGANIZATION_TYPE_Trade_type_7+
                 ORGANIZATION_TYPE_Housing+ORGANIZATION_TYPE_Electricity+ORGANIZATION_TYPE_Government+
                 OCCUPATION_TYPE_Low_skill_Laborers+ORGANIZATION_TYPE_Business_Entity_Type_1+
                 ORGANIZATION_TYPE_Business_Entity_Type_3+ORGANIZATION_TYPE_Other+OCCUPATION_TYPE_HR_staff+
                 NAME_FAMILY_STATUS_Civil_marriage+ORGANIZATION_TYPE_Industry_type_11+
                 ORGANIZATION_TYPE_Business_Entity_Type_2+DAYS_ID_PUBLISH+DAYS_REGISTRATION+
                 WALLSMATERIAL_MODE_Others+ORGANIZATION_TYPE_Construction+ORGANIZATION_TYPE_Transport_type_4+
                 NAME_INCOME_TYPE_State_servant+ORGANIZATION_TYPE_School+ORGANIZATION_TYPE_Industry_type_5+
                 ORGANIZATION_TYPE_Industry_type_13+FONDKAPREMONT_MODE_reg_oper_account+
                 FONDKAPREMONT_MODE_reg_oper_spec_account+FLAG_CONT_MOBILE+WALLSMATERIAL_MODE_Monolithic+
                 FONDKAPREMONT_MODE_not_specified+SK_DPD+OCCUPATION_TYPE_High_skill_tech_staff+
                 ORGANIZATION_TYPE_Telecom+ORGANIZATION_TYPE_Agriculture+WALLSMATERIAL_MODE_Mixed+
                 ORGANIZATION_TYPE_Insurance+ORGANIZATION_TYPE_Industry_type_3+ORGANIZATION_TYPE_Transport_type_2+
                 NAME_HOUSING_TYPE_Co_op_apartment+OCCUPATION_TYPE_Cleaning_staff+ORGANIZATION_TYPE_Trade_type_6+
                 ORGANIZATION_TYPE_Realtor+WEEKDAY_APPR_PROCESS_START_FRIDAY+ORGANIZATION_TYPE_Postal+
                 HOUSETYPE_MODE_terraced_house+ORGANIZATION_TYPE_Industry_type_12+ORGANIZATION_TYPE_Restaurant+
                 ORGANIZATION_TYPE_Industry_type_1+OCCUPATION_TYPE_Realty_agents+ORGANIZATION_TYPE_Trade_type_1+
                 ORGANIZATION_TYPE_Industry_type_7+OCCUPATION_TYPE_Secretaries+ORGANIZATION_TYPE_University+
                 ORGANIZATION_TYPE_Cleaning+ORGANIZATION_TYPE_Police+NAME_TYPE_SUITE_Other_A+
                 ORGANIZATION_TYPE_Industry_type_2+ORGANIZATION_TYPE_Advertising+
                 AMT_PAYMENT_GREATER_EQUAL_INSTALMENT+AMT_CREDIT_MAX_OVERDUE+ORGANIZATION_TYPE_Mobile+
                 ORGANIZATION_TYPE_Trade_type_4, family = binomial(), data = training)

nothing <- glm(TARGET ~ 1, family = binomial(), data = training)

logit1 <- step(nothing, list(lower=formula(nothing),upper=formula(fullmod)), direction="both",trace=0)
summary(logit1)

logit1 <- glm(formula = TARGET ~ EXT_SOURCE_2 + EXT_SOURCE_3 + CREDIT_ACTIVE + 
                EXT_SOURCE_1 + CODE_GENDER_M + NAME_EDUCATION_TYPE_Higher_education + 
                ORGANIZATION_TYPE_XNA + DAYS_EMPLOYED + CNT_INSTALMENT_FUTURE + 
                NAME_CONTRACT_TYPE_Cash_loans + SK_DPD + AMT_DRAWINGS_CURRENT + 
                DEF_30_CNT_SOCIAL_CIRCLE + FLAG_OWN_CAR_N + FLOORSMAX_AVG + 
                DAYS_ID_PUBLISH + NAME_FAMILY_STATUS_Married + OWN_CAR_AGE + 
                REGION_RATING_CLIENT + AMT_CREDIT_SUM_DEBT + NAME_FAMILY_STATUS_Widow + 
                DAYS_LAST_PHONE_CHANGE + NAME_INCOME_TYPE_State_servant + 
                ORGANIZATION_TYPE_Bank + REG_CITY_NOT_LIVE_CITY + AMT_DRAWINGS_POS_CURRENT + 
                CNT_FAM_MEMBERS + ORGANIZATION_TYPE_Self_employed + NAME_INCOME_TYPE_Commercial_associate + 
                ORGANIZATION_TYPE_Postal + WALLSMATERIAL_MODE_Panel + ORGANIZATION_TYPE_Industry_type_9 + 
                OCCUPATION_TYPE_Core_staff + DAYS_REGISTRATION + WEEKDAY_APPR_PROCESS_START_SUNDAY + 
                ORGANIZATION_TYPE_Industry_type_10 + OCCUPATION_TYPE_High_skill_tech_staff + 
                WEEKDAY_APPR_PROCESS_START_MONDAY + FONDKAPREMONT_MODE_reg_oper_account + 
                WALLSMATERIAL_MODE_Others + NAME_FAMILY_STATUS_Civil_marriage + 
                FLAG_CONT_MOBILE + ORGANIZATION_TYPE_Restaurant + OCCUPATION_TYPE_Accountants + 
                OCCUPATION_TYPE_Low_skill_Laborers + ORGANIZATION_TYPE_Military + 
                ORGANIZATION_TYPE_Medicine + ORGANIZATION_TYPE_Electricity + 
                WEEKDAY_APPR_PROCESS_START_SATURDAY + ORGANIZATION_TYPE_School + 
                ORGANIZATION_TYPE_Housing + ENTRANCES_MEDI + ORGANIZATION_TYPE_Industry_type_12 + 
                NAME_TYPE_SUITE_Unaccompanied + ORGANIZATION_TYPE_Transport_type_3 + 
                NAME_HOUSING_TYPE_House_apartment + ORGANIZATION_TYPE_Police + 
                ORGANIZATION_TYPE_Construction, family = binomial(), data = training)
summary(logit1)

# AR
library(pROC)
testing <- read.csv(file = '/Users/zackwang/Desktop/Logistic Regression/testing_adjusted.csv')
roc(testing$TARGET, predict(logit1, type='response', testing), plot=TRUE)

# Score test data set
step_logit_pred <- prediction(predict(logit1, type='response', testing), testing$TARGET)
step_logit_perf <- performance(step_logit_pred,"tpr","fpr")

# ROC
plot(step_logit_perf, lwd=2, colorize=FALSE)
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);


# Confusion matrix
library(caret)
confusionMatrix(reference = as.factor(testing$TARGET), data = as.factor(as.numeric(predict(logit1, type='response', testing) > 0.5)))





# WOE
# Model Fitting
training <- read.csv(file = '/Users/zackwang/Desktop/Logistic Regression/training_woe_new.csv')

# all features without collinearty
fullmod <- glm(formula = TARGET ~ WOE_ENTRANCES_MEDI+WOE_NONLIVINGAREA_MODE+
                 WOE_AMT_CREDIT_MAX_OVERDUE+WOE_AMT_REQ_CREDIT_BUREAU_WEEK+
                 WOE_AMT_DRAWINGS_CURRENT+WOE_AMT_TOTAL_RECEIVABLE+
                 WOE_LIVINGAPARTMENTS_MEDI+WOE_COMMONAREA_MODE+WOE_DAYS_EMPLOYED+
                 WOE_FLAG_EMP_PHONE+WOE_DEF_30_CNT_SOCIAL_CIRCLE+
                 WOE_OBS_30_CNT_SOCIAL_CIRCLE+WOE_EXT_SOURCE_2+WOE_REGION_RATING_CLIENT+
                 WOE_AMT_GOODS_PRICE+WOE_AMT_CREDIT+WOE_REG_REGION_NOT_WORK_REGION+
                 WOE_REG_CITY_NOT_WORK_CITY+WOE_FLOORSMAX_AVG+WOE_ELEVATORS_AVG+
                 WOE_CNT_INSTALMENT_FUTURE+WOE_AMT_PAYMENT_GREATER_EQUAL_INSTALMENT+
                 WOE_EXT_SOURCE_3+WOE_CREDIT_TYPE_CREDIT_CARD+WOE_FLOORSMIN_AVG+
                 WOE_FLOORSMIN_MEDI+WOE_EXT_SOURCE_1+WOE_OCCUPATION_TYPE+WOE_CODE_GENDER+
                 WOE_DAYS_REGISTRATION+WOE_NAME_FAMILY_STATUS+WOE_NAME_HOUSING_TYPE+WOE_OWN_CAR_AGE+
                 WOE_AMT_INCOME_TOTAL+WOE_FLAG_WORK_PHONE+WOE_FLAG_OWN_REALTY+WOE_NAME_CONTRACT_TYPE+
                 WOE_CNT_FAM_MEMBERS+WOE_NAME_TYPE_SUITE+WOE_WEEKDAY_APPR_PROCESS_START+WOE_DAYS_LAST_PHONE_CHANGE, family = binomial(), data = training)

nothing <- glm(TARGET ~ 1, family = binomial(), data = training)

logit2 <- step(nothing, list(lower=formula(nothing),upper=formula(fullmod)), direction="both",trace=0)
summary(logit2)

logit2 <- glm(formula = TARGET ~ WOE_EXT_SOURCE_3 + WOE_EXT_SOURCE_2 + 
                WOE_EXT_SOURCE_1 + WOE_OCCUPATION_TYPE + WOE_AMT_TOTAL_RECEIVABLE + 
                WOE_DAYS_EMPLOYED + WOE_CNT_INSTALMENT_FUTURE + WOE_AMT_CREDIT + 
                WOE_NAME_CONTRACT_TYPE + WOE_OWN_CAR_AGE + WOE_CODE_GENDER + 
                WOE_DEF_30_CNT_SOCIAL_CIRCLE + WOE_ELEVATORS_AVG + WOE_AMT_CREDIT_MAX_OVERDUE + 
                WOE_DAYS_LAST_PHONE_CHANGE + WOE_REGION_RATING_CLIENT + WOE_FLAG_WORK_PHONE + 
                WOE_NAME_FAMILY_STATUS + WOE_WEEKDAY_APPR_PROCESS_START + 
                WOE_AMT_DRAWINGS_CURRENT + WOE_DAYS_REGISTRATION + WOE_LIVINGAPARTMENTS_MEDI + 
                WOE_AMT_GOODS_PRICE + WOE_AMT_REQ_CREDIT_BUREAU_WEEK + WOE_CREDIT_TYPE_CREDIT_CARD + 
                WOE_FLAG_OWN_REALTY + WOE_NAME_TYPE_SUITE + WOE_FLAG_EMP_PHONE + 
                WOE_CNT_FAM_MEMBERS + WOE_NAME_HOUSING_TYPE + WOE_AMT_PAYMENT_GREATER_EQUAL_INSTALMENT + 
                WOE_REG_REGION_NOT_WORK_REGION, family = binomial(), data = training)

# AR
library(pROC)
testing <- read.csv(file = '/Users/zackwang/Desktop/Logistic Regression/testing_woe.csv')
roc(testing$TARGET, predict(logit2, type='response', testing), plot=TRUE)

# Score test data set
step_logit_pred <- prediction(predict(logit2, type='response', testing), testing$TARGET)
step_logit_perf <- performance(step_logit_pred,"tpr","fpr")

# ROC
plot(step_logit_perf, lwd=2, colorize=FALSE)
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);

# Confusion matrix
library(caret)
confusionMatrix(reference = as.factor(testing$TARGET), data=as.factor(as.numeric(predict(logit2, type='response', testing) > 0.5)))
