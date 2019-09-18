library(sas7bdat)
library(nnet)
library(h2o)
library(dummies)
library(MASS)
library(reshape2)
library(caret)
library(dplyr)
library(pROC)
library(randomForest)
library(gbm)
library(xgboost)
library(e1071)
library(RColorBrewer)
library(ggplot2)
library(corrplot)
library(xlsx)


# Load File
abpredi<-read.sas7bdat("C:\\Users\\pri_p\\Desktop\\SAS\\TFM\\DataSources\\airbnb_ideal.sas7bdat")
#summary(abpredi)
#summary(airbnbbis)

# 2) Delete Variables
abpredi<- subset(abpredi, select = -size)
abpredi<- subset(abpredi, select = -longitude)
abpredi<- subset(abpredi, select = -latitude)

# 1) Fix Names
# dput(names(airbnbbis))
# dput(names(abpredi))
names(abpredi)[names(abpredi) == "Terraza"] <- "Patio_or_balcony"
names(abpredi)[names(abpredi) == "REP_rooms"] <- "bedrooms"
names(abpredi)[names(abpredi) == "G_neighborhood_ab"] <- "neighbourhood"
names(abpredi)[names(abpredi) == "AC"] <- "Air_conditioning"
names(abpredi)[names(abpredi) == "REP_bathrooms"] <- "bathrooms"
names(abpredi)[names(abpredi) == "Longitude1"] <- "longitude"
names(abpredi)[names(abpredi) == "Latitude1"] <- "latitude"



#3) Add Variables
abpredi$extra_people<-sample(airbnb$extra_people,296)
abpredi$cleaning_fee<-sample(airbnb$cleaning_fee,296)
abpredi$security_deposit<-sample(airbnb$security_deposit,296)
abpredi$minimum_nights<-sample(1:3,replace = TRUE,296)
abpredi$maximum_nights<-sample(28:31,replace = TRUE,296)
abpredi$availability_rate<-runif(296,0.9,0.98)
abpredi$cancellation_policy<-"flexible"
abpredi$host_response_time<-"within an hour"
abpredi$host_identity_verified<-"t"
abpredi$instant_bookable<-"t"
abpredi$is_location_exact<-"t"
abpredi$Hot_water<-1
abpredi$Internet<-1
abpredi$checkin24h<-1
abpredi$Coffee_maker<-1
abpredi$Host_greets_you<-1
abpredi$Has_License<-1
abpredi$Shampoo<-1
abpredi$Laptop_friendly_workspace<-1
abpredi$Cooking_basics<-1
abpredi$Microwave<-1
abpredi$Refrigerator<-1
abpredi$"24_hour_check_in"<-1
abpredi$Long_term_stays_allowed<-0


#Modify Variables
#summary(abpredi)
#summary(airbnbbis)

abpredi$is_location_exact <- as.factor(as.character(abpredi$is_location_exact))
abpredi$host_identity_verified <- as.factor(as.character(abpredi$host_identity_verified))
abpredi$instant_bookable <- as.factor(as.character(abpredi$instant_bookable))
abpredi$cancellation_policy <- as.factor(as.character(abpredi$cancellation_policy))
abpredi$host_response_time <- as.factor(as.character(abpredi$host_response_time))
abpredi$accommodates <- as.factor(as.character(abpredi$accommodates))
abpredi$bedrooms <- as.factor(as.character(abpredi$bedrooms))
abpredi$beds <- as.factor(as.numeric(abpredi$beds))

levels(abpredi$accommodates)[4]<-"7+"
levels(abpredi$bedrooms)[5]<-"4+"
abpredi$latitude <- (abpredi$latitude/10000000)
abpredi$longitude <-(abpredi$longitude/10000000)

#BIS changes

continuas<-c("extra_people", "availability_rate","latitude", "longitude",
             "cleaning_fee","security_deposit", "minimum_nights","maximum_nights")

categoricas<-c("host_identity_verified", "is_location_exact", "Has_License", 
               "instant_bookable", "Internet", "Shampoo", "Microwave", "Refrigerator", 
               "Air_conditioning", "24_hour_check_in", "Laptop_friendly_workspace", 
               "Hot_water", "Coffee_maker", "Cooking_basics", "Patio_or_balcony", 
               "Long_term_stays_allowed", "Host_greets_you","cancellation_policy", "bathrooms", 
               "bedrooms", "beds", "host_response_time", "neighbourhood", "accommodates")

# We will standarize the continuous variables, therefore we need calculate the means and the standart deviation

means <-apply(abpredi[,continuas],2,mean) 
sds<-sapply(abpredi[,continuas],sd) 
abpredibis<-scale(abpredi[,continuas], center = means, scale = sds)
numerocont<-which(colnames(abpredi)%in%continuas)
abpredibis<-cbind(abpredibis,abpredi[,-numerocont])
abpredibis<- subset(abpredibis, select = -price)

#summary(abpredibis)


####PREDICTIONS########

#unipredi$predi12<-(unipredi$gbm+unipredi$rf+unipredi$xgbm+unipredi$SVMRBF)/4

predictiair<-predict(xgbm,abpredibis)
predictiair1<-predict(gbm,abpredibis)
predictiair2<-predict(rf,abpredibis)
predictiair3<-predict(SVMr,abpredibis)

predi12ideal<-(predictiair+predictiair1+predictiair2+predictiair3)/4

abpredi$rentpredictions<-predi12ideal

#########ROI###############

#ROI in cash

abpredi$ROIC<-(abpredi$rentpredictions/abpredi$price)*100

#ROI in Mortage

i<-0.0225
dp<-0.2
pr<-abpredi$price
t<-30

totalinvest<-(pr+(pr*t*i)-(pr*dp))

abpredi$ROIM<-(abpredi$rentpredictions/(totalinvest))*100
