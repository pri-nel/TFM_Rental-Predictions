library(sas7bdat)
library(nnet)
library(h2o)
library(dummies)
library(MASS)
library(reshape)
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
library(openxlsx)


# Load File
idpredi<-read.sas7bdat("C:\\Users\\pri_p\\Desktop\\SAS\\TFM\\DataSources\\pred_ideal.sas7bdat")
dput(names(idpredi))
summary(idpredi)

# dput(names(idealistabis))
# c("distance", "latitude", "longitude", "size", "SUM", "AC", "Piscina", 
#   "Terraza", "Amueblado", "Parking", "Yearly_Price", "bathrooms", 
#   "rooms", "hasLift", "neighborhood", "floor")

# dput(names(idpredi))
# c("AC", "Amueblado", "Piscina", "Terraza", "distance", "price", 
#   "size", "SUM", "Parking", "Latitude", "Longitude", "G_neighborhood", 
#   "REP_bathrooms", "REP_floor", "REP_hasLift", "REP_rooms")


names(idpredi)[names(idpredi) == "REP_hasLift"] <- "hasLift"
names(idpredi)[names(idpredi) == "REP_rooms"] <- "rooms"
names(idpredi)[names(idpredi) == "G_neighborhood"] <- "neighborhood"
names(idpredi)[names(idpredi) == "REP_floor"] <- "floor"
names(idpredi)[names(idpredi) == "Latitude"] <- "latitude"
names(idpredi)[names(idpredi) == "Longitude"] <- "longitude"

dput(names(idpredi))

#I need to transform rooms and bathroom to interval
idpredi$rooms <- as.numeric(as.character(idpredi$rooms))
idpredi$bathrooms <- as.numeric(as.character(idpredi$bathrooms))
summary(idpredi)

continuas<-c("distance", "latitude", "longitude", "size","SUM")
categoricas<-c("AC", "Piscina","Terraza", "Amueblado","Parking","bathrooms","floor", "rooms", "hasLift", "neighborhood")

# We will standarize the continuous variables, therefore we need calculate the means and the standart deviation

means <-apply(idpredi[,continuas],2,mean) 
sds<-sapply(idpredi[,continuas],sd) 
idpredibis<-scale(idpredi[,continuas], center = means, scale = sds)
numerocont<-which(colnames(idpredi)%in%continuas)
idpredibis<-cbind(idpredibis,idpredi[,-numerocont])
idpredibis<- subset(idpredibis, select = -price)

summary(idpredibis)

####PREDICTIONS########

#unipredi$predi12<-(unipredi$gbm+unipredi$rf+unipredi$xgbm+unipredi$SVMRBF)/4

predictideal<-predict(xgbm,idpredibis)
predictideal1<-predict(gbm,idpredibis)
predictideal2<-predict(rf,idpredibis)
predictideal3<-predict(SVMr,idpredibis)

predi12ideal<-(predictideal+predictideal1+predictideal2+predictideal3)/4

idpredi$rentpredictions<-predi12ideal

#########ROI###############

#ROI in cash

idpredi$ROIC<-(idpredi$rentpredictions/idpredi$price)*100

#ROI in Mortage

i<-0.0225
dp<-0.2
pr<-idpredi$price
t<-30

totalinvest<-(pr+(pr*t*i)-(pr*dp))

idpredi$ROIM<-(idpredi$rentpredictions/(totalinvest))*100



###################FINAL


idpredi$latitude <- (idpredi$latitude/10000000)
idpredi$longitude <-(idpredi$longitude/10000000)

idpredi$key<-paste(idpredi$price,idpredi$latitude,idpredi$longitude)


write.xlsx(idpredi,file="C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\00_TFM\\R_project\\Idealista_predi.xlsx",
           sheetName = "ideal", row.names=FALSE, col.names=TRUE,append = FALSE)
