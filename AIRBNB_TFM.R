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

getOption("max.print")
options(max.print = 99999999)

#Load Macros

source ("C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\Q2_Machine Learning\\R\\cruzadas avnnet y lin.R")
source ("C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\Q2_Machine Learning\\R\\cruzada arbol continua.R")
source ("C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\Q2_Machine Learning\\R\\cruzada rf continua.R")
source ("C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\Q2_Machine Learning\\R\\cruzada gbm continua.R")
source ("C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\Q2_Machine Learning\\R\\cruzada xgboost continua.R")
source ("C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\Q2_Machine Learning\\R\\cruzada SVM continua lineal.R")
source ("C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\Q2_Machine Learning\\R\\cruzada SVM continua polinomial.R")
source ("C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\Q2_Machine Learning\\R\\cruzada SVM continua RBF.R")
source ("C:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\Q2_Machine Learning\\R\\cruzadas ensamblado continuas fuente.R")


# Load File
airbnb<-read.sas7bdat("C:\\Users\\pri_p\\Desktop\\SAS\\TFM\\DataSources\\airbnb_vs.sas7bdat")
dput(names(airbnb))
summary(airbnb)

names(airbnb)[names(airbnb) == "X_24_hour_check_in"] <- "24_hour_check_in"
names(airbnb)[names(airbnb) == "REP_maximum_nights"] <- "maximum_nights"
names(airbnb)[names(airbnb) == "REP_accommodates"] <- "accommodates"
names(airbnb)[names(airbnb) == "REP_cancellation_policy"] <- "cancellation_policy"
names(airbnb)[names(airbnb) == "IMP_REP_bathrooms"] <- "bathrooms"
names(airbnb)[names(airbnb) == "IMP_REP_cleaning_fee"] <- "cleaning_fee"
names(airbnb)[names(airbnb) == "IMP_REP_host_response_time"] <- "host_response_time"
names(airbnb)[names(airbnb) == "IMP_security_deposit"] <- "security_deposit"
names(airbnb)[names(airbnb) == "G_neighbourhood_cleansed"] <- "neighbourhood"
names(airbnb)[names(airbnb) == "IMP_REP_bedrooms"] <- "bedrooms"
names(airbnb)[names(airbnb) == "IMP_REP_beds"] <- "beds"

dput(names(airbnb))

continuas<-c("extra_people", "minimum_nights", "availability_rate","latitude", "longitude",
             "maximum_nights", "cleaning_fee","security_deposit")

categoricas<-c("host_identity_verified", "is_location_exact", "Has_License", 
               "instant_bookable", "Internet", "Shampoo", "Microwave", "Refrigerator", 
               "Air_conditioning", "24_hour_check_in", "Laptop_friendly_workspace", 
               "Hot_water", "Coffee_maker", "Cooking_basics", "Patio_or_balcony", 
               "Long_term_stays_allowed", "Host_greets_you","cancellation_policy", "bathrooms", 
               "bedrooms", "beds", "host_response_time", "neighbourhood", "accommodates")

# We will standarize the continuous variables, therefore we need calculate the means and the standart deviation

means <-apply(airbnb[,continuas],2,mean) 
sds<-sapply(airbnb[,continuas],sd) 
airbnbbis<-scale(airbnb[,continuas], center = means, scale = sds)
numerocont<-which(colnames(airbnb)%in%continuas)
airbnbbis<-cbind(airbnbbis,airbnb[,-numerocont])

# Finally, with this process we have our data standarized and ready to model. 

# ***************************
# TUNING CON CARET
# ***************************

set.seed(12346)

# Validación cruzada una sola vez
control<-trainControl(method = "cv",number=4,savePredictions = "all") 

# Validación cruzada repetida
control<-trainControl(method = "repeatedcv",number=4,repeats=5,savePredictions = "all") 

# Training test una sola vez
control<-trainControl(method = "LGOCV",p=0.8,number=1,savePredictions = "all") 

# Training test repetido
control<-trainControl(method = "LGOCV",p=0.8,number=5,savePredictions = "all") 



# ***************************
# TUNING NEURAL NETWORKS
# ***************************


# EN LO SUCESIVO APLICAMOS VALIDACIÓN CRUZADA REPETIDA

set.seed(12346)
# Validación cruzada repetida
control<-trainControl(method = "repeatedcv",number=4,repeats=5,
                      savePredictions = "all",classProbs=TRUE) 

# ***************************************************************
# NEURAL NET WORK
#     Number of Hidden Units (size, numeric)
#     Weight Decay (decay, numeric)
#     linout = FALSE
# ***************************************************************

nnetgrid <-  expand.grid(size=c(2,6,8,10,13,15,20),decay=c(0.01,0.1,0.001,0.2,0.05))

rednnet<- train(Yearly_Profit~.,data=airbnbbis,
                method="nnet",linout = TRUE,maxit=100,trControl=control,tuneGrid=nnetgrid)

rednnet
#The final values used for the model were size = 20 and decay = 0.2.
#size  decay  RMSE      Rsquared    MAE   
#20    0.200  9216.811  0.10413366  6456.731


avnnetgrid <-expand.grid(size=c(8,10,12,15,18,20,22),decay=c(0.01,0.1,0.001,0.2,0.05),bag=FALSE)

redavnnet<- train(Yearly_Profit~.,data=airbnbbis,
                  method="avNNet",linout = TRUE,maxit=100,trControl=control,repeats=5,tuneGrid=avnnetgrid)

redavnnet$results

#The final values used for the model were size = 20, decay = 0.1 and bag = FALSE.
# size  decay  RMSE      Rsquared    MAE  
#  20    0.100  9069.343  0.15580387  6327.109


# ***************************************************************
# Random Forest: TUNNING
# mtry: Number of variable is randomly collected to be sampled at each split time.
# ntree: Number of branches will grow after each time split.
# Sampsize:
# Node size:
# ***************************************************************


set.seed(12345)

rfgrid<-expand.grid(mtry=c(5,8,10,12,15,18,20,25,30,32))

rf<- train(Yearly_Profit~.,data=airbnbbis,
           method="rf",trControl=control,tuneGrid=rfgrid,
           linout = TRUE,ntree=1000, nodesize=20,replace=TRUE,
           importance=TRUE)

rf$results

# mtry  RMSE      Rsquared   MAE     
#18    7522.438  0.4000053  5206.107

#Let's try to improve it by sampling.

rfgrid2<-expand.grid(mtry=c(18))

rf2<- train(Yearly_Profit~.,data=airbnbbis,
            method="rf",trControl=control,tuneGrid=rfgrid2,
            linout = TRUE,ntree=1000, sampsize=4000, nodesize=20,replace=TRUE,
            importance=TRUE)

rf2$results

#The Sampling increases the RSME, so we keep without sampling
#RMSE      Rsquared  MAE     
#7568.638  0.394712  5245.029

# PARA PLOTEAR EL ERROR OOB A MEDIDA QUE AVANZAN LAS ITERACIONES
# SE USA DIRECTAMENTE EL PAQUETE randomForest

names(airbnbbis)[names(airbnbbis) == "24_hour_check_in"] <- "checkin24h"

rfes<-randomForest(Yearly_Profit~.,
                    data=airbnbbis,
                    mtry=18,ntree=3000,nodesize=20,replace=TRUE)

rfes
plot(rfes$mse,main="RF Early Stopping Study: OBB per Ntree")

#it seems that we could stop the ntree earlier. Let's try zooming it to 500

rfes1<-randomForest(Yearly_Profit~.,
                   data=airbnbbis,
                   mtry=18,ntree=500,nodesize=20,replace=TRUE)

rfes1$
plot(rfes1$mse,main="RF Early Stopping Study: OBB per Ntree - Limit = 500")

#LET'S SEE IF THE RF improves

rfgrid3<-expand.grid(mtry=c(18))

rf3<- train(Yearly_Profit~.,data=airbnbbis,
           method="rf",trControl=control,tuneGrid=rfgrid3,
           linout = TRUE,ntree=400, nodesize=20,replace=TRUE,
           importance=TRUE)

rf3$results

#mtry     RMSE  Rsquared     MAE   RMSESD RsquaredSD    MAESD
#  18 7505.147 0.4031016 5204.11 365.5542 0.03302365 84.77704

# Variables Importance

final<-rf3$finalModel

tabla<-as.data.frame(importance(final))
tabla<-tabla[order(-tabla$IncNodePurity),]
tabla

barplot(tabla$IncNodePurity,names.arg=rownames(tabla),las= 2, col="#FF5A5F", main = "RF: Variables Importance")

#I am not happy with the variables

rfgrid5<-expand.grid(mtry=c(10,12,15,18,20,25))

rf5<- train(Yearly_Profit~minimum_nights+availability_rate+latitude+
              longitude+maximum_nights+ host_identity_verified+is_location_exact+Has_License+
              instant_bookable+Internet+Shampoo+Microwave+Refrigerator+
              Air_conditioning+checkin24h+Laptop_friendly_workspace+
              Hot_water+Coffee_maker+Cooking_basics+Patio_or_balcony+
              Long_term_stays_allowed+Host_greets_you+Yearly_Profit+
              accommodates+cancellation_policy+bathrooms+bedrooms+
              beds+host_response_time+neighbourhood,
            data=airbnbbis,
            method="rf",trControl=control,tuneGrid=rfgrid5,
            linout = TRUE,ntree=400, nodesize=20,replace=TRUE,
            importance=TRUE)

rf5$results

dput(names(airbnbbis))

#  extra_people+minimum_nights+availability_rate+latitude+
#  longitude+maximum_nights+cleaning_fee+security_deposit+
#  host_identity_verified+is_location_exact+Has_License+
#  instant_bookable+Internet+Shampoo+Microwave+Refrigerator+
#  Air_conditioning+checkin24h+Laptop_friendly_workspace+
#  Hot_water+Coffee_maker+Cooking_basics+Patio_or_balcony+
#  Long_term_stays_allowed+Host_greets_you+Yearly_Profit+
#  accommodates+cancellation_policy+bathrooms+bedrooms+
#  beds+host_response_time+neighbourhood


#******************************************************************
#TUNEADO DE GRADIENT BOOSTING CON CARET
# Caret permite tunear estos parámetros básicos:
#  
# 	shrinkage (parámetro v de regularización, mide la velocidad de ajuste, a menor v, más lento y necesita más iteraciones, pero es más fino en el ajuste)
# 	n.minobsinnode: tamaño máximo de nodos finales (el principal parámetro que mide la complejidad)
# 	n.trees=el número de iteraciones (árboles)
# 	interaction.depth (2 para árboles binarios)
#   bag.fraction --> utilizaremos 1, e ir reduciendo
#**************************************************************************************

# Validación cruzada una sola vez


gbmgrid<-expand.grid(shrinkage=c(0.1,0.05,0.03,0.01,0.001,0.2),
                     n.minobsinnode=c(5,10,20,30),
                     n.trees=c(100,300,500,1000,2000,5000),
                     interaction.depth=c(2))


gbm<- train(Yearly_Profit~.,data=airbnbbis,
            method="gbm",trControl=control,tuneGrid=gbmgrid,
            distribution="gaussian", bag.fraction=1,verbose=FALSE)

gbm$results

#shrinkage  n.minobsinnode  n.trees  RMSE      Rsquared   MAE    
#0.100      30              2000     7639.103  0.3746131  5293.577

plot(gbm,main="Airbnb GBM Early Stopping: RSME per Interactions")
#No need of early stopping

# Resampling

gbmgridr<-expand.grid(shrinkage=c(0.1),
                     n.minobsinnode=c(30),
                     n.trees=c(2000),
                     interaction.depth=c(2))


gbmr<- train(Yearly_Profit~.,data=airbnbbis,
            method="gbm",trControl=control,tuneGrid=gbmgridr,
            distribution="gaussian", bag.fraction=0.6,verbose=FALSE)

gbmr$results


# IMPORTANCIA DE VARIABLES

varImp(gbmr)
plot(varImp(gbmr,main="Airbnb GBM: Variables Importance"))

#****************************************************************************
# TUNEADO DE XGBOOST CON CARET
# nrounds (# Boosting Iterations)
# max_depth (Max Tree Depth)
# eta (Shrinkage)
# gamma (Minimum Loss Reduction)
# colsample_bytree (Subsample Ratio of Columns)
# min_child_weight (Minimum Sum of Instance Weight)
# subsample (Subsample Percentage) mostrear observaciones antes de tunearlo
#*****************************************************************************

set.seed(1234)
control<-trainControl(method = "repeatedcv",
                      number=4,repeats=5,
                      savePredictions = "all") 

xgbmgrid <-expand.grid(min_child_weight=20,
                        eta=c(0.001,0.01,0.03,0.05,0.1,0.2,0.3,0.5),
                        nrounds=c(100,300,500,1000,2000,4000,5000),
                        max_depth=c(1,2,4,6),
                        gamma=c(0,0.001,0.01,0.1,1),
                        colsample_bytree=1,
                        subsample=1)

xgbm1<- train(Yearly_Profit~.,data=airbnbbis,
             method="xgbTree",trControl=control,
             tuneGrid=xgbmgrid,objective = "reg:linear",verbose=FALSE,
             alpha=1,lambda=0)

write.table(xgbm1$results,file= "c:\\Users\\pri_p\\OneDrive\\Documentos\\MASTER\\00_TFM\\Airbnb\\xgbm1.txt",sep="\t", quote = FALSE, row.names = F)

xgbm1$results
plot(xgbm)

#The final values used for the model were nrounds = 100, max_depth = 6, eta = 0.1, gamma =
#0, colsample_bytree = 1, min_child_weight = 20 and subsample = 1.
#0.100  6          0.000   100      7407.593  0.40830031   5090.429

xgbmgrid3 <-expand.grid(min_child_weight=20,
                       eta=c(0.1),
                       nrounds=c(20,30,50,100,200),
                       max_depth=c(6),
                       gamma=c(0),
                       colsample_bytree=1,
                       subsample=1)

xgbm3<- train(Yearly_Profit~.,data=airbnbbis,
              method="xgbTree",trControl=control,
              tuneGrid=xgbmgrid3,objective = "reg:linear",verbose=FALSE,
              alpha=1,lambda=0)

xgbm3$results

plot(xgbm3,main = "eta")




# IMPORTANCIA DE VARIABLES

varImp(xgbm1)
plot(varImp(xgbm1))

# PRUEBO PARÁMETROS CON VARIABLES SELECCIONADAS


# **************************
# TUNEADO SVM BINARIA
# **************************

#  SVM LINEAL: SOLO PARÁMETRO C

SVMgridl<-expand.grid(C=c(0.01,0.05,0.1,0.2,0.5,1,2,5,10))

SVMl<- train(data=airbnbbis,Yearly_Profit~.,
             method="svmLinear",trControl=control,
             tuneGrid=SVMgridl,verbose=FALSE)
SVMl

#The final value used for the model was C = 2.
#2.00  8187.234  0.2871540  5578.763

#  SVM RBF: PARÁMETROS C, sigma

SVMrgrid<-expand.grid(C=c(0.01,0.05,0.1,0.2,0.5,1,2,5),
                      sigma=c(0.01,0.05,0.1,0.2,0.5,1,2,5))


SVMr<- train(data=idealistabis,Yearly_Price~.,
             method="svmRadial",trControl=control,
             tuneGrid=SVMrgrid,verbose=FALSE)

SVMr


#The final values used for the model were sigma = 0.01 and C = 2.
# 2.00  0.01   7811.294  0.35001533  5231.136

########## APLICACIÓN CRUZADAS PARA ENSAMBLAR##################################

dput(names(airbnbbis))


archivo<-airbnbbis

vardep<-"Yearly_Profit"

listconti<-c("extra_people", "minimum_nights", "availability_rate","latitude", "longitude",
             "maximum_nights", "cleaning_fee","security_deposit")

listclass<-c("host_identity_verified", "is_location_exact", "Has_License", 
               "instant_bookable", "Internet", "Shampoo", "Microwave", "Refrigerator", 
               "Air_conditioning", "checkin24h", "Laptop_friendly_workspace", 
               "Hot_water", "Coffee_maker", "Cooking_basics", "Patio_or_balcony", 
               "Long_term_stays_allowed", "Host_greets_you","cancellation_policy", "bathrooms", 
               "bedrooms", "beds", "host_response_time", "neighbourhood", "accommodates")

grupos<-4
sinicio<-12345
repe<-10

# Competition and Ensemble models preparation

#Regression

medias1<-cruzadalin(data=archivo,
                    vardep=vardep,listconti=listconti,
                    listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe)

medias1bis<-as.data.frame(medias1[1])
medias1bis$modelo<-"regresion"
predi1<-as.data.frame(medias1[2])
predi1$reg<-predi1$pred

#intercept     RMSE  Rsquared      MAE   RMSESD RsquaredSD    MAESD
#1      TRUE 8099.998 0.2955084 5636.813 358.2516 0.02461688 81.86842


#AVNNET************************************************************************
#The final values used for the model were size = 20, decay = 0.1 and bag = FALSE.
# size  decay  RMSE      Rsquared    MAE  
#  20    0.100  9069.343  0.15580387  6327.109
# 10    0.2

medias2<-cruzadaavnnet(data=archivo,
                         vardep=vardep,listconti=listconti,
                         listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                         size=10,decay=0.2,repeticiones=5,itera=100)

medias2bis<-as.data.frame(medias2)
medias2bis$modelo<-"avnnet"
predi2<-as.data.frame(medias2[2])
predi2$avnnet<-predi2$pred

#size decay   bag     RMSE  Rsquared      MAE   RMSESD RsquaredSD    MAESD
#10   0.2 FALSE 9213.536 0.1465931 6400.036 377.4295 0.04383953 150.2274

#RF******************************************************************************
#mtry=18,ntree=400,nodesize=20

medias3<-cruzadarf(data=archivo,
                   vardep=vardep,listconti=listconti,
                   listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                   mtry=18,ntree=400,nodesize=20,replace=TRUE)


medias3bis<-as.data.frame(medias3[1])
medias3bis$modelo<-"rf"
predi3<-as.data.frame(medias3[2])
predi3$rf<-predi3$pred

#mtry     RMSE  Rsquared      MAE   RMSESD RsquaredSD    MAESD
#1   18 7559.124 0.3955256 5230.347 376.2487 0.02881828 87.61416


#GBM***************************************************************************
#shrinkage  n.minobsinnode  n.trees  
#0.100      30              2000 

medias4<-cruzadagbm(data=archivo,
                    vardep=vardep,listconti=listconti,
                    listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                    n.minobsinnode=30,shrinkage=0.1,n.trees=2000,interaction.depth=2)

medias4bis<-as.data.frame(medias4[1])
medias4bis$modelo<-"gbm"
predi4<-as.data.frame(medias4[2])
predi4$gbm<-predi4$pred

#n.minobsinnode shrinkage n.trees interaction.depth     RMSE  Rsquared      MAE   RMSESD RsquaredSD    MAESD
#1             30       0.1    2000                 2 7653.396 0.3743865 5292.111 349.8109 0.02639858 83.80872

#XGBM************************************************************************
#nrounds = 100, max_depth = 6, eta = 0.1, gamma = 0,
#colsample_bytree = 1, min_child_weight = 20 and subsample = 1

medias5<-cruzadaxgbm(data=archivo,
                     vardep=vardep,listconti=listconti,
                     listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                     min_child_weight=20,eta=0.1,nrounds=100,max_depth=6,
                     gamma=0,colsample_bytree=1,subsample=1)


medias5bis<-as.data.frame(medias5[1])
medias5bis$modelo<-"xgbm"
predi5<-as.data.frame(medias5[2])
predi5$xgbm<-predi5$pred

#    RMSE  Rsquared      MAE   RMSESD RsquaredSD    MAESD1
# 7460.123 0.4024819 5113.431 359.5163 0.02623424 88.45266

#SVM LINEAL***************************************************************
#C = 2

medias6<-cruzadaSVM(data=archivo,
                    vardep=vardep,listconti=listconti,
                    listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                    C=2)

medias6bis<-as.data.frame(medias6[1])
medias6bis$modelo<-"SVM"
predi6<-as.data.frame(medias6[2])
predi6$SVM<-predi6$pred

#  C     RMSE  Rsquared      MAE   RMSESD RsquaredSD    MAESD
# 2 8176.711 0.2888042 5566.073 376.0882  0.0233028 82.02183

#SVM RBF*******************************************************************
#sigma = 0.01 and C = 2

medias7<-cruzadaSVMRBF(data=archivo,
                        vardep=vardep,listconti=listconti,
                        listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                        C=2,sigma=0.01)


medias7bis<-as.data.frame(medias7[1])
medias7bis$modelo<-"SVMRBF"
predi7<-as.data.frame(medias7[2])
predi7$SVMRBF<-predi7$pred

#C sigma     RMSE  Rsquared      MAE   RMSESD RsquaredSD   MAESD
# 2  0.01 7777.456 0.3545937 5213.291 368.6776 0.02344339 89.4413


union1<-rbind(medias1bis,medias2bis,medias3bis,medias4bis,medias5bis,medias6bis,medias7bis)
par(cex.axis=1.5)
boxplot(data=union1,error~modelo,col="#FF5A5F",main="Airbnb Model Competition (RMSE)")


############### Ensembled#########################

# CONSTRUCCIÓN DE TODOS LOS ENSAMBLADOS
# SE UTILIZARÁN LOS ARCHIVOS SURGIDOS DE LAS FUNCIONES LLAMADOS predi1,...

unipredi<-cbind(predi1,predi2,predi3,predi4,predi5,predi6,predi7)

# Esto es para eliminar columnas duplicadas
unipredi<- unipredi[, !duplicated(colnames(unipredi))]

# Ensenbled Models
unipredi$predi10<-(unipredi$rf+unipredi$xgbm)/2
unipredi$predi11<-(unipredi$rf+unipredi$gbm+unipredi$xgbm)/3
unipredi$predi12<-(unipredi$gbm+unipredi$rf+unipredi$xgbm+unipredi$SVMRBF)/4

dput(names(unipredi))


listado<-c("rf", "gbm",  "xgbm", "xgbm","SVMRBF",
           "predi10", "predi11", "predi12")


repeticiones<-nlevels(factor(unipredi$Rep))
unipredi$Rep<-as.factor(unipredi$Rep)
unipredi$Rep<-as.numeric(unipredi$Rep)

# Calculo el MSE para cada repeticion de validación cruzada

medias0<-data.frame(c())

for (prediccion in listado)
{
  paso <-unipredi[,c("obs",prediccion,"Rep")]
  paso$error<-(paso[,c(prediccion)]-paso$obs)^2
  paso<-paso %>%
    group_by(Rep) %>%
    summarize(error=mean(error))     
  paso$modelo<-prediccion  
  medias0<-rbind(medias0,paso) 
} 

# PRESENTACION TABLA MEDIAS

tablamedias<-medias0 %>%
  summarize(error=mean(error))     

tablamedias<-tablamedias[order(tablamedias$error),]

tablamedias

# ORDENACIÓN DEL FACTOR MODELO POR LAS MEDIAS EN ERROR
# PARA EL GRÁFICO

medias0$modelo <- with(medias0,
                       reorder(modelo,error, mean))
par(cex.axis=0.7,las=2)
boxplot(data=medias0,error~modelo,col="#FF5A5F",main="Airbnb Final Competition with ensemble models (RMSE)")

predi12

predict(predi)

