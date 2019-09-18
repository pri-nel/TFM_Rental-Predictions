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
idealista<-read.sas7bdat("C:\\Users\\pri_p\\Desktop\\SAS\\TFM\\DataSources\\idealista_vs.sas7bdat")
dput(names(idealista))
summary(idealista)

names(idealista)[names(idealista) == "REP_bathrooms"] <- "bathrooms"
names(idealista)[names(idealista) == "IMP_hasLift"] <- "hasLift"
names(idealista)[names(idealista) == "Yearly_Price"] <- "Yearly_Price"
names(idealista)[names(idealista) == "REP_rooms"] <- "rooms"
names(idealista)[names(idealista) == "REP_G_neighborhood"] <- "neighborhood"
names(idealista)[names(idealista) == "REP_REP_floor"] <- "floor"

dput(names(idealista))


continuas<-c("distance", "latitude", "longitude", "size","SUM")
categoricas<-c("AC", "Piscina","Terraza", "Amueblado","Parking","bathrooms","floor", "rooms", "hasLift", "neighborhood")

# We will standarize the continuous variables, therefore we need calculate the means and the standart deviation

means <-apply(idealista[,continuas],2,mean) 
sds<-sapply(idealista[,continuas],sd) 
idealistabis<-scale(idealista[,continuas], center = means, scale = sds)
numerocont<-which(colnames(idealista)%in%continuas)
idealistabis<-cbind(idealistabis,idealista[,-numerocont])

dput(names(idealistabis))
summary(idealistabis)

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
# nnet: parámetros
#     Number of Hidden Units (size, numeric)
#     Weight Decay (decay, numeric)
# PONER linout = FALSE
# ***************************************************************

nnetgrid <-  expand.grid(size=c(8,10,12,15,18,20,25),decay=c(0.01,0.1,0.001))

rednnet<- train(Yearly_Price~.,data=idealistabis,
                method="nnet",linout = TRUE,maxit=100,trControl=control,tuneGrid=nnetgrid)

rednnet


avnnetgrid <-expand.grid(size=c(8,10,12,15,18),decay=c(0.01,0.1,0.001),bag=FALSE)

redavnnet<- train(Yearly_Price~.,data=idealistabis,
                  method="avNNet",linout = TRUE,maxit=100,trControl=control,repeats=5,tuneGrid=avnnetgrid)

redavnnet 

#The final values used for the model were size = 10, decay = 0.1 and bag = FALSE.
# size  decay  RMSE      Rsquared   MAE 
#  10    0.100  5644.382  0.7035346  4139.805


# ***************************************************************
# Random Forest: TUNNING
# mtry: Number of variable is randomly collected to be sampled at each split time.
# ntree: Number of branches will grow after each time split.
# Sampsize:
# Node size:
# ***************************************************************


set.seed(12345)

rfgrid<-expand.grid(mtry=c(3,5,8,10,12,15))

rf<- train(Yearly_Price~.,data=idealistabis,
           method="rf",trControl=control,tuneGrid=rfgrid,
           linout = TRUE,ntree=1000, nodesize=20,replace=TRUE,
           importance=TRUE)

rf$results


boxplot(data=rf$results,RMSE~modelo,col="#E1F56F",main="RMSE")

# The final value used for the model was mtry = 10. Let's see if we can improve it by sampling.


rfgrid1<-expand.grid(mtry=c(10))

rf1<- train(Yearly_Price~.,data=idealistabis,
           method="rf",trControl=control,tuneGrid=rfgrid1,
           linout = TRUE,ntree=1000, sampsize=5000, nodesize=20,replace=TRUE,
           importance=TRUE)

rf1$results

#Sampling is not making slightly increasing the RSME, we keep it without sampling.


# Now we analyse the OBB interactions within the interaction to see wit 

rfbis<-randomForest(Yearly_Price~.,
                    data=idealistabis,
                    mtry=10,ntree=3000,nodesize=20,replace=TRUE)

rfbis
plot(rfbis$mse,main="RF Early Stopping Study: OBB per Ntree - Limit = 3000")

#it's clear the model is overfitting, we can do early stop.
#First try with 500 trees

rfbis2<-randomForest(Yearly_Price~.,
                     data=idealistabis,
                     mtry=10,ntree=500,nodesize=20,replace=TRUE)

rfbis2
plot(rfbis2$mse,
     main="RF Early Stopping Study: OBB per Ntree - Limit = 500")


# Now we could try to stop at either 400 or 300


rf2<- train(Yearly_Price~.,data=idealistabis,
            method="rf",trControl=control,tuneGrid=rfgrid1,
            linout = TRUE,ntree=300, nodesize=20,replace=TRUE,
            importance=TRUE)

rf2$results
#RMSE      Rsquared   MAE     
#2615.708  0.9016487  1621.735


rf3<- train(Yearly_Price~.,data=idealistabis,
            method="rf",trControl=control,tuneGrid=rfgrid1,
            linout = TRUE,ntree=400, nodesize=20,replace=TRUE,
            importance=TRUE)

rf3
#RMSE      Rsquared   MAE     
#2629.894  0.9003335  1628.938

#The BEST model RF model has ntree=300, mtry=10, nodesize=20

# Finally, we analyse the Variables Importance of this model.

final<-rf2$finalModel

tabla<-as.data.frame(importance(final))
tabla<-tabla[order(-tabla$IncNodePurity),]
tabla

par(mar=c(10, 4, 4, 2),cex.axis=0.7,las=2)
barplot(tabla$IncNodePurity,names.arg=rownames(tabla),
        col="#E4F56A",
        main = "Idealista RF: Variables Importance")


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

gbm<- train(Yearly_Price~.,data=idealistabis,
            method="gbm",trControl=control,tuneGrid=gbmgrid,
            distribution="gaussian", bag.fraction=1,verbose=FALSE)

gbm

plot(gbm, main = "GBM: RSME per Interactions")

#The final values used for the model were n.trees = 5000, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 30.
# 0.100      30              5000     2816.100  0.8851162  1884.352

#Estudio de Resampling
# Probamos a fijar algunos parámetros para ver como evoluciona
# en función de las iteraciones


gbmgridr<-expand.grid(shrinkage=c(0.1),
                     n.minobsinnode=c(30),
                     n.trees=c(5000),
                     interaction.depth=c(2))


gbmr<- train(Yearly_Price~.,data=idealistabis,
            method="gbm",trControl=control,tuneGrid=gbmgridr,
            distribution="gaussian", bag.fraction=0.6,verbose=FALSE)

gbmr

#Our model is a little better, let's test the early stopping
#RMSE      Rsquared   MAE     
#2811.426  0.8854568  1879.755

# ESTUDIO DE EARLY STOPPING

gbmgrides<-expand.grid(shrinkage=c(0.1),
                      n.minobsinnode=c(30),
                      n.trees=c(1000, 5000, 8000, 10000),
                      interaction.depth=c(2))


gbmes<- train(Yearly_Price~.,data=idealistabis,
             method="gbm",trControl=control,tuneGrid=gbmgrides,
             distribution="gaussian", bag.fraction=0.6,verbose=FALSE)

gbmes

plot(gbmes,main="GBM Early Stopping: RSME per Interactions")


#It seems with 10000 trees is better.
#Nonetheless the difference for one with 5000 trees is really small, so we keep with 5000 trees.


# IMPORTANCIA DE VARIABLES

varImp(gbmr)
plot(varImp(gbmr),main = "Idealista GBM: Variables Importance",col="black")

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

xgbmgrid <-expand.grid( min_child_weight=20,
                        eta=c(0.1,0.05,0.03,0.01,0.001,0.2,0.3),
                        nrounds=c(100,300,500,1000,2000,4000,5000),
                        max_depth=6,
                        gamma=0,
                        colsample_bytree=1,
                        subsample=1)

xgbm<- train(Yearly_Price~.,data=idealistabis,
             method="xgbTree",trControl=control,
             tuneGrid=xgbmgrid,objective = "reg:linear",verbose=FALSE,
             alpha=1,lambda=0)

xgbm$results
 
#0.030  2000      2660.699  0.8971928   1679.278

#Test Gama, sampling

xgbmgridg <-expand.grid(min_child_weight=20,
                        eta=0.03,
                        nrounds=2000,
                        max_depth=6,
                        gamma=c(0,0.001,0.01,0.1,1),
                        colsample_bytree1=c(1),
                        subsample=c(1))

xgbmg<- train(Yearly_Price~.,data=idealistabis,
             method="xgbTree",trControl=control,
             tuneGrid=xgbmgridg,objective = "reg:linear",verbose=FALSE,
             alpha=1,lambda=0)
xgbmg

#better gamma =0

plot(xgbm,main="Idealista Xgbm Early stopping ")

# IMPORTANCIA DE VARIABLES

varImp(xgbm)
plot(varImp(xgbm),col="black",main="Idealista Xgbm - Variables Importance")



# **************************
# TUNEADO SVM BINARIA
# **************************


#  SVM LINEAL: SOLO PARÁMETRO C

SVMgridl<-expand.grid(C=c(0.01,0.05,0.1,0.2,0.5,1,2,5,10))

SVMl<- train(data=idealistabis,Yearly_Price~.,
             method="svmLinear",trControl=control,
             tuneGrid=SVMgridl,verbose=FALSE)
SVMl$results

#  SVM RBF: PARÁMETROS C, sigma

SVMrgrid<-expand.grid(C=c(0.01,0.05,0.1,0.2,0.5,1,2,5),
                     sigma=c(0.01,0.05,0.1,0.2,0.5,1,2,5))

SVMr<- train(data=idealistabis,Yearly_Price~.,
            method="svmRadial",trControl=control,
            tuneGrid=SVMrgrid,verbose=FALSE)

SVMr


########## APLICACIÓN CRUZADAS PARA ENSAMBLAR##################################


archivo<-idealistabis
vardep<-"Yearly_Price"
listconti<-c("distance", "latitude", "longitude", "size","SUM")
listclass<-c("AC", "Piscina","Terraza", "Amueblado","Parking","bathrooms","floor", "rooms", "hasLift", "neighborhood")

grupos<-4
sinicio<-1234
repe<-20

# Competition and Ensemble models preparation

medias1<-cruzadalin(data=archivo,
                    vardep=vardep,listconti=listconti,
                    listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe)

medias1bis<-as.data.frame(medias1[1])
medias1bis$modelo<-"regression"
predi1<-as.data.frame(medias1[2])
predi1$reg<-predi1$pred


#The final values used for the model were size = 10, decay = 0.1 and bag = FALSE.
# size  decay  RMSE      Rsquared   MAE 
#  10    0.100  5644.382  0.7035346  4139.805

medias2<-cruzadaavnnet(data=archivo,
                       vardep=vardep,listconti=listconti,
                       listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                       size=c(10),decay=c(0.1),repeticiones=5,itera=100,trace=FALSE)

medias2bis<-as.data.frame(medias2[1])
medias2bis$modelo<-"avnnet"
predi2<-as.data.frame(medias2[2])
predi2$avnnet<-predi2$pred

# RF model has mtry=10,ntree=300, nodesize=20

medias3<-cruzadarf(data=archivo,
                   vardep=vardep,listconti=listconti,
                   listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                   mtry=10,ntree=300,nodesize=20,replace=TRUE)

medias3bis<-as.data.frame(medias3[1])
medias3bis$modelo<-"rf"
predi3<-as.data.frame(medias3[2])
predi3$rf<-predi3$pred

#  mtry     RMSE  Rsquared      MAE   RMSESD  RsquaredSD    MAESD
#   10 2668.869 0.8985962 1683.266 94.58963 0.007184536 40.74811

#GBM************************************************************************
#n.trees = 5000, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 30.
medias4<-cruzadagbm(data=archivo,
                    vardep=vardep,listconti=listconti,
                    listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                    n.minobsinnode=30,shrinkage=0.1,n.trees=5000,interaction.depth=2)

medias4bis<-as.data.frame(medias4[1])
medias4bis$modelo<-"gbm"
predi4<-as.data.frame(medias4[2])
predi4$gbm<-predi4$pred


#XGBM************************************************************************
#eta    nrounds 
#  0.030  2000    
medias5<-cruzadaxgbm(data=archivo,
                     vardep=vardep,listconti=listconti,
                     listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                     min_child_weight=20,eta=0.03,nrounds=2000,max_depth=6,
                     gamma=0,colsample_bytree=1,subsample=1)

medias5bis<-as.data.frame(medias5[1])
medias5bis$modelo<-"xgbm"
predi5<-as.data.frame(medias5[2])
predi5$xgbm<-predi5$pred

#    RMSE  Rsquared      MAE   RMSESD RsquaredSD    MAESD1
# 2641.337 0.898808 1672.982 97.704 0.007628697 39.31427

#SVM LINEAL***************************************************************
#C = 0.01

medias6<-cruzadaSVM(data=archivo,
                    vardep=vardep,listconti=listconti,
                    listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                    C=0.01)

medias6bis<-as.data.frame(medias6[1])
medias6bis$modelo<-"SVM"
predi6<-as.data.frame(medias6[2])
predi6$SVM<-predi6$pred

#  C     RMSE  Rsquared      MAE   RMSESD RsquaredSD    MAESD
# 0.01 4196.449 0.760319 2526.889 1059.816 0.08737969 530.3199

#SVM RBF*******************************************************************
#sigma= 0.05 and C = 5

medias7<-cruzadaSVMRBF(data=archivo,
                       vardep=vardep,listconti=listconti,
                       listclass=listclass,grupos=grupos,sinicio=sinicio,repe=repe,
                       C=5,sigma=0.05)

medias7bis<-as.data.frame(medias7[1])
medias7bis$modelo<-"SVMRBF"
predi7<-as.data.frame(medias7[2])
predi7$SVMRBF<-predi7$pred

#C sigma     RMSE  Rsquared      MAE   RMSESD RsquaredSD   MAESD
# 5  0.05 3054.813 0.8699907 1858.496 805.4191 0.03813494 558.5084

R2=


union<-rbind(medias1bis,medias2bis,medias3bis,medias4bis,medias5bis,medias6bis,medias7bis)
par(cex.axis=1.5)
boxplot(data=union,error~modelo,col="#E1F56F",main="Idealista Model Competition (RMSE)")

union1<-rbind(medias3bis,medias4bis,medias5bis,medias7bis)
par(cex.axis=1.5)
boxplot(data=union1,error~modelo,col="#E1F56F",main="Idealista Model Competition (RMSE)")

union2<-rbind(medias3bis,medias5bis)
par(cex.axis=1.5)
boxplot(data=union2,error~modelo,col="#E1F56F",main="Idealista Model Competition (RMSE)")

############### Ensembled#########################

# CONSTRUCCIÓN DE TODOS LOS ENSAMBLADOS
# SE UTILIZARÁN LOS ARCHIVOS SURGIDOS DE LAS FUNCIONES LLAMADOS predi1,...

unipredi<-cbind(predi1,predi2,predi3,predi4,predi5,predi6,predi7)

# Esto es para eliminar columnas duplicadas
unipredi<- unipredi[, !duplicated(colnames(unipredi))]

# Ensembled Models
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
boxplot(data=medias0,error~modelo,col="#E1F56F",main="Idealista Final Competition with ensemble models (RMSE)")