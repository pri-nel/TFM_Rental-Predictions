data airbnb1; set 'C:\Users\pri_p\Desktop\SAS\TFM\DataSources\aibnb_dep_trans_tree_train.sas7bdat' ;
run
;

data airbnb(rename=(PWR_minimum_nights=minimum_nights SQRT_IMP_REP_cleaning_fee=cleaning_fee TG_IMP_REP_bedrooms=bedrooms calculated_host_listings_count_e=host_listings PWR_number_of_reviews=number_of_reviews REP_review_scores_rating=review_scores_rating IMP_REP_bathrooms=bathrooms TG_neighbourhood_group_cleansed=neighbourhood_group PWR_number_of_reviews_ltm=number_of_reviews_ltm SQRT_host_since_days=host_since_days));
set airbnb1;
run;

ods output  SelectedEffects=efectos;
proc glmselect data=airbnb;
    model Yearly_Profit=_NODE_ minimum_nights cleaning_fee bedrooms host_listings number_of_reviews review_scores_rating bathrooms neighbourhood_group number_of_reviews_ltm host_since_days,
   / selection=stepwise(select=AIC choose=AIC);
;
proc print data=efectos;run;
data;set efectos;put effects ;run;

/*  Intercept IMP_REP_bathrooms TG_IMP_REP_bedrooms TG_neighbourhood_gro _NODE_ PWR_minimum_nights PWR_number_of_review REP_review_scores_ra SQRT_IMP_REP_cleanin SQRT_host_since_day
 */


ods output  SelectedEffects=efectos;
proc glmselect data=airbnb;
    model Yearly_Profit=_NODE_ minimum_nights cleaning_fee bedrooms host_listings number_of_reviews review_scores_rating bathrooms neighbourhood_group number_of_reviews_ltm host_since_days,
   / selection=stepwise(select=BIC choose=BIC);
;
proc print data=efectos;run;
data;set efectos;put effects ;run;

/*  Intercept  IMP_REP_bathrooms TG_IMP_REP_bedrooms TG_neighbourhood_gro _NODE_ PWR_minimum_nights PWR_number_of_review REP_review_scores_ra SQRT_IMP_REP_cleanin SQRT_host_since_days
 */

%randomselect(data=airbnb,
listclass=bathrooms bedrooms neighbourhood_group _NODE_,
vardepen=Yearly_Profit,
modelo= _NODE_ minimum_nights cleaning_fee bedrooms host_listings number_of_reviews review_scores_rating bathrooms neighbourhood_group number_of_reviews_ltm host_since_days,
criterio=AIC,
sinicio=12345,
sfinal=12400,
fracciontrain=0.8,directorio=C:\Users\pri_p\OneDrive\Documentos\MASTER\Q2_Machine Learning\output clase);

/* NOS QUEDAMOS CON ESTOS 

s                                                                              efecto                                                                               COUNT    PERCENT

 1     Intercept _NODE_ minimum_nights cleaning_fee bedrooms review_scores_rating bathrooms neighbourhood_group number_of_reviews_lt host_since_days                    48     85.7143
 2     Intercept _NODE_ minimum_nights cleaning_fee bedrooms host_listings review_scores_rating bathrooms neighbourhood_group number_of_reviews_lt host_since_days       8     14.2857

*/

/* Empezamos con validacion cruzada y comparamos los 4 modelos*/

%cruzada(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347);
data final1;set final;modelo=1;

%cruzada(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days host_listings,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347);
data final2;set final;modelo=2;

%cruzada(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee host_listings number_of_reviews review_scores_rating number_of_reviews_ltm host_since_days,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347);
data final3;set final;modelo=3;


data union;set final1 final2 final3;
proc boxplot data=union;plot media*modelo;run;

proc print union; run; 

/* Los modelos son muy parecidos, una vez que solo cambian una variable.

Los modelos 1 y 2 tienen ASE parecidos. Sin embargo,aun que El Modelo 1 tiene uma mayor variabilidade, tiene una variable menos y ASE inferior.
Así que intentaremos ver si mejora con el early stopping y cambiamos a cruzada neural

En cuestion de numero de nodos, el recomendado para nuestro modelo seria 21, así que intentaremos como se portan los datos variando los nodos de 4 a 30.

*/

/* Empezamos por estudiar el Early Stopping con bprop y/o levmar*/

/* 4 nodos*/

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=4,algo=BPROP MOM=0.1 LEARN=0.1,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=4,algo=levmar,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=4,algo=BPROP MOM=0.1 LEARN=0.1,acti=log);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=4,algo=levmar,acti=log);

/* 6 nodos*/

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=6,algo=BPROP MOM=0.1 LEARN=0.1,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=6,algo=levmar,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=6,algo=BPROP MOM=0.1 LEARN=0.1,acti=log);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=6,algo=levmar,acti=log);

/* 10 nodos*/

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=10,algo=BPROP MOM=0.1 LEARN=0.1,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=10,algo=levmar,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=10,algo=BPROP MOM=0.1 LEARN=0.1,acti=log);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=10,algo=levmar,acti=log);

/* 15 nodos*/

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=15,algo=BPROP MOM=0.1 LEARN=0.1,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=15,algo=levmar,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=15,algo=BPROP MOM=0.1 LEARN=0.1,acti=log);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=15,algo=levmar,acti=log);

/* 21 nodos*/

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=21,algo=BPROP MOM=0.1 LEARN=0.1,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=21,algo=levmar,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=21,algo=BPROP MOM=0.1 LEARN=0.1,acti=log);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=21,algo=levmar,acti=log);

/* 30 nodos*/

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=30,algo=BPROP MOM=0.1 LEARN=0.1,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=30,algo=levmar,acti=TANH);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=30,algo=BPROP MOM=0.1 LEARN=0.1,acti=log);

%redneuronal(archivo=airbnb,listclass=_NODE_ bedrooms bathrooms neighbourhood_group,listconti=minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days,
vardep=Yearly_Profit,porcen=0.80,semilla=471145,ocultos=30,algo=levmar,acti=log);



/* 

Los algoritomos con LEVMAR funcionan mucho mejor, así que empezamos por entrenar modelos embadasados en este algoritimo.
Además, ahora nos enfocaremos en algoritmos con apenas 30, 15, 10, 21, una vez que estos valores estan dentro de los limites de obs por parametro y estamos evitar sobre ajustes

*/



/* 30 hidden layers and levmar --> don’t use early stopping with tanh and use of early stopping (27) with log */

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=,algo=levmar,acti=log);
data final4;set final;modelo=4;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=,algo=levmar,acti=TANH);
data final5;set final;modelo=5;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=32,algo=levmar,acti=TANH);
data final6;set final;modelo=6;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=27,algo=levmar,acti=log);
data final7;set final;modelo=7;

/* 15 nodos y levmar --> use early stopping (24) or not for both activation function (log and Tahn) )*/

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=15,early=,algo=levmar,acti=TANH);
data final8;set final;modelo=8;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=15,early=,algo=levmar,acti=log);
data final9;set final;modelo=9;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=15,early=24,algo=levmar,acti=TANH);
data final10;set final;modelo=10;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=15,early=29,algo=levmar,acti=log);
data final11;set final;modelo=11;

/* 10 nodos y levmar --> we evaluate only the early stopping with log (32 or no early stopping)*/

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=10,early=32,algo=levmar,acti=log);
data final12;set final;modelo=12;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=10,early=,algo=levmar,acti=log);
data final13;set final;modelo=13;


/* 21 nodos y levmar --> we investigate if we could stop before the suggestion (28 or 15) with the log function*/

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=21,early=28,algo=levmar,acti=log);
data final14;set final;modelo=14;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=21,early=15,algo=levmar,acti=log);
data final15;set final;modelo=15;


data compare_all;set final1 final4 final5 final6 final7 final8 final9 final10 final11 final12 final13 final14 final15;
proc boxplot data=compare_all;plot media*modelo;run;

/* Volvemos a examinar si nuestro mejor modelo cambia al hacer Bprob*/

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=,algo=bprop mom=0.1 learn=0.1,acti=TANH);
data final16;set final;modelo=16;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=,algo=bprop mom=0.2 learn=0.1,acti=TANH);
data final17;set final;modelo=17;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=,algo=bprop mom=0.5 learn=0.1,acti=TANH);
data final18;set final;modelo=18;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=,algo=bprop mom=0.9 learn=0.1,acti=TANH);
data final19;set final;modelo=19;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=,algo=bprop mom=1 learn=0.1,acti=TANH);
data final20;set final;modelo=20;

%cruzadaneural(archivo=airbnb,vardepen=Yearly_Profit,
conti= minimum_nights cleaning_fee review_scores_rating number_of_reviews_ltm host_since_days ,
categor= _NODE_ bedrooms bathrooms neighbourhood_group,
ngrupos=4,sinicio=12345,sfinal=12347,ocultos=30,early=,algo=bprop mom=1 learn=0.2,acti=TANH);
data final21;set final;modelo=21;

data compare_all;set final1 final5 final16 final17 final18 final19 final20 final21;
proc boxplot data=compare_all;plot media*modelo;run;

data compare_all;set final1 final5 final20;
proc boxplot data=compare_all;plot media*modelo;run;

data compare_all;set final1 final5;
proc boxplot data=compare_all;plot media*modelo;run;
