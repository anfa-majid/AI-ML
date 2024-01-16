import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


df1 = pd.read_csv('C:\\Machine Learning Assignment\\iml-fall-2023-first-challenge\\train.csv')
df2 = pd.read_csv('C:\\Machine Learning Assignment\\iml-fall-2023-first-challenge\\test.csv')

df1_encoded = pd.get_dummies(df1)
df2_encoded = pd.get_dummies(df2)


X = df1_encoded.drop(columns=['hospital_death','RecordID','hospital_id','icu_id','ethnicity_African American','ethnicity_Asian','ethnicity_Caucasian','ethnicity_Hispanic','ethnicity_Native American','ethnicity_Other/Unknown','gender_F','gender_M','icu_admit_source_Accident & Emergency','icu_admit_source_Other ICU','icu_stay_type_readmit','icu_type_CTICU','icu_type_Med-Surg ICU','icu_type_Neuro ICU','apache_3j_bodysystem_Cardiovascular','apache_3j_bodysystem_Hematological','apache_2_bodysystem_Haematologic','apache_2_bodysystem_Undefined Diagnoses','gcs_unable_apache','h1_spo2_max','immunosuppression','icu_admit_source_Other Hospital','icu_stay_type_admit','icu_stay_type_transfer','icu_type_CCU-CTICU','icu_type_CSICU','icu_type_Cardiac ICU','icu_type_MICU','icu_type_SICU','apache_3j_bodysystem_Gastrointestinal','apache_3j_bodysystem_Genitourinary','apache_3j_bodysystem_Gynecological','apache_3j_bodysystem_Musculoskeletal/Skin','apache_3j_bodysystem_Neurological','apache_3j_bodysystem_Respiratory','apache_3j_bodysystem_Trauma','apache_2_bodysystem_Neurologic','apache_2_bodysystem_Renal/Genitourinary','apache_2_bodysystem_Respiratory','apache_2_bodysystem_Trauma','apache_2_bodysystem_Undefined diagnoses','elective_surgery','pre_icu_los_days','apache_2_diagnosis','apache_3j_diagnosis','apache_post_operative','resprate_apache','h1_heartrate_min','h1_mbp_max','h1_mbp_noninvasive_max','h1_sysbp_max','h1_sysbp_noninvasive_max','d1_glucose_max','solid_tumor_with_metastasis','icu_admit_source_Floor','icu_admit_source_Operating Room / Recovery','icu_admit_source_Other Hospital','apache_3j_bodysystem_Metabolic','apache_3j_bodysystem_Sepsis','apache_2_bodysystem_Cardiovascular','apache_2_bodysystem_Metabolic'])
y =df1_encoded['hospital_death']

df2_encoded = df2_encoded.drop(columns=['RecordID','hospital_id','icu_id','ethnicity_African American','ethnicity_Asian','ethnicity_Caucasian','ethnicity_Hispanic','ethnicity_Native American','ethnicity_Other/Unknown','gender_F','gender_M','icu_admit_source_Accident & Emergency','icu_admit_source_Other ICU','icu_stay_type_readmit','icu_type_CTICU','icu_type_Med-Surg ICU','icu_type_Neuro ICU','apache_3j_bodysystem_Cardiovascular','apache_3j_bodysystem_Hematological','apache_2_bodysystem_Haematologic','apache_2_bodysystem_Undefined Diagnoses','gcs_unable_apache','h1_spo2_max','immunosuppression','icu_admit_source_Other Hospital','icu_stay_type_admit','icu_stay_type_transfer','icu_type_CCU-CTICU','icu_type_CSICU','icu_type_Cardiac ICU','icu_type_MICU','icu_type_SICU','apache_3j_bodysystem_Gastrointestinal','apache_3j_bodysystem_Genitourinary','apache_3j_bodysystem_Gynecological','apache_3j_bodysystem_Musculoskeletal/Skin','apache_3j_bodysystem_Neurological','apache_3j_bodysystem_Respiratory','apache_3j_bodysystem_Trauma','apache_2_bodysystem_Neurologic','apache_2_bodysystem_Renal/Genitourinary','apache_2_bodysystem_Respiratory','apache_2_bodysystem_Trauma','apache_2_bodysystem_Undefined diagnoses','elective_surgery','pre_icu_los_days','apache_2_diagnosis','apache_3j_diagnosis','apache_post_operative','resprate_apache','h1_heartrate_min','h1_mbp_max','h1_mbp_noninvasive_max','h1_sysbp_max','h1_sysbp_noninvasive_max','d1_glucose_max','solid_tumor_with_metastasis','icu_admit_source_Floor','icu_admit_source_Operating Room / Recovery','icu_admit_source_Other Hospital','apache_3j_bodysystem_Metabolic','apache_3j_bodysystem_Sepsis','apache_2_bodysystem_Cardiovascular','apache_2_bodysystem_Metabolic'])


knn_imputer = KNNImputer(n_neighbors=20)
X = knn_imputer.fit_transform(X)
df2_encoded = knn_imputer.fit_transform(df2_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df2_encoded_scaled = scaler.fit_transform(df2_encoded)


#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

KNN = KNeighborsClassifier(n_neighbors=500,n_jobs=-1)

KNN.fit(X_scaled,y)

pred = KNN.predict_proba(df2_encoded_scaled)

pred = pred[:,1]

print(pred)

#md_auc = roc_auc_score(y_test,pred)
#print(md_auc)


df_sample = pd.read_csv('C:\\Machine Learning Assignment\\iml-fall-2023-first-challenge\\sample.csv')
df_sample['hospital_death'] = pred
df_sample.to_csv('sample.csv',index=False)



