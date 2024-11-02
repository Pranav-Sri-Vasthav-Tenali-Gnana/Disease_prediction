import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash
import os
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from flask import *
import mysql.connector

db=mysql.connector.connect(user='root',port=3306,database='disease')
cur=db.cursor()





app= Flask(__name__)
app.config['UPLOAD_FOLDER']=r"E:\Disease_Prediction\CODE\DATASET"
app.config['SECRET_KEY']='b0b4fbefdc48be27a6123605f02b6b86'

global data, x_train, x_test, y_train, y_test

TrainData = pd.read_csv('E:\\Disease_Prediction\\CODE\\DATASET\\Training.csv')
TestData = pd.read_csv('E:\\Disease_Prediction\\CODE\\DATASET\\Testing.csv')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/load', methods=["POST","GET"])
def load():
    if request.method=="POST":
        train_file=request.files['Training']
        test_file=request.files['Testing']
        ext1=os.path.splitext(train_file.filename)[1]
        ext2 = os.path.splitext(test_file.filename)[1]
        if ext1.lower() == ".csv" and ext2.lower()=='.csv':
            try:
                shutil.rmtree(app.config['UPLOAD_FOLDER'])
            except:
                pass
            os.mkdir(app.config['UPLOAD_FOLDER'])
            train_file.save(os.path.join(app.config['UPLOAD_FOLDER'],'Training.csv'))
            test_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'Testing.csv'))
            flash('The data is loaded successfully','success')
            return render_template('load.html')
        else:
            flash('Please upload a txt type documents only','warning')
            return render_template('load.html')
    return render_template('load.html')

@app.route('/view', methods=['POST', 'GET'])
def view():
    if request.method=='POST':
        myfile=request.form['df']
        if myfile=='0':
            flash(r"Please select an option",'warning')
            return render_template('view.html')
        temp_df= load_data(os.path.join(app.config["UPLOAD_FOLDER"],myfile))
        # full_data=clean_data(full_data)
        return render_template('view.html', col=temp_df.columns.values, df=list(temp_df.values.tolist()))
    return render_template('view.html')

x_train=None; y_train =None;
x_test=None; y_test=None




@app.route('/preprocess',methods=['POST','GET'])
def preprocess():
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        df= pd.concat([TrainData,TestData], axis=0, join='inner')

        progno_names= [['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
               'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
               'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
               'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
               'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
               'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
               'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
               'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
               'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
               'Osteoarthristis', 'Arthritis',
               '(vertigo) Paroymsal  Positional Vertigo', 'Acne','Urinary tract infection', 'Psoriasis', 'Impetigo']]

        progno_names= pd.DataFrame(progno_names, columns=['progno_names'])

        le= LabelEncoder()
        progno_names['prognosis'] = le.fit_transform(progno_names['progno_names'])
        print(progno_names)

        df['prognosis']=le.fit_transform(df['prognosis'])

        col= (['itching','skin rash','continuous sneezing','joint pain','stomach pain','acidity',
       'ulcers on tongue','vomiting','burning micturition','spotting urination','fatigue','weight gain','anxiety',
       'restlessness','cough','high fever','breathlessness','dehydration','indigestion',
       'dark urine','nausea','back pain','constipation','yellowing of eyes','chest pain'])

        x = df.iloc[:,:-1]
        y = df.iloc[:, -1]

        pca= PCA(n_components=25)
        pca.fit(x)
        pca_x= pca.transform(x)

        pca_x= pd.DataFrame(pca_x, columns= col)

        x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=size, random_state=42)
        print(x_train.columns)
        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')

    return render_template('preprocess.html')




@app.route('/training', methods= ['GET','POST'])
def training():
    x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=size, random_state=42)

    if request.method== 'POST':
        model_no= int(request.form['algo'])

        if model_no==0:
            flash(r"You have not selected any model", "info")

        elif model_no == 1:
            model = SVC()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            svcr = accuracy_score(y_test, pred)
            msg = "Accuracy of SVM is :" + str(svcr)


        elif model_no== 2:
            cfr = RandomForestClassifier()
            model = cfr.fit(x_train, y_train)
            pred = model.predict(x_test)
            rfcr= accuracy_score(y_test, pred)
            msg= "Accuracy of Random Forest is :"+ str(rfcr)




        elif model_no== 3:
            xgc = xgb.XGBClassifier()
            model = xgc.fit(x_train, y_train)
            pred = model.predict(x_test)
            xgcr = accuracy_score(y_test, pred)
            msg = "Accuracy of XgBoost is :" + str(xgcr)



        elif model_no== 4:
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            pred_y = dt.predict(x_test)
            acc_dt = accuracy_score(y_test, pred_y)
            msg = "Accuracy of Decision Tree is :" + str(acc_dt)



        elif model_no== 5:
            model = KNeighborsClassifier()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            accuracy_score(y_test, pred)
            kncr = accuracy_score(y_test, pred)
            msg = "Accuracy of KNeighbors is :" + str(kncr)
        return render_template('model.html', mag = msg)
    return render_template('model.html')


@app.route('/prediction', methods= ['GET', 'POST'])
def prediction():
    if request.method== "POST":
        itching= request.form['itching']
        print(itching)
        skin_rash= request.form['skin rash']
        print(skin_rash)
        continuous_sneezing= request.form['continuous sneezing']
        print(continuous_sneezing)
        joint_pain= request.form['joint pain']
        print(joint_pain)
        stomach_pain= request.form['stomach pain']
        print(stomach_pain)
        acidity= request.form['acidity']
        print(acidity)
        ulcers_on_tongue= request.form['ulcers on tongue']
        print(ulcers_on_tongue)
        vomiting= request.form['vomiting']
        print(vomiting)
        burning_micturition= request.form['burning micturition']
        print(burning_micturition)
        spotting_urination= request.form['spotting urination']
        print(spotting_urination)
        fatigue = request.form['fatigue']
        print(fatigue)
        weight_gain = request.form['weight gain']
        print(weight_gain)
        anxiety = request.form['anxiety']
        print(anxiety)
        restlessness = request.form['restlessness']
        print(restlessness)
        cough = request.form['cough']
        print(cough)
        high_fever = request.form['high fever']
        print(high_fever)
        breathlessness = request.form['breathlessness']
        print(breathlessness)
        dehydration = request.form['dehydration']
        print(dehydration)
        indigestion = request.form['indigestion']
        print(indigestion)
        dark_urine = request.form['dark urine']
        print(dark_urine)
        nausea= request.form['nausea']
        print(nausea)
        back_pain = request.form['back pain']
        print(back_pain)
        constipation= request.form['constipation']
        print(constipation)
        yellowing_of_eyes = request.form['yellowing of eyes']
        print(yellowing_of_eyes)
        chest_pain = request.form['chest pain']
        print(chest_pain)


        di= {'itching' : [itching], 'skin rash' : [skin_rash], 'continuous sneezing' : [continuous_sneezing], 'joint pain' : [joint_pain],
             'stomach pain' : [stomach_pain],'acidity' : [acidity], 'ulcers on tongue' : [ulcers_on_tongue],
             'vomiting' : [vomiting], 'burning micturition' : [burning_micturition],'spotting urination' : [spotting_urination],
             'fatigue' :[fatigue], 'weight gain' :[weight_gain], 'anxiety':[anxiety],'restlessness' :[restlessness],
             'cough' :[cough], 'high fever':[high_fever],'breathlessness' :[breathlessness], 'dehydration' :[dehydration],
             'indigestion':[indigestion],'dark urine':[dark_urine], 'nausea':[nausea],'back pain':[back_pain],
             'constipation': [constipation], 'yellowing of eyes': [yellowing_of_eyes],'chest_pain' :[chest_pain]}

        test= pd.DataFrame.from_dict(di)
        print(test)
        x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=size, random_state=42)
        cfr = RandomForestClassifier()
        model = cfr.fit(x_train, y_train)
        output = model.predict(test)
        print(output)

        if output[0] == 'anomaly':
            msg = 'There is a possible <span style = color:red;>INTRUSION DETECTED</span></b> in the system'

        else:
            msg = 'The system is working normally <span style = color:green;>WITHOUT ANY INTRUSION(s)</span></b>'


        return render_template('prediction.html', mag=msg)
    return render_template('prediction.html')



if __name__=='__main__':
    app.run(debug=True)

