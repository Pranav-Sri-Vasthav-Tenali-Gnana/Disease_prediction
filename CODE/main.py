import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash
import os
import shutil
from flask_mail import *
from sklearn.datasets._base import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.svm import SVC
from flask import *
import mysql.connector
import os
import random
OTP = random.randint(0000, 9999)

db=mysql.connector.connect(user='root',port=3306,database='disease')
cur=db.cursor()

app = Flask(__name__)

app.config['UPLOAD_FOLDER']=r"E:\Disease_Prediction\CODE\UPLOADS"
app.config['SECRET_KEY']='b0b4fbefdc48be27a6123605f02b6b86'

global data, x_train, x_test, y_train, y_test

TrainData = pd.read_csv('E:\Disease_Prediction\CODE\DATASET\Training.csv')
TestData = pd.read_csv('E:\Disease_Prediction\CODE\DATASET\Testing.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient',methods=['POST','GET'])
def Patientlog():
    if request.method=='POST':
        name=request.form['Name']
        password=request.form['Password']
        OTP = request.form['OTP']
        cur.execute("select * from patientreg where Name='"+name+"' and Password='"+password+"' and OTP='"+OTP+"'")
        content=cur.fetchall()

        db.commit()
        if content == []:
            msg="Credentials Does't exist"
            return render_template('patientlog.html',msg=msg)
        else:
            msg="Login Successful."
            return render_template('patienthome.html',name=name)
    return render_template('patientlog.html')

@app.route('/patientreg',methods=['POST','GET'])
def Patientreg():
    if request.method=='POST':
        name=request.form['Name']
        age=request.form['Age']
        email=request.form['Email']
        password1=request.form['Password']
        password2=request.form['Con_Password']
        if password1 == password2:
            sql="select * from patientreg where Name='%s' and Email='%s'"%(name,email)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print('----',data)
            if data==[]:

                sender_address = 'diseasepredictionproject@gmail.com'
                sender_pass = 'PTSV1234@4'
                content = "Your Request Is Accepted by the Management!! Your OTP is :" + str(OTP)
                receiver_address = email
                message = MIMEMultipart()
                message['From'] = sender_address
                message['To'] = receiver_address
                message['Subject'] = "Disease Prediction"
                message.attach(MIMEText(content, 'plain'))
                ss = smtplib.SMTP('smtp.gmail.com', 587)
                ss.starttls()
                ss.login(sender_address, sender_pass)
                text = message.as_string()
                ss.sendmail(sender_address, receiver_address, text)
                ss.quit()
                sql="insert into patientreg(Name,Age,Email,Password, OTP) values(%s,%s,%s,%s,%s)"
                val=(name,age,email,password1, OTP)
                cur.execute(sql,val)
                db.commit()

                return render_template('patientlog.html')
            else:
                warning='Details already Exist'
                return render_template('patientreg.html',msg=warning)
        error='password not matched'
        flash(error)
    return render_template('patientreg.html')


@app.route('/proceed')
def proceed():
    return render_template('proceed.html')


@app.route('/phome')
def phome():
    return render_template('patienthome.html')



@app.route('/doctorreg',methods=['POST','GET'])
def doctorreg():
    if request.method=='POST':
        name=request.form['Name']
        email=request.form['email']
        password=request.form['password']
        conpassword = request.form['conpassword']
        if password == conpassword:
            print("True")
            sql="select * from doctorreg"
            x=pd.read_sql_query(sql,db)
            all_emails=x['Email'].values
            if email in all_emails:
                    msg="Email already Exist's"
                    return render_template('doctorreg.html',msg=msg)
            else:

                sender_address = 'diseasepredictionproject@gmail.com'
                sender_pass = 'PTSV1234@4'
                content = "Your Request Is Accepted by the Management!! Your OTP is :" + str(OTP)
                receiver_address = email
                message = MIMEMultipart()
                message['From'] = sender_address
                message['To'] = receiver_address
                message['Subject'] = "Disease Prediction"
                message.attach(MIMEText(content, 'plain'))
                ss = smtplib.SMTP('smtp.gmail.com', 587)
                ss.starttls()
                ss.login(sender_address, sender_pass)
                text = message.as_string()
                ss.sendmail(sender_address, receiver_address, text)
                ss.quit()

                sql="insert into doctorreg(Name,Email,Password,OTP) values('%s','%s','%s',%s)"%(name,email,password,OTP)
                cur.execute(sql)
                db.commit()

                msg="Your Request Sent to Management"
                return render_template('doctorreg.html',msg=msg)
        else:
            msg = "Password doesn't Match"
            return render_template('doctorreg.html', msg=msg)

    return render_template('doctorreg.html')




@app.route('/doctor_log',methods=['POST','GET'])
def doctorlog():
    if request.method=='POST':
        name=request.form['Name']
        password=request.form['Password']
        OTP = request.form['OTP']
        cur.execute("select * from doctorreg where Name='" + name + "' and Password='" + password + "' and OTP='" + OTP + "'")
        data=cur.fetchall()
        db.commit()
        print(data)

        if data == []:
            msg = "Credentials Does't exist"
            return render_template('doctorlog.html', msg=msg)
        else:
            msg = "Login Successful."
            return render_template('patienthome.html', msg= msg)
    return render_template('doctorlog.html')







@app.route('/load', methods=["POST","GET"])
def load():
    if request.method=="POST":
        train_file=request.files['ext1']
        test_file=request.files['ext2']
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
            msg='THE DATASET LOADED SUCCESSFULLY.'
            return render_template('preprocess.html', msg= msg)
        else:
            msg='Please upload a txt type documents only.'
            return render_template('load.html', msg= msg)
    return render_template('load.html')




@app.route('/preprocess',methods=['POST','GET'])
def preprocess():
    global pca_x, y, size
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

        # progno_names= pd.DataFrame(progno_names, columns=['progno_names'])
        #
        le= LabelEncoder()
        # progno_names['prognosis'] = le.fit_transform(progno_names['progno_names'])
        # print(progno_names)

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
        return render_template('model.html', msg='Data Preprocessed and It Splits Successfully')

    return render_template('preprocess.html')




@app.route('/model', methods= ['GET','POST'])
def model():

    x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=size, random_state=42)

    if request.method== 'POST':
        model_no= int(request.form['algo'])

        if model_no==0:
            msg= "You have not selected any model"

        elif model_no == 1:
            model = GaussianNB()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            nb = accuracy_score(y_test, pred)*100
            msg = "ACCURACY OF GAUSSIAN NAIVE BAYES IS :" + str(nb) + str('%')


        elif model_no== 2:
            cfr = RandomForestClassifier()
            cfr.fit(x_train, y_train)
            pred = cfr.predict(x_test)
            rfcr= accuracy_score(y_test, pred) *100
            msg= "ACCURACY OF RANDOM FOREST CLASSIFIER IS :"+ str(rfcr)+ str('%')




        elif model_no== 3:
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            pred = dt.predict(x_test)
            dtac = accuracy_score(y_test, pred)*100
            msg = "ACCURACY OF DECISION TREE CLASSIFIER IS :" + str(dtac)+ str('%')



        elif model_no== 4:
            svm = SVC()
            svm.fit(x_train, y_train)
            pred = svm.predict(x_test)
            accsvm = accuracy_score(y_test, pred)*100
            msg = "ACCURACY OF SUPPORT VECTOR MACHINE IS :" + str(accsvm)+ str('%')

        return render_template('model.html', mag = msg)
    return render_template('model.html')





@app.route('/prediction', methods= ['GET', 'POST'])
def prediction():
    global pca_x, y, size, df, x_train,x_test,y_train,y_test
    x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=size, random_state=42)
    if request.method== "POST":
        ITCHING_RELATED= request.form['itching']
        print(ITCHING_RELATED)
        SKIN_RASH_RELATED= request.form['skin rash']
        print(SKIN_RASH_RELATED)
        CONTINUOUS_SNEEZING_RELATED= request.form['continuous sneezing']
        print(CONTINUOUS_SNEEZING_RELATED)
        JOINT_PAIN_RELATED= request.form['joint pain']
        print(JOINT_PAIN_RELATED)
        STOMACH_PAIN_RELATED= request.form['stomach pain']
        print(STOMACH_PAIN_RELATED)
        ACIDITY_RELATED= request.form['acidity']
        print(ACIDITY_RELATED)
        ULCER_ON_TONGUE_RELATED= request.form['ulcers on tongue']
        print(ULCER_ON_TONGUE_RELATED)
        VOMITING_RELATED= request.form['vomiting']
        print(VOMITING_RELATED)
        BURNING_MICTURITION_RELATED= request.form['burning micturition']
        print(BURNING_MICTURITION_RELATED)
        SPOTTING_URINATION_RELATED= request.form['spotting urination']
        print(SPOTTING_URINATION_RELATED)
        FATIGUE_RELATED = request.form['fatigue']
        print(FATIGUE_RELATED)
        WEIGHT_GAIN_RELATED = request.form['weight gain']
        print(WEIGHT_GAIN_RELATED)
        ANXIETY_RELATED = request.form['anxiety']
        print(ANXIETY_RELATED)
        RESTLESSNESS_RELATED = request.form['restlessness']
        print(RESTLESSNESS_RELATED)
        COUGH_RELATED = request.form['cough']
        print(COUGH_RELATED)
        HIGH_FEVER_RELATED = request.form['high fever']
        print(HIGH_FEVER_RELATED)
        BREATHLESSNESS_RELATED = request.form['breathlessness']
        print(BREATHLESSNESS_RELATED)
        DEHYDRATION_RELATED = request.form['dehydration']
        print(DEHYDRATION_RELATED)
        INDIGESTION_RELATED = request.form['indigestion']
        print(INDIGESTION_RELATED)
        DARK_URINE_RELATED = request.form['dark urine']
        print(DARK_URINE_RELATED)
        NAUSEA_RELATED= request.form['nausea']
        print(NAUSEA_RELATED)
        BACK_PAIN_RELATED = request.form['back pain']
        print(BACK_PAIN_RELATED)
        CONSTIPATION_RELATED= request.form['constipation']
        print(CONSTIPATION_RELATED)
        YELLOWING_EYES_RELATED = request.form['yellowing of eyes']
        print(YELLOWING_EYES_RELATED)
        CHEST_PAIN_RELATED = request.form['chest pain']
        print(CHEST_PAIN_RELATED)


        di= {'itching' : [ITCHING_RELATED], 'skin rash' : [SKIN_RASH_RELATED], 'continuous sneezing' : [CONTINUOUS_SNEEZING_RELATED], 'joint pain' : [JOINT_PAIN_RELATED],
             'stomach pain' : [STOMACH_PAIN_RELATED],'acidity' : [ACIDITY_RELATED], 'ulcers on tongue' : [ULCER_ON_TONGUE_RELATED],
             'vomiting' : [VOMITING_RELATED], 'burning micturition' : [BURNING_MICTURITION_RELATED],'spotting urination' : [SPOTTING_URINATION_RELATED],
             'fatigue' :[FATIGUE_RELATED], 'weight gain' :[WEIGHT_GAIN_RELATED], 'anxiety':[ANXIETY_RELATED],'restlessness' :[RESTLESSNESS_RELATED],
             'cough' :[COUGH_RELATED], 'high fever':[HIGH_FEVER_RELATED],'breathlessness' :[BREATHLESSNESS_RELATED], 'dehydration' :[DEHYDRATION_RELATED],
             'indigestion':[INDIGESTION_RELATED],'dark urine':[DARK_URINE_RELATED], 'nausea':[NAUSEA_RELATED],'back pain':[BACK_PAIN_RELATED],
             'constipation': [CONSTIPATION_RELATED], 'yellowing of eyes': [YELLOWING_EYES_RELATED],'chest_pain' :[CHEST_PAIN_RELATED]}

        test= pd.DataFrame.from_dict(di)
        print(test)
        x_train, x_test, y_train, y_test = train_test_split(pca_x, y, test_size=size, random_state=42)
        cfr = RandomForestClassifier()
        model = cfr.fit(x_train, y_train)
        output = model.predict(test)
        print(output)

        if output[0] == 0:
            msg = '(vertigo) Paroymsal Positional Vertigo'

        elif output[0] == 1:
            msg = 'AIDS'

        elif output[0] == 2:
            msg = 'Acne'

        elif output[0] == 3:
            msg = 'Alcoholic hepatitis'

        elif output[0] == 4:
            msg = 'Allergy'

        elif output[0] == 5:
            msg = 'Arthritis'

        elif output[0] == 6:
            msg = 'Bronchial Asthma'

        elif output[0] == 7:
            msg = 'Cervical spondylosis'

        elif output[0] == 8:
            msg = 'Chicken pox'

        elif output[0] == 9:
            msg = 'Chronic cholestasis'

        elif output[0] == 10:
            msg = 'Common Cold'

        elif output[0] == 11:
            msg = 'Dengue'

        elif output[0] == 12:
            msg = 'Diabetes'

        elif output[0] == 13:
            msg = 'Dimorphic hemmorhoids(piles)'

        elif output[0] == 14:
            msg = 'Drug Reaction'

        elif output[0] == 15:
            msg = 'Fungal infection'

        elif output[0] == 16:
            msg = 'GERD'

        elif output[0] == 17:
            msg = 'Gastroenteritis'

        elif output[0] == 18:
            msg = 'Heart attack'

        elif output[0] == 19:
            msg = 'Hepatitis B'

        elif output[0] == 20:
            msg = 'Hepatitis C'

        elif output[0] == 21:
            msg = 'Hepatitis D'

        elif output[0] == 22:
            msg = 'Hepatitis E'

        elif output[0] == 23:
            msg = 'Hypertension'

        elif output[0] == 24:
            msg = 'Hyperthyroidism'

        elif output[0] == 25:
            msg = 'Hypoglycemia'

        elif output[0] == 26:
            msg = 'Hypothyroidism'

        elif output[0] == 27:
            msg = 'Impetigo'

        elif output[0] == 28:
            msg = 'Jaundice'

        elif output[0] == 29:
            msg = 'Malaria'

        elif output[0] == 30:
            msg = 'Migraine'

        elif output[0] == 31:
            msg = 'Osteoarthristis'

        elif output[0] == 32:
            msg = 'Paralysis (brain hemorrhage)'

        elif output[0] == 33:
            msg = 'Peptic ulcer disease'

        elif output[0] == 34:
            msg = 'Pneumonia'

        elif output[0] == 35:
            msg = 'Psoriasis'

        elif output[0] == 36:
            msg = 'Tuberculosis'

        elif output[0] == 37:
            msg = 'Typhoid'

        elif output[0] == 38:
            msg = 'Urinary tract infection'

        elif output[0] == 39:
            msg = 'Varicose veins'

        else:
            msg = 'hepatitis A'


        return render_template('prediction.html', mag=msg)
    return render_template('prediction.html')



@app.route('/chat')
def chat():
    return render_template('Chat.html')



@app.route('/logout')
def logout():
    return redirect(url_for('index'))

if __name__=="__main__":
    app.run(debug=True, port=8000)




