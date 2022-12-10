import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, plot_roc_curve, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


data = pd.read_csv('https://raw.githubusercontent.com/yunus-shariff/Customer-Retention/main/Telco%20Customer%20Churn.csv')
data = data.replace(r'^\s*$', np.nan, regex=True) #Replace missing values with NaN


st.title("Customer Retention Analysis Project [C.R.A.P.]")
st.header("What is Customer Churn?");

st.markdown("* Customer churn is the percentage of customers that stopped using company's product or service during a certain time frame.");
st.markdown("* Customer churn is an important metric for a growing business as it is less expensive to retain existing customers");
st.markdown("* This dataset explores the same in the telecom industry where customers can choose from a variety of service providers");
st.markdown("* As this is a highly competitive market, average rate of churn can range between 15-25%");
st.image('https://inmoment.com/wp-content/uploads/2021/05/Customer-Churn-Rate.png', caption = "You should know what it means, it comes up a LOT.")



st.subheader("Features:")
st.text("Features are listed below, categorized based on the type of information stored")
st.markdown("   `Customer ID`: A unique ID that identifies each customer.")
st.markdown("**Demographic Info:**")
st.markdown("*   `gender`: Whether the customer is a male or a female")
st.markdown("*   `SeniorCitizen`: Whether the customer is a senior citizen or not (1, 0)")
st.markdown("*   `Partner`: Whether the customer has a partner or not (Yes, No)")
st.markdown("*   `Dependents`: Whether the customer has dependents or not (Yes, No)")
st.markdown("**Services enlisted:**")
st.markdown("*   `PhoneService`: Whether the customer has a phone service or not (Yes, No)")
st.markdown("*   `MultipleLines`: Whether the customer has multiple lines or not (Yes, No, No phone service)")
st.markdown("*   `InternetService`: Customer’s internet service provider (DSL, Fiber optic, No)")
st.markdown("*   `OnlineSecurity`: Whether the customer has online security or not (Yes, No, No internet service)")
st.markdown("*   `TechSupport`: Whether the customer has tech support or not (Yes, No, No internet service)")
st.markdown("*   `StreamingTV`: Whether the customer has streaming TV or not (Yes, No, No internet service)")
st.markdown("**Account Info:**")
st.markdown("*   `tenure`: Number of months the customer has stayed with the company")
st.markdown("*   `Contract`: The contract term of the customer (Month-to-month, One year, Two year)")
st.markdown("*   `PaperlessBilling`: Whether the customer has paperless billing or not (Yes, No)")
st.markdown("*   `PaymentMethod`: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))")
st.markdown("*   `MonthlyCharges`: The amount charged to the customer monthly")
st.markdown("*   `TotalCharges`: The total amount charged to the customer")
st.markdown("*   **`Churn`**: Target, Whether the customer has left within the last month or not (Yes or No) \n\n")


st.write('\n\n')
st.write('\nA snapshot of the dataset is displayed below:')    
st.write(data.head(8))

# data.shape

st.text('There are 7043 cutomers and 21 features in the dataset.')


st.subheader('Missingness')
st.markdown('`TotalCharges` is the only attribute with missing values')
f = msno.matrix(data)
st.pyplot(f.figure)
# st.text("Missing values are handled by replacing with NaNs")

st.markdown("If we examine the data carefully, we can actually estimate the value of the missing data. Contract length in month * tenure (if not 0) * monthly charges can provide total charges. This is more accurate than filling missing values with mean or median.")
ind = data[data['TotalCharges'].isnull()].index.tolist()
for i in ind:
  if data['Contract'].iloc[i,] == 'Two year':
    data['TotalCharges'].iloc[i,] = int(np.maximum(data['tenure'].iloc[i,], 1)) * data['MonthlyCharges'].iloc[i,] * 24
  elif data['Contract'].iloc[i,] == 'One year':
    data['TotalCharges'].iloc[i,] = int(np.maximum(data['tenure'].iloc[i,], 1)) * data['MonthlyCharges'].iloc[i,] * 12
  else:
    data['TotalCharges'].iloc[i,] = int(np.maximum(data['tenure'].iloc[i,], 1)) * data['MonthlyCharges'].iloc[i,]

plots = st.selectbox('Plot Type', ('Gender vs. Churn', 'Sunburst Plot', 'Customer Contract Distribution', 'Payment Method Distribution', 'Internet Service', 'Dependents', 'Partner', 'Paperless Billing'))


if plots == 'Gender vs. Churn':

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=data['gender'].unique(), values=data['gender'].value_counts(), name='Gender', 
                         legendgroup = '1', marker_colors=['xkcd:rose pink', 'xkcd:denim']), 1, 1)
    fig.add_trace(go.Pie(labels=data['Churn'].unique(), values=data['Churn'].value_counts(), name='Churn', 
                       legendgroup = '2',  marker_colors=['xkcd:cerulean', 'xkcd:greenish']), 1, 2)
    fig.update_traces(hole=0.5, textfont_size=20, marker=dict(line=dict(color='black', width=2)))
    
    fig.update_layout(
        title_text='<b>Gender and Churn Distributions<b>', 
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                     dict(text='Churn', x=0.83, y=0.5, font_size=20, showarrow=False)])
    fig
    
    
    st.subheader("Observations:")
    st.markdown('* Data is imbalanced.')
    st.markdown('* $26.6 \%$ of customers switched to another company')
    st.markdown('* Customers are $49.5 \%$ female and $50.5\%$ male')
    
elif plots == 'Sunburst Plot':    

    fig = px.sunburst(data, path=['Churn', 'gender'], title='<b>Sunburst Plot of Gender and churn<b>')
    fig 
    
    st.text(f'A female customer has a probability of {round(data[(data["gender"] == "Female") & (data["Churn"] == "Yes")].count()[0] / data[(data["gender"] == "Female")].count()[0] *100,2)} % churn')
    st.text(f'A male customer has a probability of {round(data[(data["gender"] == "Male") & (data["Churn"] == "Yes")].count()[0] / data[(data["gender"] == "Male")].count()[0]*100,2)} % churn')
    st.text('Also, gender does not have any impact on the percentage of customers migrating to another service provider ')
    
elif plots == 'Customer Contract Distribution':  
    
    fig = px.histogram(data, x='Churn', color='Contract', barmode='group', title='<b>Customer Contract Distribution in relation to Churn<b>', 
                       color_discrete_sequence = ['#EC7063','#E9F00B','#0BF0D1'], text_auto=True)
    fig.update_traces(marker_line_width=2,marker_line_color='black')
    fig
    
    st.text(f'A customer with month-to-month contract has a probability of {round(data[(data["Contract"] == "Month-to-month") & (data["Churn"] == "Yes")].count()[0] / data[(data["Contract"] == "Month-to-month")].count()[0] *100,2)} % churn')
    st.text(f'A customer with one year contract has a probability of {round(data[(data["Contract"] == "One year") & (data["Churn"] == "Yes")].count()[0] / data[(data["Contract"] == "One year")].count()[0]*100,2)} % churn')
    st.text(f'A customer with two year contract has a probability of {round(data[(data["Contract"] == "Two year") & (data["Churn"] == "Yes")].count()[0] / data[(data["Contract"] == "Two year")].count()[0]*100,2)} % churn')
    st.text('A majority of employees who left the company had Month-to-Month Contracts.')
    st.text('This is actually logical since people with long-term contracts are more loyal')


elif plots == 'Payment Method Distribution':  
        
    fig = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
    
    fig.add_trace(go.Pie(labels=data['PaymentMethod'].unique(), values=data['PaymentMethod'].value_counts(), name='Payment Method',
                         marker_colors=['gold', 'mediumturquoise','darkorange', 'lightgreen']), 1, 1)
    
    fig.update_traces(hole=0.5, textfont_size=20, marker=dict(line=dict(color='black', width=2)))
    
    fig.update_layout(
        title_text='<b>Payment Method Distributions<b>', 
        annotations=[dict(text='Payment Method', x=0.5, y=0.5, font_size=18, showarrow=False)])
    fig


    
    fig = px.histogram(data, x='Churn', color='PaymentMethod', barmode='group', title='<b>Payment Method Distribution in reation to Churn<b>', 
                       color_discrete_sequence = ['#EC7063', '#0BF0D1', '#E9F00B', '#5DADE2'], text_auto=True)
    
    fig.update_layout(width=1100, height=500, bargap=0.3)
    fig.update_traces(marker_line_width=2,marker_line_color='black')
    fig
    
    st.subheader('Observations:')
    st.text(f'A customer that use Electronic check for paying has a probability of {round(data[(data["PaymentMethod"] == "Electronic check") & (data["Churn"] == "Yes")].count()[0] / data[(data["PaymentMethod"] == "Electronic check")].count()[0] *100,2)} % churn')
    
    st.text(f'A customer that use Mailed check for paying has a probability of {round(data[(data["PaymentMethod"] == "Mailed check") & (data["Churn"] == "Yes")].count()[0] / data[(data["PaymentMethod"] == "Mailed check")].count()[0]*100,2)} % churn')
    
    st.text(f'A customer that use Bank transfer (automatic) for paying has a probability of {round(data[(data["PaymentMethod"] == "Bank transfer (automatic)") & (data["Churn"] == "Yes")].count()[0] / data[(data["PaymentMethod"] == "Bank transfer (automatic)")].count()[0]*100,2)} % churn')
    
    st.text(f'A customer that use Credit card (automatic) for paying has a probability of {round(data[(data["PaymentMethod"] == "Credit card (automatic)") & (data["Churn"] == "Yes")].count()[0] / data[(data["PaymentMethod"] == "Credit card (automatic)")].count()[0]*100,2)} % churn')

    st.text('\nWhat have we learnt?')
    st.markdown("* Majority of customers who moved out had Electronic Check as Payment Method")
    st.markdown("* Customers who chose Credit-Card automatic transfer or Bank Automatic Transfer and Mailed Check as Payment Method were less likely to move out")


# st.write(data[data['gender']=='Male'][['InternetService', 'Churn']].value_counts())
# st.write(data[data['gender']=='Female'][['InternetService', 'Churn']].value_counts())

elif plots == 'Internet Service':  
        
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
      x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
           ['Female', 'Male', 'Female', 'Male']],
      y = [965, 992, 219, 240],
      name = 'DSL', 
    ))
    
    fig.add_trace(go.Bar(
      x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
           ['Female', 'Male', 'Female', 'Male']],
      y = [889, 910, 664, 633],
      name = 'Fiber optic',
    ))
    
    fig.add_trace(go.Bar(
      x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
           ['Female', 'Male', 'Female', 'Male']],
      y = [690, 717, 56, 57],
      name = 'No Internet',
    ))
    
    fig.update_layout(title_text='<b>Churn Distribution in relation to Internet Service and Gender</b>')
    fig.update_traces(marker_line_width=2,marker_line_color='black')
    fig
    
    st.markdown("* A lot of customers choose the Fiber optic service and it's also evident that the customers who use Fiber optic have high churn rate.")
    st.markdown("* This might suggest a dissatisfaction with this type of internet service")
    st.markdown("* Customers having DSL service are majority in number and have less churn rate compared to Fibre optic service")

elif plots == 'Dependents':  
    
    fig = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
    
    fig.add_trace(go.Pie(labels=data['Dependents'].unique(), values=data['Dependents'].value_counts(), name='Dependents',
                         marker_colors=['#E5527A ', '#AAB7B8']), 1, 1)
    
    fig.update_traces(hole=0.5, textfont_size=20, marker=dict(line=dict(color='black', width=2)))
    
    fig.update_layout(
        title_text='<b>Dependents Distribution<b>', 
        annotations=[dict(text='Dependents', x=0.5, y=0.5, font_size=18, showarrow=False)])
    fig

    fig = px.histogram(data, x='Dependents', color='Churn', barmode='group', title='<b>Dependents Distribution in relation to Churn<b>', 
                       color_discrete_sequence = ['#00CC96','#FFA15A'], text_auto=True)
    
    fig.update_layout(width=1100, height=500, bargap=0.3)
    fig.update_traces(marker_line_width=2,marker_line_color='black')
    fig
    
    st.text("Customers without dependents are more likely to churn")
    
elif plots == 'Partner':  
    fig = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
    
    fig.add_trace(go.Pie(labels=data['Partner'].unique(), values=data['Partner'].value_counts(), name='Partner',
                         marker_colors=['gold', 'purple']), 1, 1)
    
    fig.update_traces(hole=0.5, textfont_size=20, marker=dict(line=dict(color='black', width=2)))
    
    fig.update_layout(
        title_text='<b>Partner Distribution<b>', 
        annotations=[dict(text='Partner', x=0.5, y=0.5, font_size=18, showarrow=False)])
    fig


    fig = px.histogram(data, x='Churn', color='Partner', barmode='group', title='<b>Partner Distribution in relation to Churn<b>', 
                       color_discrete_sequence = ['#C82735','#BCC827'], text_auto=True)
    
    fig.update_layout(width=1100, height=500, bargap=0.3)
    fig.update_traces(marker_line_width=2,marker_line_color='black')
    
    fig
    st.text("Customers without partners are more likely to churn")

elif plots == 'Paperless Billing':  
    fig = px.histogram(data, x='Churn', color='PaperlessBilling', barmode='group', title='<b>Paperless Billing Distribution in relation to Churn<b>', 
                       color_discrete_sequence = ['#9FE2BF', '#FF7F50'], text_auto=True)
    
    fig.update_layout(width=1100, height=500, bargap=0.3)
    fig.update_traces(marker_line_width=2,marker_line_color='black')
    
    fig
    
    st.text("Customers with Paperless Billing are most likely to churn.")

st.subheader("Outlier Detection")
data=data.drop(labels=['customerID'],axis=1)
st.image("https://miro.medium.com/proxy/1*ghJQrcLZXGWxDPzppLWULA.png", caption="Because I don't understand it myself")



fig = make_subplots(rows=1, cols=3)
fig.add_trace(go.Box(y=data['MonthlyCharges'], notched=True, name='Monthly Charges', marker_color = '#6699ff', 
                     boxmean=True, boxpoints='suspectedoutliers'), 1, 2)
fig.add_trace(go.Box(y=data['TotalCharges'], notched=True, name='Total Charges', marker_color = '#ff0066', 
                     boxmean=True, boxpoints='suspectedoutliers'), 1, 1)
fig.add_trace(go.Box(y=data['tenure'], notched=True, name='Tenure', marker_color = 'lightseagreen', 
                     boxmean=True, boxpoints='suspectedoutliers'), 1, 3)
fig.update_layout(title_text='<b>Box Plots for Numerical Variables<b>')
fig

st.markdown("**There is no outlier.**")

st.subheader("Balancing Act")
categorical = [var for var in data.columns if data[var].dtype=='O'];
data['Churn'] = data['Churn'].map({'Yes':1,'No':0})


def category(df):
    for var in categorical:
        ordered_labels = df.groupby([var])['Churn'].mean().sort_values().index

        ordinal_label = {k:i for i, k in enumerate(ordered_labels, 0)} 
        df[var] = df[var].map(ordinal_label)

category(data);

fig = px.bar(x=data['Churn'].unique()[::-1], y=[data[data['Churn']==1].count()[0], data[data['Churn']==0].count()[0]],
       text=[np.round(data[data['Churn']==1].count()[0]/data.shape[0], 4), np.round(data[data['Churn']==0].count()[0]/data.shape[0], 4)]
       , color_discrete_sequence =['#ff9999'])

fig.update_layout(title_text='<b>Churn Count Plot<b>', xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1),
                  width=700, height=400, bargap=0.4)

fig.update_layout({'yaxis': {'title':'Count'}, 'xaxis': {'title':'Churn'}})
fig
st.markdown("As shown in the plot above, we are dealing with an imbalanced dataset.The `BorderlineSMOTE` method is used which involves selecting those instances of the minority class that are misclassified, such as with a k-nearest neighbor classification model. This method oversamples just those difficult instances, providing more resolution only where it may be required.");


X = data.drop(['Churn'], axis = 1);
y = data['Churn'];
oversample = BorderlineSMOTE()
X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# X_train.shape, X_test.shape

scaler = StandardScaler()
X_train[['TotalCharges','MonthlyCharges','tenure']] = scaler.fit_transform(X_train[['TotalCharges','MonthlyCharges','tenure']])
X_test[['TotalCharges','MonthlyCharges','tenure']] = scaler.transform(X_test[['TotalCharges','MonthlyCharges','tenure']]) 

CV = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)


st.subheader('Feature Engineering')

RF_I = RandomForestClassifier(n_estimators=70, random_state=42)
RF_I.fit(X, y)
d = {'Features': X_train.columns, 'Feature Importance': RF_I.feature_importances_}
df = pd.DataFrame(d)
df_sorted = df.sort_values(by='Feature Importance', ascending = True)
# df_sorted
df_sorted.style.background_gradient(cmap='Blues')

fig = px.bar(x=df_sorted['Feature Importance'], y=df_sorted['Features'], color_continuous_scale=px.colors.sequential.Blues,
             text_auto='.4f', color=df_sorted['Feature Importance'])

fig.update_traces(marker=dict(line=dict(color='black', width=2)))
fig.update_layout({'yaxis': {'title':'Features'}, 'xaxis': {'title':'Feature Importance'}})
fig



classifier_type = st.selectbox('Classifier', ('Logistic Regression', 'Random Forest', 'KNN', 'Decision Tree'))

if classifier_type == 'Logistic Regression':

    # LR
    
    
    LR_S = LogisticRegression(random_state = 42)
    params_LR = {'C': list(np.arange(1,12)), 'penalty': ['l2', 'elasticnet', 'none'], 'class_weight': ['balanced','None']}
    est_values = st.slider( 'Random state', 0, 120, (42))
    penalty_values = st.slider('Penalty',0, 100,6)
    
    grid_LR = RandomizedSearchCV(LR_S, param_distributions=params_LR, cv=5, n_jobs=-1, n_iter=20, random_state= 42, return_train_score=True)
    grid_LR.fit(X_train, y_train)
    st.write('Best parameters:', grid_LR.best_estimator_)
    LR = LogisticRegression(random_state = est_values, penalty= 'l2', class_weight= 'balanced', C=penalty_values)
    fitting =  LogisticRegression(random_state = est_values, penalty= 'l2', class_weight= 'balanced', C=penalty_values).fit(X_train,y_train)
    y_pred = fitting.predict(X_test)
    cross_val_LR_Acc = cross_val_score(LR, X_train, y_train, cv = CV, scoring = 'accuracy') 
    cross_val_LR_f1 = cross_val_score(LR, X_train, y_train, cv = CV, scoring = 'f1')
    cross_val_LR_AUC = cross_val_score(LR, X_train, y_train, cv = CV, scoring = 'roc_auc')
    
        
    # Confusion Matrix
    # y_pred = LR.fit(X,y)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype(int)
           
    fig = ff.create_annotated_heatmap(z=cm[::-1], x=['No','Yes'], y=['Yes', 'No'], colorscale='Blues', annotation_text=cm[::-1]) 
    
    fig.update_layout(title_text='<b>Confusion Matrix <b>',
                      xaxis_title = 'Predicted value', yaxis_title = 'Real value', width=800, height=500)
    fig
    
    # Classification report
    # # y_pred = LR.fit(X,y)
    # cr = classification_report(y_test, y_pred)
    # # cr = cr.astype(int)
           
    # fig = ff.create_annotated_heatmap(z=cr[::-1], x=['No','Yes'], y=['Yes', 'No'], colorscale='Blues', annotation_text=cr[::-1]) 
    
    # fig.update_layout(title_text='<b>Colassification report of Stacking Model<b>',
    #                   xaxis_title = 'Predicted value', yaxis_title = 'Real value', width=800, height=500)
    # fig
    # st.write(classification_report(y_test,y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write('Classificaton Report', report_df)
    
elif classifier_type == 'Random Forest':

# RF
    est_values = st.slider( 'Estimators', 50, 100, (70))
    rand_state = st.slider( 'Random state', 0, 120, (42))
    RF_S = RandomForestClassifier(random_state = 42)
    params_RF = {'n_estimators': list(range(50,100)), 'min_samples_leaf': list(range(1,5)), 'min_samples_split': list(range(1,5))}
    grid_RF = RandomizedSearchCV(RF_S, param_distributions=params_RF, cv=5, n_jobs=-1, n_iter=20, random_state=42, return_train_score=True)
    grid_RF.fit(X_train, y_train)
    st.write('Best parameters:', grid_RF.best_estimator_)
    
    fitting =  RandomForestClassifier(random_state = rand_state, max_depth = 2).fit(X_train, y_train)
    y_pred = fitting.predict(X_test)
    RF = RandomForestClassifier(n_estimators=est_values, random_state=rand_state)
    cross_val_RF_Acc = cross_val_score(RF, X_train, y_train, cv = CV, scoring = 'accuracy') 
    cross_val_RF_f1 = cross_val_score(RF, X_train, y_train, cv = CV, scoring = 'f1')
    cross_val_RF_AUC = cross_val_score(RF, X_train, y_train, cv = CV, scoring = 'roc_auc')
    
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype(int)
           
    fig = ff.create_annotated_heatmap(z=cm[::-1], x=['No','Yes'], y=['Yes', 'No'], colorscale='Blues', annotation_text=cm[::-1]) 
    
    fig.update_layout(title_text='<b>Confusion Matrix <b>',
                      xaxis_title = 'Predicted value', yaxis_title = 'Real value', width=800, height=500)
    fig
    
    # Classification report
    # # y_pred = LR.fit(X,y)
    # cr = classification_report(y_test, y_pred)
    # # cr = cr.astype(int)
           
    # fig = ff.create_annotated_heatmap(z=cr[::-1], x=['No','Yes'], y=['Yes', 'No'], colorscale='Blues', annotation_text=cr[::-1]) 
    
    # fig.update_layout(title_text='<b>Colassification report of Stacking Model<b>',
    #                   xaxis_title = 'Predicted value', yaxis_title = 'Real value', width=800, height=500)
    # fig
    # st.write(classification_report(y_test,y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write('Classificaton Report', report_df)
    
elif classifier_type == 'KNN':

    # KNN
    
    neighbors = st.slider('Neighbors',1,20,1)
    KNN_S = KNeighborsClassifier()
    params_KNN = {'n_neighbors': list(range(1,20))}
    grid_KNN = RandomizedSearchCV(KNN_S, param_distributions=params_KNN, cv=5, n_jobs=-1, n_iter=20, random_state=42, return_train_score=True)
    grid_KNN.fit(X_train, y_train)
    st.write('Best parameters:', grid_KNN.best_estimator_)
    KNN = KNeighborsClassifier(n_neighbors= neighbors)
    fitting = KNeighborsClassifier(n_neighbors= neighbors).fit(X_train, y_train)
    y_pred = fitting.predict(X_test)
    cross_val_KNN_Acc = cross_val_score(KNN, X_train, y_train, cv = CV, scoring = 'accuracy') 
    cross_val_KNN_f1 = cross_val_score(KNN, X_train, y_train, cv = CV, scoring = 'f1')
    cross_val_KNN_AUC = cross_val_score(KNN, X_train, y_train, cv = CV, scoring = 'roc_auc')

    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype(int)
           
    fig = ff.create_annotated_heatmap(z=cm[::-1], x=['No','Yes'], y=['Yes', 'No'], colorscale='Blues', annotation_text=cm[::-1]) 
    
    fig.update_layout(title_text='<b>Confusion Matrix <b>',
                      xaxis_title = 'Predicted value', yaxis_title = 'Real value', width=800, height=500)
    fig
    
    # Classification report
    # # y_pred = LR.fit(X,y)
    # cr = classification_report(y_test, y_pred)
    # # cr = cr.astype(int)
           
    # fig = ff.create_annotated_heatmap(z=cr[::-1], x=['No','Yes'], y=['Yes', 'No'], colorscale='Blues', annotation_text=cr[::-1]) 
    
    # fig.update_layout(title_text='<b>Colassification report of Stacking Model<b>',
    #                   xaxis_title = 'Predicted value', yaxis_title = 'Real value', width=800, height=500)
    # fig
    # st.write(classification_report(y_test,y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write('Classificaton Report', report_df)
     
elif classifier_type == 'Decision Tree':

    # Decision Tree
    
    # est_values = st.slider( 'Estimators', 50, 100, (70))
    rand_state = st.slider( 'Random state', 0, 120, (42))
    
    DT_S = DecisionTreeClassifier(random_state=42)
    params_DT = {'min_samples_leaf': list(range(1,6)), 'min_samples_split': list(range(1,6))}
    grid_DT = RandomizedSearchCV(DT_S, param_distributions=params_DT, cv=5, n_jobs=-1, n_iter=20, random_state=42, return_train_score=True)
    grid_DT.fit(X_train, y_train)
    st.write('Best parameters:', grid_DT.best_estimator_)
    DT = DecisionTreeClassifier(random_state=rand_state )
    fitting = DecisionTreeClassifier(random_state=rand_state).fit(X_train, y_train)
    y_pred = fitting.predict(X_test)
    cross_val_DT_Acc = cross_val_score(DT, X_train, y_train, cv = CV, scoring = 'accuracy') 
    cross_val_DT_f1 = cross_val_score(DT, X_train, y_train, cv = CV, scoring = 'f1')
    cross_val_DT_AUC = cross_val_score(DT, X_train, y_train, cv = CV, scoring = 'roc_auc')
    
    
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype(int)
           
    fig = ff.create_annotated_heatmap(z=cm[::-1], x=['No','Yes'], y=['Yes', 'No'], colorscale='Blues', annotation_text=cm[::-1]) 
    
    fig.update_layout(title_text='<b>Confusion Matrix <b>',
                      xaxis_title = 'Predicted value', yaxis_title = 'Real value', width=800, height=500)
    fig
    
    # Classification report
    # # y_pred = LR.fit(X,y)
    # cr = classification_report(y_test, y_pred)
    # # cr = cr.astype(int)
           
    # fig = ff.create_annotated_heatmap(z=cr[::-1], x=['No','Yes'], y=['Yes', 'No'], colorscale='Blues', annotation_text=cr[::-1]) 
    
    # fig.update_layout(title_text='<b>Colassification report of Stacking Model<b>',
    #                   xaxis_title = 'Predicted value', yaxis_title = 'Real value', width=800, height=500)
    # fig
    # st.write(classification_report(y_test,y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write('Classificaton Report', report_df)


#Choose estimators:

# estimators = [('DT', DT),
#               ('RF', RF),
#               ('LR', LR),
#               ('KNN', KNN)]
              
# Stack = StackingClassifier(estimators = estimators, final_estimator = MLPClassifier())
# cross_val_ST_Acc = cross_val_score(Stack, X_train, y_train, cv = CV, scoring = 'accuracy') 
# cross_val_ST_f1 = cross_val_score(Stack, X_train, y_train, cv = CV, scoring = 'f1')
# cross_val_ST_AUC = cross_val_score(Stack, X_train, y_train, cv = CV, scoring = 'roc_auc')


st.subheader('Model Comparison')

    
LR_S = LogisticRegression(random_state = 42)    
params_LR = {'C': list(np.arange(1,12)), 'penalty': ['l2', 'elasticnet', 'none'], 'class_weight': ['balanced','None']}
grid_LR = RandomizedSearchCV(LR_S, param_distributions=params_LR, cv=5, n_jobs=-1, n_iter=20, random_state=42, return_train_score=True)
grid_LR.fit(X_train, y_train)
LR = LogisticRegression(random_state = 42, penalty= 'l2', class_weight= 'balanced', C=6)
cross_val_LR_Acc = cross_val_score(LR, X_train, y_train, cv = CV, scoring = 'accuracy') 
cross_val_LR_f1 = cross_val_score(LR, X_train, y_train, cv = CV, scoring = 'f1')
cross_val_LR_AUC = cross_val_score(LR, X_train, y_train, cv = CV, scoring = 'roc_auc')
  

    
RF_S = RandomForestClassifier(random_state = 42)
params_RF = {'n_estimators': list(range(50,100)), 'min_samples_leaf': list(range(1,5)), 'min_samples_split': list(range(1,5))}
grid_RF = RandomizedSearchCV(RF_S, param_distributions=params_RF, cv=5, n_jobs=-1, n_iter=20, random_state=42, return_train_score=True)
grid_RF.fit(X_train, y_train)

RF = RandomForestClassifier(n_estimators=70, random_state=42)
cross_val_RF_Acc = cross_val_score(RF, X_train, y_train, cv = CV, scoring = 'accuracy') 
cross_val_RF_f1 = cross_val_score(RF, X_train, y_train, cv = CV, scoring = 'f1')
cross_val_RF_AUC = cross_val_score(RF, X_train, y_train, cv = CV, scoring = 'roc_auc')

    # KNN

KNN_S = KNeighborsClassifier()
params_KNN = {'n_neighbors': list(range(1,20))}
grid_KNN = RandomizedSearchCV(KNN_S, param_distributions=params_KNN, cv=5, n_jobs=-1, n_iter=20, random_state=42, return_train_score=True)
grid_KNN.fit(X_train, y_train)
KNN = KNeighborsClassifier(n_neighbors=1)
cross_val_KNN_Acc = cross_val_score(KNN, X_train, y_train, cv = CV, scoring = 'accuracy') 
cross_val_KNN_f1 = cross_val_score(KNN, X_train, y_train, cv = CV, scoring = 'f1')
cross_val_KNN_AUC = cross_val_score(KNN, X_train, y_train, cv = CV, scoring = 'roc_auc')

    
    #
    
DT_S = DecisionTreeClassifier(random_state=42)
params_DT = {'min_samples_leaf': list(range(1,6)), 'min_samples_split': list(range(1,6))}
grid_DT = RandomizedSearchCV(DT_S, param_distributions=params_DT, cv=5, n_jobs=-1, n_iter=20, random_state=42, return_train_score=True)
grid_DT.fit(X_train, y_train)
DT = DecisionTreeClassifier(random_state=42)
cross_val_DT_Acc = cross_val_score(DT, X_train, y_train, cv = CV, scoring = 'accuracy') 
cross_val_DT_f1 = cross_val_score(DT, X_train, y_train, cv = CV, scoring = 'f1')
cross_val_DT_AUC = cross_val_score(DT, X_train, y_train, cv = CV, scoring = 'roc_auc')

compare_models = [('Logistic Regression', cross_val_LR_Acc.mean(),cross_val_LR_f1.mean(),cross_val_LR_AUC.mean(), ''),
                  ('Random Forest', cross_val_RF_Acc.mean(),cross_val_RF_f1.mean(),cross_val_RF_AUC.mean(), ''),
                  ('KNN', cross_val_KNN_Acc.mean(),cross_val_KNN_f1.mean(),cross_val_KNN_AUC.mean(), ''),
                  ('Decision Tree', cross_val_DT_Acc.mean(), cross_val_DT_f1.mean(),cross_val_DT_AUC.mean(), '')]


compare = pd.DataFrame(data = compare_models, columns=['Model','Accuracy Mean', 'F1 Score Mean', 'AUC Score Mean', 'Description'])
compare.style.background_gradient(cmap='YlGn')

d1 = {'Logistic Regression':cross_val_LR_Acc, 'Random Forest':cross_val_RF_Acc, 'KNN':cross_val_KNN_Acc, 'Decision Tree':cross_val_DT_Acc}
d_accuracy = pd.DataFrame(data = d1)

d2 = {'Logistic Regression':cross_val_LR_f1, 'Random Forest':cross_val_RF_f1, 'KNN':cross_val_KNN_f1, 'Decision Tree':cross_val_DT_f1}
d_f1 = pd.DataFrame(data = d2)

d3 = {'Logistic Regression':cross_val_LR_AUC, 'Random Forest':cross_val_RF_AUC, 'KNN':cross_val_KNN_AUC, 'Decision Tree':cross_val_DT_AUC}
d_auc = pd.DataFrame(data = d3)


fig = go.Figure()
fig.add_trace(go.Box(name='Logistic Regression', y=d_accuracy.iloc[:,0]))
fig.add_trace(go.Box(name='Random Forest', y=d_accuracy.iloc[:,1]))
fig.add_trace(go.Box(name='KNN', y=d_accuracy.iloc[:,2]))
fig.add_trace(go.Box(name='Decision Tree', y=d_accuracy.iloc[:,3]))
fig.update_traces(boxpoints='all', boxmean=True)

fig.update_layout(title_text='<b>Box Plots for Models Accuracy <b>')
fig

fig = go.Figure()
fig.add_trace(go.Box(name='Logistic Regression', y=d_f1.iloc[:,0]))
fig.add_trace(go.Box(name='Random Forest', y=d_f1.iloc[:,1]))
fig.add_trace(go.Box(name='KNN', y=d_f1.iloc[:,2]))
fig.add_trace(go.Box(name='Decision Tree', y=d_f1.iloc[:,3]))

fig.update_traces(boxpoints='all', boxmean=True)

fig.update_layout(title_text='<b>Box Plots for Models F1 Score <b>')
fig

fig = go.Figure()
fig.add_trace(go.Box(name='Logistic Regression', y=d_auc.iloc[:,0]))
fig.add_trace(go.Box(name='Random Forest', y=d_auc.iloc[:,1]))
fig.add_trace(go.Box(name='KNN', y=d_auc.iloc[:,2]))
fig.add_trace(go.Box(name='Decision Tree', y=d_auc.iloc[:,3]))


fig.update_traces(boxpoints='all', boxmean=True)

fig.update_layout(title_text='<b>Box Plots for Models AUC <b>')
fig

# print(classification_report(y_test,y_pred))

# y_prob = Stack.predict_proba(X_test)

# roc_auc_score(y_test, y_prob[:,1],average='macro')

# fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])

# fig = px.area(
#     x=fpr, y=tpr,
#     title=f'<b>ROC Curve (AUC={auc(fpr, tpr):.4f})<b>',
#     labels=dict(x='False Positive Rate', y='True Positive Rate'),
#     width=700, height=500, color_discrete_sequence=['#DA598A'])

# fig.add_shape(
#     type='line', line=dict(dash='dash'),
#     x0=0, x1=1, y0=0, y1=1
# )

# fig.update_yaxes(scaleanchor="x", scaleratio=1)
# fig.update_xaxes(constrain='domain')
# fig


st.subheader('Results')
st.markdown("* Customer churn is definitely bad to a firm ’s profitability")
st.markdown("* Various strategies can be implemented to eliminate customer churn ")
st.markdown("* The best way to avoid customer churn is for a company to truly know its customers")
st.markdown("* This includes identifying customers who are at risk of churning and working to improve their satisfaction")
st.markdown("* Improving customer service is, of course, at the top of the priority for tackling this issue")
st.markdown("* Building customer loyalty through relevant experiences and specialized service is another strategy to reduce customer churn")
st.markdown("* Some firms survey customers who have already churned to understand their reasons for leaving in order to adopt a proactive approach to avoiding future customer churn")