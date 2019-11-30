import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import json
import plotly
from plotly import graph_objs as go
from xgboost import XGBClassifier

app = Flask(__name__)
clf = pickle.load(open('xgB.pickle.dat', 'rb'))
lEncoder = pickle.load(open('labelEncoder.pkl', 'rb'))
jsEncoder = pickle.load(open('jamesStein.pkl', 'rb'))
sampleCsv = pickle.load(open('sample.pkl', 'rb'))
df = pickle.load(open('plot.pkl','rb'))
givenData = pickle.load(open('geoPlot.pkl','rb'))

cityList=list(givenData['Customer City'].unique())
stateList=list(givenData['State'].unique())
reviewZoneList=list(givenData['Review Zone'].unique())
reTerritoryList=list(givenData['Re Territory'].unique())

#Final Function for hosting

def geoFunction(geo, productFeature):
    #Analysis for Categorical Product Features:
    if geo in cityList:
        tempData=givenData[givenData['Customer City']==geo]
    elif geo in stateList:
        tempData=givenData[givenData['State']==geo]
    elif geo in reviewZoneList:
        tempData=givenData[givenData['Review Zone']==geo]
    elif geo in reTerritoryList:
        tempData=givenData[givenData['Re Territory']==geo]


    try:
        tempMode=statistics.mode(tempData[productFeature])
        print(f'The mode statistic for {productFeature}, in {geo}: {tempMode}')
    except:
        print(f'The following values are equally distributed, so no mode found for {productFeature} in {geo} :')
        values=tempData[productFeature]

    
    tileStyles=tempData[productFeature].unique()
    months=givenData['Month'].unique()
    data=[]
    for style in tileStyles:
        tempName=style
        x=months
        y=[]
        for month in months:
            tempData1=tempData[tempData['Month']==month]
            tempData2=tempData1[tempData1[productFeature]==style]
            tempCount=tempData2.shape[0]
            y.append(tempCount)
        data.append(go.Bar(name=tempName,
                        x=x,
                        y=y
                        ))
        
    # fig=go.Figure(data=data)
    # fig.update_layout(barmode='group', hoverlabel_align='right', title=f'Monthly Distribution of sales for {productFeature} in {geo}')
    # fig.show()
    graphJSON1 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    # tempData=givenData[givenData['Customer City']==city]
    print(f'----Distribution of {productFeature} for {geo}----')
    # plt.figure()
    # sns.countplot(tempData[productFeature]).set_title(f'Total Sale Count Distribution, across different {productFeature}, for {geo}')
    # plt.xticks(rotation=90)
    # plt.figure()
    # sns.countplot(x='Month', hue=productFeature, data=tempData)


    data = [
        go.Bar(
            y=tempData[productFeature].value_counts().to_dense().keys(),
            x=tempData[productFeature].value_counts(),
            orientation='h',
            text="d",
        )]
    layout = go.Layout(
        height=500,
        title=f'Total Sale Distribution for different {productFeature}, for {geo}',
        hovermode='closest',
        xaxis=dict(title=f'Counts', ticklen=5, zeroline=False, gridwidth=2, domain=[0.1, 1]),
        yaxis=dict(title=f'{productFeature}', ticklen=5, gridwidth=2),
        showlegend=False
    )

    graphJSON2 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON1,graphJSON2 
    # fig = go.Figure(data=data, layout=layout)
    # py.iplot(fig, filename='Sector/ Area of Coaches - Combined')

    # fig.show()

df_binned = df.copy()
numeric_columns = ['AD1/Sqm','AD2/Sqm','AD3/Sqm','AD4/Sqm','AD5/Sqm','AD6/Sqm','Sq. Mt.','Total AD/Sqm','Billing Rate/Sqm','Buyer Rate/Sqm','MRP /BOX','MRP /Sqm','Value','Price List Ex.',]
for column in numeric_columns:
  maxi = df[column].max()
  mini = df[column].min()
  rangei = (maxi-mini)/5
  bins = [mini+rangei,mini+rangei*2,mini+rangei*3,mini+rangei*4,maxi]
  labels = [str(round(mini,2))+'-'+str(round(bins[0],2)), str(round(bins[0],2))+'-'+str(round(bins[1],2)), str(round(bins[1],2))+'-'+str(round(bins[2],2)), str(round(bins[2],2))+'-'+str(round(bins[3],2)), str(round(bins[3],2))+'-'+str(round(bins[4],2))]
  def binned(x):
    if(x<=bins[0]): return labels[0]
    elif(x<=bins[1]): return labels[1]
    elif(x<=bins[2]): return labels[2]
    elif(x<=bins[3]): return labels[3]
    else: return labels[4]
  df_binned[column] = df[column].apply(binned)

def sunburst_customer(customer_code,feature_list):
  train = df_binned[df_binned['Customer Code']==customer_code]
  no_of_data = train.shape[0]
  print("Customer name:", train['Customer Name & City'].unique())
  print("Number of data rows for customer code:",no_of_data)
  
  def value(i,train,ida,label):
    parents.append(ida)
    if(ida!=labels[0]): ids.append(ida+" - "+label)
    else: ids.append(label)
    labels.append(label)
    values.append(train.shape[0])
    if i==len(feature_list)-1: return
    for k in train[feature_list[i+1]].unique():
      if (ida!=labels[0]):  value(i+1,train[train[feature_list[i+1]]==k],ida+" - "+label,k)
      else: value(i+1,train[train[feature_list[i+1]]==k],label,k)
    return 

  labels,ids,parents,values = ['Customer Code: '+customer_code],['Customer Code: '+customer_code],[""],[no_of_data]
  for j in train[feature_list[0]].unique():
    value(0,train[train[feature_list[0]]==j],labels[0],j)
  
  data = [go.Sunburst(ids=ids,labels=labels,parents=parents,values=values,branchvalues="total",)]
  graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON

def sunburst_state(state,feature_list):
  if(state in stateList):
    train = df_binned[df_binned['State']==state]
    no_of_data = train.shape[0]
    labels,ids,parents,values = ['State: '+state],['State: '+state],[""],[no_of_data]
  elif(state in cityList):
    train = df_binned[df_binned['Customer City']==state]
    no_of_data = train.shape[0]
    labels,ids,parents,values = ['City: '+state],['City: '+state],[""],[no_of_data]
  elif(state in reviewZoneList):
    train = df_binned[df_binned['Review Zone']==state]
    no_of_data = train.shape[0]
    labels,ids,parents,values = ['Review Zone: '+state],['City: '+state],[""],[no_of_data]
  elif(state in reTerritoryList):
    train = df_binned[df_binned['Re Territory']==state]
    no_of_data = train.shape[0]

  
  print("Number of data rows for this Geography:",no_of_data)
  
  def value(i,train,ida,label):
    parents.append(ida)
    if(ida!=labels[0]): ids.append(ida+" - "+label)
    else: ids.append(label)
    labels.append(label)
    values.append(train.shape[0])
    if i==len(feature_list)-1: return
    for k in train[feature_list[i+1]].unique():
      if (ida!=labels[0]):  value(i+1,train[train[feature_list[i+1]]==k],ida+" - "+label,k)
      else: value(i+1,train[train[feature_list[i+1]]==k],label,k)
    return 

  
  for j in train[feature_list[0]].unique():
    value(0,train[train[feature_list[0]]==j],labels[0],j)

  data = [go.Sunburst(ids=ids,labels=labels,parents=parents,values=values,branchvalues="total",)]
  graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON 

# def sunburst_city(city,feature_list):
#   train = df_binned[df_binned['Customer City']==city]
#   no_of_data = train.shape[0]
#   print("Number of data rows for this city:",no_of_data)
  
#   def value(i,train,ida,label):
#     parents.append(ida)
#     if(ida!=labels[0]): ids.append(ida+" - "+label)
#     else: ids.append(label)
#     labels.append(label)
#     values.append(train.shape[0])
#     if i==len(feature_list)-1: return
#     for k in train[feature_list[i+1]].unique():
#       if (ida!=labels[0]):  value(i+1,train[train[feature_list[i+1]]==k],ida+" - "+label,k)
#       else: value(i+1,train[train[feature_list[i+1]]==k],label,k)
#     return 

#   labels,ids,parents,values = ['City: '+city],['City: '+city],[""],[no_of_data]
#   for j in train[feature_list[0]].unique():
#     value(0,train[train[feature_list[0]]==j],labels[0],j)

#   # fig = go.Figure(go.Sunburst(ids=ids,labels=labels,parents=parents,values=values,branchvalues="total",))
#   # fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
#   # fig.show()  
#   data =[go.Sunburst(ids=ids,labels=labels,parents=parents,values=values,branchvalues="total",)]
#   graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
#   return graphJSON

def bargraph(label,freq,feature):
  data = [go.Bar(x=label, y=freq)]
  # fig.update_xaxes(title_text=feature)
  # fig.update_yaxes(title_text='Frequency')
  graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON

def piechart(label,freq,feature):
  data=[go.Pie(labels=label, values=freq, hole=.5, title=feature)]
  graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON

# def item_feature(train,feature):
#   unique = train[feature].unique()
#   if(len(unique)==1): print (unique[0])
#   else:
#     unique.sort()
#     label,freq = [],[]
#     for i in unique:
#       label.append(i)
#       freq.append(train[train[feature]==i].shape[0])
#     return piechart(label,freq,feature)

# def customer_feature(train,feature):
#   unique = train[feature].unique()
#   if(len(unique)==1): print (unique[0])
#   else:
#     unique.sort()
#     label,freq = [],[]
#     for i in unique:
#       label.append(i)
#       freq.append(train[train[feature]==i].shape[0])
#     return bargraph(label,freq,feature)

# def item_name(train,feature):
#   unique = train[feature].unique()
#   if(len(unique)==1): print (unique[0])
#   else:
#     unique.sort()
#     label,freq = [],[]
#     for i in unique:
#       label.append(i)
#       freq.append(train[train[feature]==i].shape[0])
#     return bargraph(label,freq,feature)

# def numeric_feature(train,feature):
#   mean = train[feature].mean()
#   print("Average: {:.2f}".format(mean))
#   print(train[feature].quantile([.1, .25, .5, .75, .9])) 


# def item_preference(customer_code,feature):
#   train = df[df['Customer Code']==customer_code]
#   no_of_data = train.shape[0]
#   #print("Customer name:", train['Customer Name & City'].unique()[0])
#   print("Number of data rows for customer code:",customer_code)
#   item_columns1 = ['Item name']
#   item_columns2 = ['Pcs Prem','ERP Size','Wall / Floor','Tile Body','Item Classification','Item Cat. Code','Category','Category 2','Ship-to City','FY']
#   numeric_columns = ['AD1/Sqm','AD2/Sqm','AD3/Sqm','AD4/Sqm','AD5/Sqm','AD6/Sqm','AD7/Sqm','Sq. Mt.','Total AD/Sqm','Billing Rate/Sqm','Buyer Rate/Sqm','MRP /BOX','MRP /Sqm','Value','Price List Ex.',]
#   customer_columns = ['Review Zone','Re Territory','Customer City','Customer Type','Sales Type','Customer Name & City','State','Date']
#   if feature in item_columns1:  return item_name(train,feature)
#   elif feature in item_columns2:  return item_feature(train,feature)
#   elif feature in numeric_columns:  return numeric_feature(train,feature)
#   elif feature in customer_columns:  return customer_feature(train,feature)
#   else:  print("This feature does not exist")

def feature_analysis(train,feature):
  unique = train[feature].unique()
  if(len(unique)==1): print (unique[0])
  else:
    unique.sort()
    label,freq = [],[]
    for i in unique:
      label.append(i)
      freq.append(train[train[feature]==i].shape[0])
    if(len(label)>20): return bargraph(label,freq,feature)
    else: return piechart(label,freq,feature)

def numeric_feature(train,feature):
  mean = train[feature].mean()
  # print("Average: {:.2f}".format(mean))
  data = px.box(train, y=feature)
  graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON


def customer_item_preference(customer_code,feature):
  train = df[df['Customer Code']==customer_code]
  no_of_data = train.shape[0]
  # print("Customer name:", train['Customer Name & City'].unique())
  # print("Number of data rows for customer code:",no_of_data)
  item_columns = ['Item name','Pcs Prem','ERP Size','Wall / Floor','Tile Type','Color/Design','Tile Body','Item Classification','Item Cat. Code','Category','Category 2','Ship-to City','FY']
  numeric_columns = ['AD1/Sqm','AD2/Sqm','AD3/Sqm','AD4/Sqm','AD5/Sqm','AD6/Sqm','Sq. Mt.','Total AD/Sqm','Billing Rate/Sqm','Buyer Rate/Sqm','MRP /BOX','MRP /Sqm','Value','Price List Ex.',]
  customer_columns = ['Review Zone','Re Territory','Customer City','Customer Type','Sales Type','Customer Name & City','State','Date']
  if feature in item_columns + customer_columns:  return feature_analysis(train,feature)
  elif feature in numeric_columns:  return numeric_feature(train,feature)
  else:  print("This feature does not exist")


topN_Names = []
customerCode = ""
geoSpecific = ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    return render_template('prediction.html')

@app.route('/generateLocGraph',methods=['POST'])
def generateLocGraph():
    feature = str(request.form['feature'])
    graphMonth,graphCount = geoFunction(geoSpecific,feature)

    
    return jsonify({'data': render_template("locPlotter.html",graphMonth=graphMonth, graphCount=graphCount ,feature=str(feature))})

@app.route('/generateGraph',methods=['POST'])
def generateGraph():
    feature = str(request.form['feature'])
    graph = customer_item_preference(customerCode,feature)
    
    return jsonify({'data': render_template("plotter.html",graph=graph, feature=str(feature))})

@app.route('/generateMultiLocGraph',methods=['POST'])
def generateMultiLocGraph():
    featuresList = []
    numFeatures = request.form['numFeatures']
    i = 1
    while(i<=int(numFeatures)):
        feature = str(request.form['feature_'+str(i)])
        featuresList.append(feature)
        i = i+1
    graph = sunburst_state(geoSpecific,featuresList)
    
    #return jsonify({'data': "<p>Sandeep</p>"})
    return jsonify({'data': render_template("sunburstPlotter.html",graph=graph)})


@app.route('/generateMultiGraph',methods=['POST'])
def generateMultiGraph():
    featuresList = []
    numFeatures = request.form['numFeatures']
    i = 1
    while(i<=int(numFeatures)):
        feature = str(request.form['feature_'+str(i)])
        featuresList.append(feature)
        i = i+1
    graph = sunburst_customer(customerCode,featuresList)
    
    #return jsonify({'data': "<p>Sandeep</p>"})
    return jsonify({'data': render_template("sunburstPlotter.html",graph=graph)})

@app.route('/generateTopN',methods=['POST'])
def generateTopN():
    nVal = int(request.form['nValue'])
    
    return jsonify({'data': render_template("rankCards.html", nVal = nVal,topN_Names = topN_Names)})

@app.route('/visualize/customerWise', methods=['POST'])
def customerWise():
    global customerCode
    customerCode = str(request.form.get('custID'))
    train = df[df['Customer Code']==customerCode]
    return render_template( 'customerWise.html',cName=(train['Customer Name & City'].unique()[0]) )

@app.route('/visualize/geographyWise', methods=['POST'])
def geographyWise():
    global geoSpecific
    geoSpecific = str(request.form.get('geoID'))
    return render_template( 'geographyWise.html',cName=(geoSpecific))    

@app.route('/visualize')
def visualHome():
    return render_template('visualize.html')

@app.route('/analytics/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    df = pd.DataFrame(columns=sampleCsv.columns)
    row = []
    for i,x in enumerate(request.form.values()):
        if(i in [3,5,6]):
            row.append(float(x))
        else:
            row.append(x)
            
    df.loc[0] = row        
    
    X_val = jsEncoder.transform(df)

    probas = clf.predict_proba(X_val)
    # return render_template('prediction.html',debug=probas)

    topN = sorted(range(617), key= lambda i: probas[0][i])[-50:]
    global topN_Names 
    topN_Names = lEncoder.inverse_transform(topN)

    return render_template('topN.html', topN_Names=topN_Names)
    
if __name__ == "__main__":
    app.run(debug=True)