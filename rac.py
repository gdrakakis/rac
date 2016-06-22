#!flask/bin/python

from __future__ import division
from flask import Flask, jsonify, abort, request, make_response, url_for
import json
import pickle
import base64
import numpy
import math
import scipy
from copy import deepcopy
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn import linear_model
from numpy  import array, shape, where, in1d
import ast
import threading
import Queue
import time
import random
from random import randrange
import sklearn
from sklearn import cross_validation
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
import cStringIO
from numpy import random
import scipy
from scipy.stats import chisquare
from copy import deepcopy
import operator 
import matplotlib
import io
from io import BytesIO
#matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
#from PIL import Image ## Hide for production

app = Flask(__name__, static_url_path = "")

"""
    JSON Parser for Read Across Training
"""
def getJsonTrainRA (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        predictionFeature = jsonInput["predictionFeature"]
        parameters = jsonInput["parameters"]

        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)

        variables = dataEntry[0]["values"].keys() 
        variables.sort() 
        datapoints =[]
        target_variable_values = []
        substances = [] ## new 21/06/16
        for i in range(len(dataEntry)):
            datapoints.append([])

        for i in range(len(dataEntry)):
            substances.append(dataEntry[i]["compound"]["URI"]) ## new 21/06/16
            for j in variables:
                if j == predictionFeature:
                    target_variable_values.append(dataEntry[i]["values"].get(j))
                else:
                    datapoints[i].append(dataEntry[i]["values"].get(j))

        variables.remove(predictionFeature)

    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"

    return variables, datapoints, predictionFeature, target_variable_values, parameters, substances ## new 21/06/16


def getJsonTestRA (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        rawModel = jsonInput["rawModel"]
        additionalInfo = jsonInput["additionalInfo"]

        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)

        predictionFeature = additionalInfo[0].get("predictedFeature", None)

        variables = dataEntry[0]["values"].keys() 
        variables.sort() 

        readAcrossURIs =[] # new 21/06/16
        datapoints =[]
        for i in range(len(dataEntry)):
            datapoints.append([])

        for i in range(len(dataEntry)):
            readAcrossURIs.append(dataEntry[i]["compound"]["URI"]) # new 21/06/16
            for j in variables:
                #datapoints[i].append(dataEntry[i]["values"].get(j)) ## previous
                if j + " predicted" != predictionFeature:
                    datapoints[i].append(dataEntry[i]["values"].get(j)) ## hack # new 21/06/16

    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"

    return variables, datapoints, predictionFeature, rawModel, readAcrossURIs # new 21/06/16

"""
def getJsonContentsRA (jsonInput):
    try:
        dataset = jsonInput["dataset"]
        predictionFeature = jsonInput["predictionFeature"]
        parameters = jsonInput["parameters"]

        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)
        readAcrossURIs = parameters.get("readAcrossURIs", None) # nanoparticles for readAcross

        variables = dataEntry[0]["values"].keys() 
        variables.sort()  # NP features including predictionFeature

        datapoints =[] # list of nanoparticle feature vectors not for readacross
        read_across_datapoints = [] #list of readacross nanoparticle feature vectors

        nanoparticles=[] # nanoparticles not in readAcrossURIs list
        target_variable_values = [] # predictionFeature values

        for i in range(len(dataEntry)-len(readAcrossURIs)):
            datapoints.append([])

        for i in range(len(readAcrossURIs)):
            read_across_datapoints.append([])


        counter = 0
        RAcounter = 0
        for i in range(len(dataEntry)):

            if dataEntry[i]["compound"].get("URI") not in readAcrossURIs:
                nanoparticles.append(dataEntry[i]["compound"].get("URI"))
                for j in variables:
                    if j == predictionFeature:
                        target_variable_values.append(dataEntry[i]["values"].get(j))
                    else:
                        datapoints[counter].append(dataEntry[i]["values"].get(j))
                counter+=1
            else:
                for j in variables:
                    if j != predictionFeature:
                        read_across_datapoints[RAcounter].append(dataEntry[i]["values"].get(j))
                RAcounter+=1

        variables.remove(predictionFeature) # NP features

    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"
    #print len(nanoparticles), len(read_across_datapoints)
    #print readAcrossURIs, read_across_datapoints
    return variables, datapoints, read_across_datapoints, predictionFeature, target_variable_values, byteify(readAcrossURIs), nanoparticles
"""

"""
    Byte-ify or utf-8 
"""
def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

"""
    [[],[]]  Matrix to dictionary for Nearest Neighboura
"""
def mat2dicNN(matrix, name):
    myDict = {}
    for i in range (len (matrix[0])):
        myDict[name + " NN_" + str(i+1)] = [matrix[0][i], matrix[1][i]]
    return byteify(myDict)

"""
    [[],[]]  Matrix to dictionary 
"""
def mat2dic(matrix):
    myDict = {}
    for i in range (len (matrix[0])):
        myDict["Row_" + str(i+1)] = [matrix[0][i], matrix[1][i]]
    return byteify(myDict)

"""
    [[]]  Matrix to dictionary Single Row
"""
def mat2dicSingle(matrix):
    myDict = {}
    myDict["Row_1"] = matrix
    return byteify(myDict)

"""
    Normaliser
"""
def manual_norm(myTable, myMax, myMin):
    if myMax>myMin:
        for i in range (len(myTable)):
            myTable[i] = (myTable[i]-myMin)/(myMax-myMin)
    else:
        for i in range (len(myTable)):
            myTable[i] = 0
    return myTable

"""
    Distances
"""
def distances (read_across_datapoints, datapoints, variables, readAcrossURIs, nanoparticles):

    datapoints_transposed = map(list, zip(*datapoints)) 
    RA_datapoints_transposed = map(list, zip(*read_across_datapoints)) 
    
    for i in range (len(datapoints_transposed)):
        max4norm = numpy.max(datapoints_transposed[i])
        min4norm = numpy.min(datapoints_transposed[i])

        datapoints_transposed[i] = manual_norm(datapoints_transposed[i], max4norm, min4norm)
        RA_datapoints_transposed[i] = manual_norm(RA_datapoints_transposed[i], max4norm, min4norm)

    #print RA_datapoints_transposed[0]
    #print datapoints_transposed[0]

    term1 = []
    term2 = []
    for i in range (len(variables)):
        #term1.append(numpy.min(datapoints_transposed))
        #term2.append(numpy.max(datapoints_transposed))
        term1.append(0)
        term2.append(1)

    #transpose back
    datapoints_norm = map(list, zip(*datapoints_transposed)) 
    RA_datapoints_norm = map(list, zip(*RA_datapoints_transposed)) 

    #print numpy.max(RA_datapoints_norm)
    #print numpy.max(datapoints_norm)

    #for i in range (len(datapoints)):
    #    datapoints[i] = manual_norm(datapoints[i], max4norm, min4norm)
    #for i in range (len(read_across_datapoints)):
    #    read_across_datapoints[i] = manual_norm(read_across_datapoints[i], max4norm, min4norm)

    #print len(RA_datapoints_norm), len(RA_datapoints_norm[0])
    #print len(datapoints_norm), len(datapoints_norm[0])
    #"""
    max_eucl_dist = euclidean_distances(term1, term2)
    eucl_dist = euclidean_distances(RA_datapoints_norm, datapoints_norm)
    eucl_dist = numpy.array(eucl_dist)
    eucl_dist = eucl_dist/max_eucl_dist
    eucl_dist = numpy.round(eucl_dist,4)

    np_plus_eucl = []
    for i in range (len(readAcrossURIs)):
        np_plus_eucl.append([nanoparticles, eucl_dist[i]]) 
    #print np_plus_eucl

    eucl_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_eucl[i][0], np_plus_eucl[i][1]
        np = zip (np_plus_eucl[i][1], np_plus_eucl[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        eucl_sorted.append([np_sorted, dist_sorted])
    #print "\n\nSorted\n\n", eucl_sorted
    ## [ [ [names] [scores] ] [ [N] [S] ]]
    ##       00      01          10  11    


    #eucl_transposed = map(list, zip(*eucl_sorted)) 
    eucl_dict = {} # []
    for i in range (len(readAcrossURIs)):
        #print "\n HERE \n ", eucl_sorted[i]
        #eucl_dict.append(mat2dicNN(eucl_sorted[i], readAcrossURIs[i])) #
        for j in range (len (eucl_sorted[i][0])):
            eucl_dict[readAcrossURIs[i] + " NN_" + str(j+1)] = [eucl_sorted[i][0][j], eucl_sorted[i][1][j]]
    eucl_dict = byteify(eucl_dict)
    #print "\n\nDict\n\n",eucl_dict

    max_manh_dist = metrics.pairwise.manhattan_distances(term1, term2)
    manh_dist = metrics.pairwise.manhattan_distances(RA_datapoints_norm, datapoints_norm)
    manh_dist = numpy.array(manh_dist)
    manh_dist = manh_dist/max_manh_dist
    manh_dist = numpy.round(manh_dist,4)

    np_plus_manh = []
    for i in range (len(readAcrossURIs)):
        np_plus_manh.append([nanoparticles, manh_dist[i]]) 

    manh_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_manh[i][0], np_plus_manh[i][1]
        np = zip (np_plus_manh[i][1], np_plus_manh[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        manh_sorted.append([np_sorted, dist_sorted])
    #print manh_sorted

    manh_dict = {}
    for i in range (len(readAcrossURIs)):
        #manh_dict.append(mat2dicNN(manh_sorted[i], readAcrossURIs[i]))
        for j in range (len (manh_sorted[i][0])):
            manh_dict[readAcrossURIs[i] + " NN_" + str(j+1)] = [manh_sorted[i][0][j], manh_sorted[i][1][j]]
    manh_dict = byteify(manh_dict)

    ensemble_dist = (eucl_dist + manh_dist)/2
    #print "Eucl.: ", eucl_dist, "\n Manh.: ", manh_dist,"\n Ens.: ", ensemble_dist

    np_plus_ens = []
    for i in range (len(readAcrossURIs)):
        np_plus_ens.append([nanoparticles, ensemble_dist[i]]) 

    ens_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_ens[i][0], np_plus_ens[i][1]
        np = zip (np_plus_ens[i][1], np_plus_ens[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        ens_sorted.append([np_sorted, dist_sorted])
    #print ens_sorted

    ens_dict = {}
    for i in range (len(readAcrossURIs)):
        #ens_dict.append(mat2dicNN(ens_sorted[i], readAcrossURIs[i]))
        for j in range (len (ens_sorted[i][0])):
            ens_dict[readAcrossURIs[i] + " NN_" + str(j+1)] = [ens_sorted[i][0][j], ens_sorted[i][1][j]]
    ens_dict = byteify(ens_dict)

    """
    ### PLOT PCA
    pcafig = plt.figure()
    ax = pcafig.add_subplot(111, projection='3d')

    pca = decomposition.PCA(n_components=3)
    pca.fit(datapoints_norm)
    dt = pca.transform(datapoints_norm)
    ax.scatter(dt[:,0], dt[:,1], dt[:,2], c='r',  label = 'Original Values')

    RA_dt = pca.transform(RA_datapoints_norm)
    ax.scatter(RA_dt[:,0], RA_dt[:,1], RA_dt[:,2], c='b', label = 'Read Across Values')

    ax.set_xlabel("1st Principal Component") 
    ax.set_ylabel("2nd Principal Component")
    ax.set_zlabel("3rd Principal Component")
    ax.set_title("3D Projection of Datapoints")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    #plt.tight_layout()
    plt.show() #HIDE show on production

    figfile = BytesIO()
    pcafig.savefig(figfile, dpi=300, format='png', bbox_inches='tight') #bbox_inches='tight'
    figfile.seek(0) 
    pcafig_encoded = base64.b64encode(figfile.getvalue())    
    """
    #return 0,0,0,0,0,0,0 ## , ""
    return eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict # , pcafig_encoded# new 21/06/16

"""
    Prediction function
"""
def RA_predict(euclidean, manhattan, ensemble, name, predictionFeature, nano2value):
    eu_score = 0
    ma_score = 0
    en_score = 0

    eu_div = 0
    ma_div = 0
    en_div = 0

    for i in range (len(euclidean[0])):
        if euclidean[1][i] < 1:
            eu_score += (1 - euclidean[1][i])*(nano2value[euclidean[0][i]]) #just the name
            eu_div += 1 - euclidean[1][i]
        if manhattan[1][i] < 1:
            ma_score += (1 - manhattan[1][i])*(nano2value[euclidean[0][i]]) #just the name
            ma_div += 1 - manhattan[1][i]
        if ensemble[1][i] < 1:
            en_score += (1 - ensemble[1][i])*(nano2value[euclidean[0][i]]) #just the name
            en_div += 1 - ensemble[1][i]
    eu_score = eu_score/eu_div
    ma_score = ma_score/ma_div
    en_score = en_score/en_div
    #print eu_score
    return [name, round(eu_score,2)], [name, round(ma_score,2)], [name, round(en_score,2)]

"""
    Pseudo AD
"""
def RA_applicability(euclidean, manhattan, ensemble, name):
    eu_score = 0
    ma_score = 0
    en_score = 0
    for i in range (len(euclidean[1])): # list of vals
        if euclidean[1][i] < 0.4:
            eu_score +=1
        if manhattan[1][i] < 0.33:
            ma_score +=1
        if ensemble[1][i] < 0.36:
            en_score +=1
    eu_score = eu_score/len(euclidean[1])
    ma_score = ma_score/len(euclidean[1])
    en_score = en_score/len(euclidean[1])

    return [name, eu_score], [name, ma_score], [name, en_score]
    
"""
    tasks
"""
@app.route('/pws/readacross/train', methods = ['POST'])
def create_task_readacross_train():

    if not request.environ['body_copy']:
        abort(500)

    readThis = json.loads(request.environ['body_copy'])

    #return variables, datapoints, predictionFeature, target_variable_values, parameters, substances ## new 21/06/16
    variables, datapoints, predictionFeature, target_variable_values, parameters, substances  = getJsonTrainRA(readThis)
    
    # Parameters: euclidean, manhattan, ensemble (0-1) and confidence (0 or 1)
    euclidean = parameters.get("euclidean", 0.4)
    manhattan = parameters.get("manhattan", 0.33)
    ensemble = parameters.get("ensemble", 0.36)
    confidence = parameters.get("confidence", 1)

    taskDic = {}
    #taskDic["variables"] = variables # sent separately
    taskDic["substances"] = substances
    taskDic["datapoints"] = datapoints
    taskDic["predictionFeature"] = predictionFeature
    taskDic["target_variable_values"] = target_variable_values
    taskDic["euclidean"] = euclidean
    taskDic["manhattan"] = manhattan
    taskDic["ensemble"] = ensemble

    encoded = base64.b64encode(str(taskDic))

    #predictedString = predictionFeature + " predicted" ## removed # temp # 'new' 21/06/16

    # new 21/06/16
    predictedString1 = predictionFeature + " Euclidean"
    predictedString2 = predictionFeature + " Manhattan"
    predictedString3 = predictionFeature + " Ensemble"
    predictedString4 = predictionFeature + " Confidence Euclidean"
    predictedString5 = predictionFeature + " Confidence Manhattan"
    predictedString6 = predictionFeature + " Confidence Ensemble"

    # new 21/06/16 # check -> some is only for internal usage
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictionFeature + " predicted"}], 
        "independentFeatures": variables, 
        "predictedFeatures": [
            predictedString1, predictedString2, predictedString3, predictedString4, predictedString5, predictedString6  
            ] 
        }

    #xxx = open("C:/Python27/RA_train_Delete.txt", "w")
    #xxx.writelines(str(task))
    #xxx.close 
    #task = {}
    jsonOutput = jsonify( task )
    
    return jsonOutput, 201 

"""
    prediction
"""
@app.route('/pws/readacross/test', methods = ['POST'])
def create_task_readacross_test():

    if not request.environ['body_copy']:
        abort(500)

    readThis = json.loads(request.environ['body_copy'])


    #return variables, datapoints, predictionFeature, rawModel, readAcrossURIs # new 21/06/16
    variables, read_across_datapoints, predictionFeature, rawModel, readAcrossURIs  = getJsonTestRA(readThis)

    #print len(readAcrossURIs), len(read_across_datapoints), len(read_across_datapoints[0])
    got_raw = base64.b64decode(rawModel) ## check
    decoded = ast.literal_eval(got_raw) ## check

    datapoints = decoded["datapoints"] 
    substances = decoded["substances"] 
    predictionFeature = decoded["predictionFeature"] 
    target_variable_values = decoded["target_variable_values"]  
    euclidean = decoded["euclidean"] 
    manhattan = decoded["manhattan"] 
    ensemble = decoded["ensemble"] 

    #print len(substances), len(datapoints), len(datapoints[0])
    #print predictionFeature, len (target_variable_values)
    #print euclidean, manhattan, ensemble


    nano2value = {}
    for i in range (len(substances)):
        nano2value[substances[i]] = target_variable_values[i]
    # included IDs # new 21/06/16
    # removed PCA  # new 21/06/16
    #eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict, pcafig_encoded = distances (read_across_datapoints, datapoints, variables, readAcrossURIs, substances)
    eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict = distances (read_across_datapoints, datapoints, variables, readAcrossURIs, substances)
    
    #print len (eucl_sorted), len(manh_sorted), len(ens_sorted)
    #"""
    eucl_predictions = []
    manh_predictions = [] 
    ens_predictions = []

    eucl_applicability = []
    manh_applicability = [] 
    ens_applicability = []
    for i in range (len(readAcrossURIs)):
        eu,ma,en = RA_predict(eucl_sorted[i], manh_sorted[i], ens_sorted[i], readAcrossURIs[i], predictionFeature, nano2value)
        eucl_predictions.append(eu)
        manh_predictions.append(ma)
        ens_predictions.append(en)
        
        eu,ma,en = RA_applicability(eucl_sorted[i], manh_sorted[i], ens_sorted[i], readAcrossURIs[i])
        eucl_applicability.append(eu)
        manh_applicability.append(ma)
        ens_applicability.append(en)
    #print ens_predictions # works 21/06/16
    #"""
    if len (eucl_predictions) > 1:
        # predictions
        eucl_predictions_transposed = map(list, zip(*eucl_predictions)) 
        #print "\n\n\n", eucl_predictions,"\n\n\n"
        #print "\n\n\n", eucl_predictions_transposed,"\n\n\n"
        eucl_pred_dict = mat2dic(eucl_predictions_transposed) # Checked: Changed mat2dic # new 21/06/16
        #eucl_pred_dict = mat2dic(eucl_predictions) # curr

        manh_predictions_transposed = map(list, zip(*manh_predictions)) 
        manh_pred_dict = mat2dic(manh_predictions_transposed)
        #print manh_pred_dict

        ens_predictions_transposed = map(list, zip(*ens_predictions)) 
        ens_pred_dict = mat2dic(ens_predictions_transposed)
        #print ens_pred_dict

        # applicability
        eucl_applicability_transposed = map(list, zip(*eucl_applicability)) 
        eucl_appl_dict = mat2dic(eucl_applicability_transposed)
        #print eucl_appl_dict

        manh_applicability_transposed = map(list, zip(*manh_applicability)) 
        manh_appl_dict = mat2dic(manh_applicability_transposed)
        #print manh_appl_dict

        ens_applicability_transposed = map(list, zip(*ens_applicability)) 
        ens_appl_dict = mat2dic(ens_applicability_transposed)
        #print ens_appl_dict
    else: 
        eucl_pred_dict = mat2dicSingle(eucl_predictions[0])
        manh_pred_dict = mat2dicSingle(manh_predictions[0])
        ens_pred_dict = mat2dicSingle(ens_predictions[0])
        eucl_appl_dict = mat2dicSingle(eucl_applicability[0])
        manh_appl_dict = mat2dicSingle(manh_applicability[0])
        ens_appl_dict = mat2dicSingle(ens_applicability[0])
    #print ens_pred_dict
    #print ens_appl_dict

    eu_p = []
    ma_p = []
    en_p = []
    eu_a = []
    ma_a = []
    en_a = []

    for i in range (len(eucl_predictions)):
        eu_p.append(eucl_predictions[i][1])
        ma_p.append(manh_predictions[i][1])
        en_p.append(ens_predictions[i][1])
        eu_a.append(eucl_applicability[i][1])
        ma_a.append(manh_applicability[i][1])
        en_a.append(ens_applicability[i][1])

    #"""
    task = {
        "predictionsEuclidean": eu_p,
        "predictionsManhattan": ma_p,
        "predictionsEnsemble": en_p,
        "confidenceEuclidean": eu_a,
        "confidenceManhattan": ma_a,
        "confidenceEnsemble": en_a
        }

    #xxx = open("C:/Python27/RApredict_delete.txt", "w")
    #xxx.writelines(str(task))
    #xxx.close 
    #task = {}
    jsonOutput = jsonify( task )
    
    return jsonOutput, 201 

############################################################
############################################################

class WSGICopyBody(object):
    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        from cStringIO import StringIO
        input = environ.get('wsgi.input')
        length = environ.get('CONTENT_LENGTH', '0')
        length = 0 if length == '' else int(length)
        body = ''
        if length == 0:
            environ['body_copy'] = ''
            if input is None:
                return
            if environ.get('HTTP_TRANSFER_ENCODING','0') == 'chunked':
                size = int(input.readline(),16)
                while size > 0:
                    temp = str(input.read(size+2)).strip()
                    body += temp
                    size = int(input.readline(),16)
        else:
            body = environ['wsgi.input'].read(length)
        environ['body_copy'] = body
        environ['wsgi.input'] = StringIO(body)

        # Call the wrapped application
        app_iter = self.application(environ, 
                                    self._sr_callback(start_response))

        # Return modified response
        #print app_iter
        return app_iter

    def _sr_callback(self, start_response):
        def callback(status, headers, exc_info=None):

            # Call upstream start_response
            start_response(status, headers, exc_info)
        #print callback
        return callback

############################################################

if __name__ == '__main__': 
    app.wsgi_app = WSGICopyBody(app.wsgi_app) ##
    app.run(host="0.0.0.0", port = 5000, debug = True)	
# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/ractrain.json http://localhost:5000/pws/readacross/train
# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/ractest.json http://localhost:5000/pws/readacross/test
# C:\Python27\Flask-0.10.1\python-api 
# C:/Python27/python rac.py