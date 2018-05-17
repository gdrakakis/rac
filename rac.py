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
        #print predictedFeatures
        datasetURI = dataset.get("datasetURI", None)
        dataEntry = dataset.get("dataEntry", None)

        # for internal testing
        predictionFeature = "" #jsonInput["predictionFeature"]
        predictedFeatures = additionalInfo[0].get("predictedFeatures", None)

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
                    datapoints[i].append(dataEntry[i]["values"].get(j)) ## FOR INTERNAL USAGE ONLY hack # new 21/06/16

    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"

    return variables, datapoints, predictionFeature, rawModel, readAcrossURIs, predictedFeatures # new 21/06/16


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
        myDict[name + " 's NN_" + str(i+1)] = [matrix[0][i], matrix[1][i]]
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
def distances (read_across_datapoints, datapoints, variables, readAcrossURIs, nanoparticles, threshold):

    datapoints_transposed = map(list, zip(*datapoints)) 
    RA_datapoints_transposed = map(list, zip(*read_across_datapoints)) 

    #print "\n\n\n", RA_datapoints_transposed, "\n\n\n"
    #print "\n\n\n WORKS FINE UNTIL HERE 1\n\n\n"
    for i in range (len(datapoints_transposed)):
        max4norm = numpy.max(datapoints_transposed[i])
        min4norm = numpy.min(datapoints_transposed[i])

        datapoints_transposed[i] = manual_norm(datapoints_transposed[i], max4norm, min4norm)
        RA_datapoints_transposed[i] = manual_norm(RA_datapoints_transposed[i], max4norm, min4norm)

    #print RA_datapoints_transposed[0]
    #print datapoints_transposed[0]
    #print "\n\n\n WORKS FINE UNTIL HERE2\n\n\n"
    term1 = []
    term2 = []
    for i in range (len(variables)):
        #term1.append(numpy.min(datapoints_transposed))
        #term2.append(numpy.max(datapoints_transposed))
        term1.append(0)
        term2.append(1)
    #print "\n\n\n WORKS FINE UNTIL HERE3\n\n\n"
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

    max_eucl_dist = euclidean_distances([term1], [term2]) ## []
    #max_eucl_dist = euclidean_distances(term1, term2) ## []
    #print "\n\n\n", max_eucl_dist,"\n\n\n"
    eucl_dist = euclidean_distances(RA_datapoints_norm, datapoints_norm) ##
    #print "\n\n\n", eucl_dist,"\n\n\n"
    eucl_dist = numpy.array(eucl_dist)
    eucl_dist = eucl_dist/max_eucl_dist
    eucl_dist = numpy.round(eucl_dist,4)
    
    np_plus_eucl = []
    for i in range (len(readAcrossURIs)):
        np_plus_eucl.append([nanoparticles, eucl_dist[i]]) 
    #print "YOLO\n\n\n", np_plus_eucl, "\n\n\n"

    eucl_sorted = []
    for i in range (len(readAcrossURIs)):
        #np_plus_eucl[i][0], np_plus_eucl[i][1]
        np = zip (np_plus_eucl[i][1], np_plus_eucl[i][0])
        np.sort()
        np_sorted = [n for d,n in np] # np, dist
        dist_sorted = [round(d,4) for d,n in np]
        eucl_sorted.append([np_sorted, dist_sorted])
    #print "\n\n\n WORKS FINE UNTIL HERE5\n\n\n"
    #print "\n\nSorted\n\n", eucl_sorted
    ## [ [ [names] [scores] ] [ [N] [S] ]]
    ##       00      01          10  11    


    #eucl_transposed = map(list, zip(*eucl_sorted)) 
    eucl_dict = {} # []
    for i in range (len(readAcrossURIs)):
        #print "\n HERE \n ", eucl_sorted[i]
        #eucl_dict.append(mat2dicNN(eucl_sorted[i], readAcrossURIs[i])) #
        for j in range (len (eucl_sorted[i][0])):
            if eucl_sorted[i][1][j] < threshold:
                eucl_dict[readAcrossURIs[i] + " NP's NN No." + str(j+1)] = [eucl_sorted[i][0][j], eucl_sorted[i][1][j]]
            else:
                break
    eucl_dict = byteify(eucl_dict)
    #print "\n\nDict\n\n",eucl_dict
    #print "\n\n\n WORKS FINE UNTIL HERE5\n\n\n"
    max_manh_dist = metrics.pairwise.manhattan_distances([term1], [term2]) ## []
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
            if manh_sorted[i][1][j] < threshold:
                manh_dict[readAcrossURIs[i] + " NP's NN No." + str(j+1)] = [manh_sorted[i][0][j], manh_sorted[i][1][j]]
            else: 
                break
    manh_dict = byteify(manh_dict)

    ensemble_dist = (eucl_dist + manh_dist)/2
    #print "Eucl.: ", eucl_dist, "\n Manh.: ", manh_dist,"\n Ens.: ", ensemble_dist

    #print "\n\n\n WORKS FINE UNTIL HERE 97\n\n\n"

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
            if ens_sorted[i][1][j] < threshold:
                ens_dict[readAcrossURIs[i] + " NP's NN No." + str(j+1)] = [ens_sorted[i][0][j], ens_sorted[i][1][j]]
            else:
                break
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
    #plt.show() #HIDE show on production

    figfile = BytesIO()
    pcafig.savefig(figfile, dpi=300, format='png', bbox_inches='tight') #bbox_inches='tight'
    figfile.seek(0) 
    pcafig_encoded = base64.b64encode(figfile.getvalue())    
    """
    #return 0,0,0,0,0,0,0 ## , ""
    return eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict # , pcafig_encoded# new 21/06/16 ->16/05/2018
    #return eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict, pcafig_encoded 

"""
    Prediction function
"""
def RA_predict(euclidean, manhattan, ensemble, name, predictionFeature, nano2value, threshold): ##
    eu_score = 0
    ma_score = 0
    en_score = 0

    eu_div = 0
    ma_div = 0
    en_div = 0

    for i in range (len(euclidean[0])):
        if euclidean[1][i] < threshold: #27/06/16 - was < 1:
            eu_score += (1 - euclidean[1][i])*(nano2value[euclidean[0][i]]) #just the name
            eu_div += 1 - euclidean[1][i]
        if manhattan[1][i] < threshold: #27/06/16 - was < 1:
            ma_score += (1 - manhattan[1][i])*(nano2value[euclidean[0][i]]) #just the name
            ma_div += 1 - manhattan[1][i]
            #print manhattan[1][i], threshold, i
        if ensemble[1][i] < threshold: #27/06/16 - was < 1:
            en_score += (1 - ensemble[1][i])*(nano2value[euclidean[0][i]]) #just the name
            en_div += 1 - ensemble[1][i]
    if eu_div == 0:
        eu_score = 0
    else: 
        eu_score = eu_score/eu_div

    if ma_div == 0:
        ma_score = 0
    else: 
        ma_score = ma_score/ma_div

    if en_div == 0:
        en_score = 0
    else: 
        en_score = en_score/en_div
    
    
    #print eu_score
    return [name, round(eu_score,2)], [name, round(ma_score,2)], [name, round(en_score,2)]

"""
    Pseudo AD
"""
def RA_applicability(euclidean, manhattan, ensemble, name, threshold): # FIX this to accept cut-offs -DONE
    eu_score = 0
    ma_score = 0
    en_score = 0
    eu_div = 0
    ma_div = 0
    en_div = 0

    for i in range (len(euclidean[1])): # list of vals
        if euclidean[1][i] < threshold: # 0.4:
            eu_score +=1 - euclidean[1][i] #1
            eu_div += 1 #- threshold
        if manhattan[1][i] < threshold: # 0.33:
            ma_score += 1 - manhattan[1][i] #1
            ma_div += 1 #- threshold
        if ensemble[1][i] < threshold: # 0.36:
            en_score +=1 - ensemble[1][i] #1
            en_div += 1 #- threshold

    #eu_score = eu_score/eu_div #len(euclidean[1])
    #ma_score = ma_score/ma_div #len(euclidean[1])
    #en_score = en_score/en_div #len(euclidean[1])
    #print "\n\n\n", eu_score, "\n\n\n"
    if eu_div == 0:
        eu_score = 0
    else: 
        eu_score = eu_score/eu_div

    if ma_div == 0:
        ma_score = 0
    else: 
        ma_score = ma_score/ma_div

    if en_div == 0:
        en_score = 0
    else: 
        en_score = en_score/en_div

    return [name, round(eu_score,2)], [name, round(ma_score,2)], [name, round(en_score,2)]
    
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
    #euclidean = parameters.get("euclidean", 0.4)
    #manhattan = parameters.get("manhattan", 0.33)
    #ensemble = parameters.get("ensemble", 0.36)
    #confidence = parameters.get("confidence", 1)
    distance = parameters.get("distance", "euclidean")
    threshold = parameters.get("threshold", 0.4)
    confidence = parameters.get("confidence", 0)

    taskDic = {}
    #taskDic["variables"] = variables # sent separately
    taskDic["substances"] = substances
    taskDic["datapoints"] = datapoints
    taskDic["predictionFeature"] = predictionFeature
    taskDic["target_variable_values"] = target_variable_values
    #taskDic["euclidean"] = euclidean
    #taskDic["manhattan"] = manhattan
    #taskDic["ensemble"] = ensemble
    taskDic["distance"] = distance
    taskDic["confidence"] = confidence
    taskDic["threshold"] = threshold

    encoded = base64.b64encode(str(taskDic))

    #predictedString = predictionFeature + " predicted" ## removed # temp # 'new' 21/06/16

    # new 21/06/16
    """
    predictedString1 = predictionFeature + "\sEuclidean" # fix this to accept names
    predictedString2 = predictionFeature + "\sManhattan"
    predictedString3 = predictionFeature + "\sEnsemble"
    predictedString4 = predictionFeature + "\sConfidence\sEuclidean"
    predictedString5 = predictionFeature + "\sConfidence\sManhattan"
    predictedString6 = predictionFeature + "\sConfidence\sEnsemble"
    """
    predictedString1 = predictionFeature + " " + distance # fix this to accept names
    predictedString2 = predictionFeature + " " + "confidence"

    # new 21/06/16 # check -> some is only for internal usage
    """
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeature': predictionFeature + " predicted"}], 
        "independentFeatures": variables, 
        "predictedFeatures": [
            predictedString1, predictedString2, predictedString3, predictedString4, predictedString5, predictedString6  
            ] 
        }
    task = {
        "rawModel": encoded,
        "pmmlModel": "", 
        "additionalInfo" : [{'predictedFeatures': [predictedString1, predictedString2, predictedString3, predictedString4, predictedString5, predictedString6]}], 
        "independentFeatures": variables, 
        "predictedFeatures": [
            predictedString1, predictedString2, predictedString3, predictedString4, predictedString5, predictedString6  
            ] 
        }
    """
    if confidence == 1:
        task = {
            "rawModel": encoded,
            "pmmlModel": "", 
            "additionalInfo" : [{'predictedFeatures': [predictedString1, predictedString2]}], 
            "independentFeatures": variables, 
            "predictedFeatures": [
                predictedString1, predictedString2  
                ] 
            }
    else:
        task = {
            "rawModel": encoded,
            "pmmlModel": "", 
            "additionalInfo" : [{'predictedFeatures': [predictedString1]}], 
            "independentFeatures": variables, 
            "predictedFeatures": [
                predictedString1  
                ] 
            }
    ## DEBUG
    """
    xxx = open("C:\Python27\Flask-0.10.1\python-api\RA_train_Delete.txt", "w")
    xxx.writelines(str(encoded)) #or "task"
    xxx.close 
    """
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

    variables, read_across_datapoints, predictionFeature, rawModel, readAcrossURIs, predictedFeatures  = getJsonTestRA(readThis)

    got_raw = base64.b64decode(rawModel) 
    decoded = ast.literal_eval(got_raw) 

    datapoints = decoded["datapoints"] 
    substances = decoded["substances"] 
    predictionFeature = decoded["predictionFeature"] 
    target_variable_values = decoded["target_variable_values"]  

    distance = decoded["distance"] 
    threshold = decoded["threshold"] 
    confidence = decoded["confidence"] 

    ### test threshold manually
    #threshold = 0.1
    #threshold = 1

    euclidean = threshold
    manhattan = threshold
    ensemble = threshold 

    ### Leave in (in case we allow multiple distance metrics)
    #euclidean = decoded["euclidean"] 
    #manhattan = decoded["manhattan"] 
    #ensemble = decoded["ensemble"] 

    ### test confidence manually
    #confidence = 0
    #confidence = 1

    nano2value = {}
    for i in range (len(substances)):
        nano2value[substances[i]] = target_variable_values[i]
    eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict = distances (read_across_datapoints, datapoints, variables, readAcrossURIs, substances, threshold) ## 16 05 2018
    #eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict, pcafig_encoded = distances (read_across_datapoints, datapoints, variables, readAcrossURIs, substances, threshold)

    ## PREVIOUS 16 05 2018
    #eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict, pcafig_encoded = distances (read_across_datapoints, datapoints, variables, readAcrossURIs, substances, threshold)

    #"""
    eucl_predictions = []
    manh_predictions = [] 
    ens_predictions = []

    eucl_applicability = []
    manh_applicability = [] 
    ens_applicability = []
    for i in range (len(readAcrossURIs)):
        eu,ma,en = RA_predict(eucl_sorted[i], manh_sorted[i], ens_sorted[i], readAcrossURIs[i], predictionFeature, nano2value, threshold)
        eucl_predictions.append(eu)
        manh_predictions.append(ma)
        ens_predictions.append(en)
        
        eu,ma,en = RA_applicability(eucl_sorted[i], manh_sorted[i], ens_sorted[i], readAcrossURIs[i], threshold)
        eucl_applicability.append(eu)
        manh_applicability.append(ma)
        ens_applicability.append(en)
    #"""

    if len (eucl_predictions) > 1:
        # predictions
        eucl_predictions_transposed = map(list, zip(*eucl_predictions)) 
        eucl_pred_dict = mat2dic(eucl_predictions_transposed) # Checked: Changed mat2dic # new 21/06/16

        manh_predictions_transposed = map(list, zip(*manh_predictions)) 
        manh_pred_dict = mat2dic(manh_predictions_transposed)

        ens_predictions_transposed = map(list, zip(*ens_predictions)) 
        ens_pred_dict = mat2dic(ens_predictions_transposed)

        # applicability
        eucl_applicability_transposed = map(list, zip(*eucl_applicability)) 
        eucl_appl_dict = mat2dic(eucl_applicability_transposed)

        manh_applicability_transposed = map(list, zip(*manh_applicability)) 
        manh_appl_dict = mat2dic(manh_applicability_transposed)

        ens_applicability_transposed = map(list, zip(*ens_applicability)) 
        ens_appl_dict = mat2dic(ens_applicability_transposed)
    else: 
        eucl_pred_dict = mat2dicSingle(eucl_predictions[0])
        manh_pred_dict = mat2dicSingle(manh_predictions[0])
        ens_pred_dict = mat2dicSingle(ens_predictions[0])
        eucl_appl_dict = mat2dicSingle(eucl_applicability[0])
        manh_appl_dict = mat2dicSingle(manh_applicability[0])
        ens_appl_dict = mat2dicSingle(ens_applicability[0])


    predictionList = []
    # check if parameters are capitalised!! 
    if distance == "euclidean":
        if confidence == 1:
            for i in range (len(readAcrossURIs)):
                predictionList.append({ predictedFeatures[0]: eucl_predictions[i][1], 
                                        predictedFeatures[1]: eucl_applicability[i][1]})
        else:
            for i in range (len(readAcrossURIs)):
                predictionList.append({ predictedFeatures[0]: eucl_predictions[i][1]})
    elif distance == "manhattan":
        if confidence == 1:
            for i in range (len(readAcrossURIs)):
                predictionList.append({ predictedFeatures[0]: manh_predictions[i][1], 
                                        predictedFeatures[1]: manh_applicability[i][1]})
        else:
            for i in range (len(readAcrossURIs)):
                predictionList.append({ predictedFeatures[0]: manh_predictions[i][1]})
    else:
        if confidence == 1:
            for i in range (len(readAcrossURIs)):
                predictionList.append({ predictedFeatures[0]: ens_predictions[i][1], 
                                        predictedFeatures[1]: ens_applicability[i][1]})
        else:
            for i in range (len(readAcrossURIs)):
                predictionList.append({ predictedFeatures[0]: ens_predictions[i][1]})

    ### Leave in (in case we end up supporting multiple)
    """
    for i in range (len(readAcrossURIs)):
        predictionList.append({ predictedFeatures[0]: eucl_predictions[i][1], 
                                predictedFeatures[1]: manh_predictions[i][1], 
                                predictedFeatures[2]: ens_predictions[i][1], 
                                predictedFeatures[3]: eucl_applicability[i][1], 
                                predictedFeatures[4]: manh_applicability[i][1],
                                predictedFeatures[5]: ens_applicability[i][1] })
    """

    task = {
        "predictions": predictionList
        }

    ### DEBUG
    #xxx = open("C:/Python27/RApredict_delete.txt", "w")
    #xxx.writelines(str(task))
    #xxx.close 

    #task = {}

    jsonOutput = jsonify( task )
    
    return jsonOutput, 201 

	
"""
    Report (prediction)
"""
@app.route('/pws/readacross/report', methods = ['POST'])
def create_task_readacross_report():

    if not request.environ['body_copy']:
        abort(500)

    readThis = json.loads(request.environ['body_copy'])

    variables, read_across_datapoints, predictionFeature, rawModel, readAcrossURIs, predictedFeatures  = getJsonTestRA(readThis)

    got_raw = base64.b64decode(rawModel) 
    decoded = ast.literal_eval(got_raw) 

    datapoints = decoded["datapoints"] 
    substances = decoded["substances"] 
    predictionFeature = decoded["predictionFeature"] 
    target_variable_values = decoded["target_variable_values"]  

    distance = decoded["distance"] 
    threshold = decoded["threshold"] 
    confidence = decoded["confidence"] 

    ### test threshold manually
    #threshold = 0.1
    #threshold = 1

    euclidean = threshold
    manhattan = threshold
    ensemble = threshold 

    ### Leave in (in case we allow multiple distance metrics)
    #euclidean = decoded["euclidean"] 
    #manhattan = decoded["manhattan"] 
    #ensemble = decoded["ensemble"] 

    ### test confidence manually
    #confidence = 0
    #confidence = 1

    nano2value = {}
    for i in range (len(substances)):
        nano2value[substances[i]] = target_variable_values[i]
    eucl_sorted, eucl_dict, manh_sorted, manh_dict, ens_sorted, ens_dict, pcafig_encoded  = distances (read_across_datapoints, datapoints, variables, readAcrossURIs, substances, threshold)
    
    #"""
    eucl_predictions = []
    manh_predictions = [] 
    ens_predictions = []

    eucl_applicability = []
    manh_applicability = [] 
    ens_applicability = []
    for i in range (len(readAcrossURIs)):
        eu,ma,en = RA_predict(eucl_sorted[i], manh_sorted[i], ens_sorted[i], readAcrossURIs[i], predictionFeature, nano2value, threshold)
        eucl_predictions.append(eu)
        manh_predictions.append(ma)
        ens_predictions.append(en)
        
        eu,ma,en = RA_applicability(eucl_sorted[i], manh_sorted[i], ens_sorted[i], readAcrossURIs[i], threshold)
        eucl_applicability.append(eu)
        manh_applicability.append(ma)
        ens_applicability.append(en)
    #"""

    if len (eucl_predictions) > 1:
        # predictions
        eucl_predictions_transposed = map(list, zip(*eucl_predictions)) 
        eucl_pred_dict = mat2dic(eucl_predictions_transposed) # Checked: Changed mat2dic # new 21/06/16

        manh_predictions_transposed = map(list, zip(*manh_predictions)) 
        manh_pred_dict = mat2dic(manh_predictions_transposed)

        ens_predictions_transposed = map(list, zip(*ens_predictions)) 
        ens_pred_dict = mat2dic(ens_predictions_transposed)

        # applicability
        eucl_applicability_transposed = map(list, zip(*eucl_applicability)) 
        eucl_appl_dict = mat2dic(eucl_applicability_transposed)

        manh_applicability_transposed = map(list, zip(*manh_applicability)) 
        manh_appl_dict = mat2dic(manh_applicability_transposed)

        ens_applicability_transposed = map(list, zip(*ens_applicability)) 
        ens_appl_dict = mat2dic(ens_applicability_transposed)
    else: 
        eucl_pred_dict = mat2dicSingle(eucl_predictions[0])
        manh_pred_dict = mat2dicSingle(manh_predictions[0])
        ens_pred_dict = mat2dicSingle(ens_predictions[0])
        eucl_appl_dict = mat2dicSingle(eucl_applicability[0])
        manh_appl_dict = mat2dicSingle(manh_applicability[0])
        ens_appl_dict = mat2dicSingle(ens_applicability[0])


    predictionList = []
    # check if parameters are capitalised!! 
    pred_dict = {}
    appl_dict = {} 
    nn_dict = {}
    if confidence == 1:
        if distance == "euclidean":
            pred_dict = eucl_pred_dict
            nn_dict = eucl_dict
            appl_dict = eucl_appl_dict
        elif distance == "manhattan":
            pred_dict = manh_pred_dict
            nn_dict = manh_dict
            appl_dict = manh_appl_dict
        else:
            pred_dict = ens_pred_dict
            nn_dict = ens_dict
            appl_dict = ens_appl_dict
        task = {
                "singleCalculations": {
                                       distance + " Cut-off" : threshold
                                      },
                "arrayCalculations": {
                                       "Predictions based on " + distance + " Distances":
                                           {"colNames": ["Nanoparticle", "Prediction"],
                                            "values": pred_dict
                                           },
                                       "Applicability Domain for " + distance + "Distances":
                                           {"colNames": ["Nanoparticle", "AD Value"],
                                            "values": appl_dict
                                           },
                                       "Nearest Neighbour based on " + distance + " Distances":
                                           {"colNames": ["Nanoparticle", "Distance"],
                                            "values": nn_dict
                                           }
                                     },
                "figures": {
                           "PCA of datapoints vs. Read-Across" : pcafig_encoded
                           }
            }
    else:
        if distance == "euclidean":
            pred_dict = eucl_pred_dict
            nn_dict = eucl_dict
        elif distance == "manhattan":
            pred_dict = manh_pred_dict
            nn_dict = manh_dict
        else:
            pred_dict = ens_pred_dict
            nn_dict = ens_dict
        task = {
                "singleCalculations": {
                                       distance + " Cut-off" : threshold
                                      },
                "arrayCalculations": {
                                       "Predictions based on " + distance + " Distances":
                                           {"colNames": ["Nanoparticle", "Prediction"],
                                            "values": pred_dict
                                           },
                                       "Nearest Neighbour based on " + distance + " Distances":
                                           {"colNames": ["Nanoparticle", "Distance"],
                                            "values": nn_dict
                                           }
                                     },
                "figures": {
                           "PCA of datapoints vs. Read-Across" : pcafig_encoded
                           }
            }

    ### DEBUG
    #xxx = open("C:/Python27/RA_report_delete.txt", "w")
    #xxx.writelines(str(task))
    #xxx.close 

    #task = {}

    jsonOutput = jsonify( task )
    
    return jsonOutput, 201 

############################################################

# Middleware for chunked input
#from http://stackoverflow.com/questions/14146824/flask-and- transfer-encoding-chunked/21342631

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
                while (1):
                    temp = input.readline() ## 
                    
                    if not temp:
                        break
                    body +=temp
            size = len(body)
        else:
            body = environ['wsgi.input'].read(length)
        environ['body_copy'] = body
        environ['wsgi.input'] = StringIO(body)
        app_iter = self.application(environ, 
                                    self._sr_callback(start_response))
        return app_iter

    def _sr_callback(self, start_response):
        def callback(status, headers, exc_info=None):
            start_response(status, headers, exc_info)
        return callback

if __name__ == '__main__': 
    app.wsgi_app = WSGICopyBody(app.wsgi_app) ##
    app.run(host="0.0.0.0", port = 5000, debug = True)	

############################################################

### DEBUG (local)
# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/ractrain2.json http://localhost:5000/pws/readacross/train
# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/ractest2.json http://localhost:5000/pws/readacross/test
# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/ractest2.json http://localhost:5000/pws/readacross/report
# C:\Python27\Flask-0.10.1\python-api 
# C:/Python27/python rac.py