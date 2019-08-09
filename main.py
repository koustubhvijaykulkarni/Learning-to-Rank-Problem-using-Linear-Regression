
# coding: utf-8

# In[5]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math

# In[6]:

C_Lambda = 0.01 #This is regularization parameter- used to avoid overfitting
TrainingPercent = 80 #We are dividing our data set into training, validation and testing set. 80% for training, 10 % for validation adn 10 % for testing.
ValidationPercent = 10
TestPercent = 10
M = 10 #This indicates no. of clusters to be formed. Our model will have 'M' no of basis functions.
PHI = []
IsSynthetic = False


# In[158]:

#This method reads target csv file and stores targets into an array.
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'r') as f:
        reader = csv.reader(f) #Return a reader object which will iterate over lines in the target CSV file.
        for row in reader: 
            t.append(int(row[0]))
    return t
#This method reads dataset csv file stores data for 41 features and deletes 5 features as they are all 0s. So they won't make difference when we apply k means clustering to the dataset.
# Another reason we are deleting features with 0 values is, we would not be able to find inverse of bigsigma matrix if it has 0 value in it.
def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'r') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
    dataMatrix = np.transpose(dataMatrix)     
    return dataMatrix
#This method seprates 80% of target(0,1,2) data for training purpose.
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    return t
#This method seprates 80% of data for training purpose.
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    return d2
#This method seprates 10% of data for validation purpose. same code is used for separating data for testing purpose.
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix
#This method seprates 10% of target(0,1,2) data for validation purpose.same code is used for separating target data for testing purpose.
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t
#This method generates BIgSIgma Matrix which is required for Radial basis function calculation.
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))#Return a new array of given shape and type, filled with zeros.
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    print("printing len(DataT[0])",len(DataT[0]))
    print("int(TrainingLen)",int(TrainingLen))
    print("bigsigma shape",BigSigma.shape)
    print("DataT shape",DataT.shape)
    print("Data shape",Data.shape)
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])
        varVect.append(np.var(vct))
        #np.var-Returns the variance of the array elements, a measure of the spread of a distribution. 
    print("Data len",len(Data))
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    return BigSigma
#This method returns the scalar value for each basis function for each datapoint. These values will ultimately be part of our design matrix.
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow) # Represents (X-Mu) term in radial basis fucntion formula
    T = np.dot(BigSigInv,np.transpose(R)) # We are taking transpose of BIgSigma matrix that we generated and taking dot prodcut with (X-Mu)  
    L = np.dot(R,T)
    return L
#This method computes radial basis function for respictive data and Mu. It basically forms a design matrix. 
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x
#This method creates design matrix.
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma) #computes the inverse of our BigSigma matrix
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    return PHI
#This method is used to compute weights for closed form solution based on formula.
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0])) #Return the identity array of similar length data of PHI[0].
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    print("w shape--",W.shape)
    return W

def GetValTest(VAL_PHI,W):#here we calculate predicted output which we get from transpose of our design matrix and weight matrix.
    Y = np.dot(W,np.transpose(VAL_PHI))
    return Y
#This method computes ERMS for our training,validation and testing data. Our model is behavinf correct if the ERMS we get is low.
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    accuracy = 0.0
    counter = 0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# ## Fetch and Prepare Dataset

# In[127]:


RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)


# ## Prepare Training Data

# In[128]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[129]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[130]:


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(TestDataAct.shape)
print(TestData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[155]:


ErmsArr = []
AccuracyArr = []
#KMeans-we are using K-means clustering to find out M clusters in our data and apply M no. of basis functions to that dataset.
#Parameters and methods explained-
#n_clusters-The number of clusters to form as well as the number of centroids to generate.
#random_state-Determines random number generation for centroid initialization.
#fit-Compute k-means clustering.
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_
#cluster_centers_- returns the Coordinates of cluster centers i.e. 'Mu'
BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent) #Design matrix for training
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) #Weights generated in closed form solution. 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) #Design matrix for testing data
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100) #Design matrix for validation data


# In[156]:
print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[159]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))

# In[160]:


print ('UBITname      = Kkulkarn')
print ('Person Number = 50288207')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = "+str(M) +"\nLambda ="+str(C_Lambda))
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
print ("Training  Accuracy= " + str(float(TrainingAccuracy.split(',')[0])))
print ("Validation Accuracy= " + str(float(ValidationAccuracy.split(',')[0])))
print ("Testing    Accuracy= " + str(float(TestAccuracy.split(',')[0])))


# ## Gradient Descent solution for Linear Regression

# In[138]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[ ]:


W_Now        = np.dot(220, W) #In SGD, we are initalising our weights by scaling weights we got from closed form.
La           = 3 # This is regularization parameter
learningRate = 0.05 #This is learning rate which decides how fast algo will converge.
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
#I have defined three new arrays to store accuracy.
L_Accu_Val   = []
L_Accu_TR    = []
L_Accu_Test  = []
W_Mat        = []

for i in range(0,400):
    
    #Here we are calculating delta_ED,Delta E and Delta W. once we get these values we are calcuating new weights. In next few lines we are simply implementing formulas for these computations.
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    L_Accu_TR.append(float(Erms_TR.split(',')[0]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    L_Accu_Val.append(float(Erms_Val.split(',')[0]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    L_Accu_Test.append(float(Erms_Test.split(',')[0]))


# In[ ]:


print ('----------Gradient Descent Solution--------------------')
print ("M ="+str(M)+" \nLambda  = "+str(La)+"\neta=" +str(learningRate))
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
print ("Accuracy Training   = " + str(np.around(max(L_Accu_TR),5)))
print ("Accuracy Validation = " + str(np.around(max(L_Accu_Val),5)))
print ("Accuracy Testing    = " + str(np.around(max(L_Accu_Test),5)))
