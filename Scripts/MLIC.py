import time
import sys
import os
import argparse
import pickle
import numpy
from subprocess import STDOUT, check_output

def ParseFiles(datafile):
    x = pickle.load(open(datafile,"rb"))
    AMatrix = x['A']
    yVector  = x['y']
    groupList = []
    groupMap = {}
    currentIndex = 0
    listPosition = 0
    groupList.append([])
    previousElement = x['col_to_feat'][0]
    currentGroupIndex = 0
    for i in x['col_to_feat']:
        listPosition += 1
        if (not(i == previousElement)):
            currentGroupIndex += 1
        if (not(i == previousElement)):
            currentIndex += 1
            currElement = i
            groupList.append([])
            previousElement = i
        groupMap[listPosition] = currentGroupIndex
        groupList[currentIndex].append(listPosition)
    return AMatrix,yVector,groupList,groupMap,len(AMatrix[0])
#    f = open(AMatrixFile,'r')
#    lines = f.readlines()
#    f.close()
#    AMatrix = {}
#    rowIndex = 0
#    for line in lines:
#        AMatrix[rowIndex] = {}
#        fields = line.split(',')
#        colIndex = 0
#        for field in fields:
#            AMatrix[rowIndex][colIndex] = int(field)
#            colIndex += 1
#        rowIndex += 1
#    f = open(yVectorFile,'r')
#    lines = f.readlines()
#    f.close()
#    yVector = {}
#    rowIndex = 0
#    for line in lines:
#        yVector[rowIndex] = int(line.strip())
#        rowIndex += 1
#    return AMatrix,yVector,len(AMatrix[0])
def GenerateWCNFFile(AMatrix,yVector,alpha,beta,xSize,wCNFFileName):
    cnfClauses = ''
    topWeight = alpha*len(yVector)+1+beta*xSize
    numClauses = 0
    for i in range(1,xSize+1):
        numClauses += 1
        cnfClauses += str(beta)+' '+str(-i)+' 0\n'
    for i in range(xSize+1, xSize+len(yVector)+1):
        numClauses += 1
        cnfClauses += str(alpha)+' '+str(-i)+' 0\n'
    for i in range(len(yVector)):
        noise = xSize+i+1
        implyClause = ''
        reverseImplyClauses = ''
        if (yVector[i] == 0):
            noise = -noise
        implyClause += str(topWeight)+' '+str(-noise)+' '
        for j in range(len(AMatrix[i])):
            if (AMatrix[i][j] == 1):
                continue
            numClauses += 1
            implyClause += str((j+1)*(1-AMatrix[i][j]))+' '
            reverseImplyClauses += str(topWeight)+' '+str(noise)+' '+str(-1*(j+1)*(1-AMatrix[i][j]))+' 0\n'
        implyClause += ' 0\n'
        numClauses += 1
        cnfClauses += implyClause
        cnfClauses += reverseImplyClauses
    header = 'p wcnf '+str(xSize+2*(len(yVector)))+' '+str(numClauses)+' '+str(topWeight)+'\n'
    f = open(wCNFFileName,'w')
    f.write(header)
    f.write(cnfClauses)
    f.close()
def GenerateWCNFFileImplication(AMatrix,yVector,alpha,beta,xSize,wCNFFileName):
    cnfClauses = ''
    topWeight = alpha*len(yVector)+1+beta*xSize
    numClauses = 0
    for i in range(1,xSize+1):
        numClauses += 1
        cnfClauses += str(beta)+' '+str(-i)+' 0\n'
    for i in range(xSize+1, xSize+len(yVector)+1):
        numClauses += 1
        cnfClauses += str(alpha)+' '+str(-i)+' 0\n'
    for i in range(len(yVector)):
        noise = xSize+i+1
        implyClause = ''
        reverseImplyClauses = ''
        implyClause += str(topWeight)+' '+str(noise)+' '
        for j in range(len(AMatrix[i])):
            if (AMatrix[i][j] == 1):
                continue
            if (yVector[i] == 1):
                numClauses += 1
            implyClause += str((j+1)*(1-AMatrix[i][j]))+' '
            reverseImplyClauses += str(topWeight)+' '+str(noise)+' '+str(-1*(j+1)*(1-AMatrix[i][j]))+' 0\n'
        
        implyClause += ' 0\n'
        if (yVector[i] == 0):
            numClauses += 1
            cnfClauses += implyClause
        else:
            cnfClauses += reverseImplyClauses
    header = 'p wcnf '+str(xSize+(len(yVector)))+' '+str(numClauses)+' '+str(topWeight)+'\n'
    f = open(wCNFFileName,'w')
    f.write(header)
    f.write(cnfClauses)
    f.close()
def ExtractClausesFromCNFFile(cnfFileName,topWeight,noise,auxVariableStart):
    f = open(cnfFileName,'r')
    lines =f.readlines()
    f.close()
    addedClauses = 0
    cnfClauses = ''
    headerSeen = False
    for line in lines:
        if (line.strip().startswith("p cnf")):
            headerSeen = True
            fields = line.strip().split()
            totalVars = int(fields[2])
            if (totalVars >= auxVariableStart):
                auxVariableStart = totalVars+1
            continue
        if (headerSeen):
            if (line.strip().startswith('c')):
                continue
            addedClauses += 1
            cnfClauses += str(topWeight)+" "
            if (not(noise == 0)):
                cnfClauses += str(noise)+" "
            cnfClauses += line.strip()+"\n"
    return (cnfClauses,addedClauses,auxVariableStart)
def GeneratePositiveConstraints(tempPBFile, tempOutFile, xSize, topWeight, AMatrix,rowNum, 
        matrixTrue, mValue, noise, auxVariableStart,level):
    matrixFalse = 1-matrixTrue
    cnfClauses = ''
    numClauses = 0
    opbClauses = ''
    addedNewClauses = ''
    addedNumClauses = 0
    for i in range(level):
        clause = ''
        rowLen = 0
        kValue = mValue
        for j in range(len(AMatrix[rowNum])):
            if (AMatrix[rowNum][j] == matrixFalse):
                continue
            rowLen += 1
            clause += "+1 x"+str(i*xSize+j+1)+" "
        clause += " >= "+str(kValue)+ ";\n"
        opbClauses += str(noise)+':'+clause
        #f = open(tempPBFile,'w')
        #f.write("* #variable= "+str(xSize)+" #constraint= 1\n*\n")
        #f.write(clause)
        #f.close()
        cmd = "pbencoder "+tempPBFile+" -auxVar="+str(auxVariableStart)+"  > "+str(tempOutFile)
        #os.system(cmd)
        #(addedNewClauses, addedNumClauses, auxVariableStart) =  ExtractClausesFromCNFFile(tempOutFile,topWeight,noise, auxVariableStart)
        cnfClauses += addedNewClauses
        numClauses += addedNumClauses
    return (cnfClauses,numClauses,auxVariableStart,opbClauses)
def DirectlyGenerateNegativeConstraints(tempPBFile, tempOutFile, xSize, topWeight, AMatrix,rowNum, matrixTrue, mValue, 
        noise, groupRowNoise, groupMap, groupNoiseFlag, auxVariableStart, level):
    clause = ''
    rowLen = 0
    matrixFalse = 1-matrixTrue
    maxVarIndex = xSize
    addedClauses = ''
    addedNumClauses = 0
    groupClause = ''
    firstGroupClause = True
    levelVariable = {}
    
    for i in range(level):
        levelVariable[i] = auxVariableStart
        for j in range(len(AMatrix[rowNum])):
            if (AMatrix[rowNum][j] == matrixFalse):
                continue
            addedNumClauses += 1
            addedClauses +=str(topWeight)+' '+str(-(i*xSize+j+1))+' '+str(-levelVariable[i])+' '
            if (groupNoiseFlag):
                groupNoiseVar = -1
                if (groupMap[j+1] in groupRowNoise[rowNum]):
                    groupNoiseVar = groupRowNoise[rowNum][groupMap[j+1]]
                else:
                    groupNoiseVar = auxVariableStart
                    auxVariableStart += 1
                    groupRowNoise[rowNum][groupMap[j+1]] = groupNoiseVar
                    if (not(firstGroupClause)):
                        groupClause += '0\n'
                        addedNumClauses += 1
                    else:
                        firstGroupClause = False
                    groupClause += str(topWeight)+' '+str(-groupNoiseVar)+' '+str(noise)+' '
                if(groupNoiseVar > maxVarIndex):
                    maxVarIndex = groupNoiseVar
                addedClauses += str(groupNoiseVar)+' '
                groupClause += str(j+1)+' '
            addedClauses +=' 0\n'
        if (groupNoiseFlag):
            addedNumClauses += 1
            groupClause += ' 0\n'
            addedClauses += groupClause
        auxVariableStart += 1
    addedClauses += str(topWeight)+' '+str(noise)+' '
    for i in range(level):
        addedClauses += str(levelVariable[i])+' '
    addedClauses += ' 0\n'
    addedNumClauses += 1
    #print(AMatrix[rowNum])
    #print(addedClauses)
    return (addedClauses, addedNumClauses, auxVariableStart, groupRowNoise)
def GenerateNegativeConstraints(tempPBFile, tempOutFile, xSize, topWeight, AMatrix,rowNum, matrixTrue, mValue, 
        noise, groupRowNoise, groupMap, groupNoiseFlag, auxVariableStart):
    #TODO: GroupNoise encoding still to be added fully
    clause = ''
    rowLen = 0
    matrixFalse = 1-matrixTrue
    maxVarIndex = xSize
    groupClause = ''
    opbClauses = ''
    addedNumClauses = 0
    addedClauses = ''
    for j in range(len(AMatrix[rowNum])):
        if (AMatrix[rowNum][j] == matrixFalse):
            continue
        rowLen += 1
        clause += "+1 x"+str(j+1)+" "
        if (groupNoiseFlag):
            groupNoiseVar = -1
            if (groupMap[j+1] in groupRowNoise[rowNum]):
                groupNoiseVar = groupRowNoise[rowNum][groupMap[j+1]]
            else:
                groupNoiseVar = auxVariableStart
                auxVariableStart += 1
                groupRowNoise[rowNum][groupMap[j+1]] = groupNoiseVar
                groupClause += '+1 x'+str(groupNoiseVar)
            if(groupNoiseVar > maxVarIndex):
                maxVarIndex = groupNoiseVar
            clause += "~x"+str(groupNoiseVar)+" "
            groupClause += " ~x"+str(j+1)+" "
    #kValue = rowLen-mValue+1
    clause += groupClause
    clause += " < "+str(mValue)+";\n"
    #clause += " >= "+str(kValue)+ ";\n"
    opbClauses += str(noise)+':'+clause
    #f = open(tempPBFile,'w')
    #f.write("* #variable= "+str(maxVarIndex)+" #constraint= 1\n*\n")
    #f.write(clause)
    #f.close()
    #cmd = "pbencoder "+tempPBFile+" -auxVar="+str(auxVariableStart)+"  > "+str(tempOutFile)
    #os.system(cmd)
    #(addedClauses, addedNumClauses, auxVariableStart) = ExtractClausesFromCNFFile(tempOutFile,topWeight,noise, auxVariableStart)
    return (addedClauses, addedNumClauses, auxVariableStart, groupRowNoise, opbClauses)
def GenerateWCNFFileForPB(AMatrix,yVector, alpha, beta, gamma, mValue,xSize,wCNFFileName,groupList,
            groupMap,groupNoiseFlag,level,runIndex,assignList):
    cnfClauses = ''
    numClauses = 0
    if (not(groupNoiseFlag)):
        gamma = 0
    topWeight = alpha*len(yVector)+beta*2*xSize+gamma*len(yVector)*len(groupList)+1
    matrixTrue = 1
    clauseList = ''
    matrixFalse = 1-matrixTrue
    for i in range(1,level*xSize+1):
        numClauses += 1
        if (i%xSize != 1):
            cnfClauses += str(beta)+' '+str(-i)+' 0\n'
    auxVariableStart = level*xSize+1
    for i in range(auxVariableStart, auxVariableStart+len(yVector)):
        weight = alpha
        if (yVector[i-auxVariableStart-1] == matrixFalse and groupNoiseFlag):
            weight = topWeight
        numClauses += 1
        cnfClauses += str(weight)+' '+str(-i)+' 0\n'
    for i in assignList:
        numClauses += 1
        cnfClauses += str(topWeight)+' '+str(i)+' 0\n'
    tempPBFile = wCNFFileName[:-5]+"_"+str(runIndex)+"_temp.pb"
    tempOutFile = wCNFFileName[:-5]+"_"+str(runIndex)+"_temp.out"
    auxVariableStart += len(yVector)
    groupRowNoise = {}
    opbClauses = ''
    for i in range(len(yVector)):
        noise = (level*xSize+i+1)
        addedOPBClause = ''
        if (yVector[i] == matrixTrue):
            (addedClauses,addedNumClauses,auxVariableStart,addedOPBClause) = GeneratePositiveConstraints(tempPBFile, tempOutFile, xSize, topWeight, AMatrix,i, 
                    matrixTrue, mValue, noise, auxVariableStart,level)
        else:
            if (i not in groupRowNoise):
                groupRowNoise[i] = {}
            if (False and mValue==1):
                (addedClauses,addedNumClauses,auxVariableStart, groupRowNoise) = DirectlyGenerateNegativeConstraints(tempPBFile, tempOutFile, xSize,topWeight,
                    AMatrix, i, matrixTrue, mValue, noise, groupRowNoise, groupMap, groupNoiseFlag, auxVariableStart, level)
            else:
                (addedClauses,addedNumClauses,auxVariableStart, groupRowNoise, addedOPBClause) = GenerateNegativeConstraints(tempPBFile, tempOutFile, xSize,topWeight,
                        AMatrix, i, matrixTrue, mValue, noise, groupRowNoise, groupMap, groupNoiseFlag, auxVariableStart)
        cnfClauses += addedClauses
        numClauses += addedNumClauses
        opbClauses += addedOPBClause
        continue
        kValue =mValue
        clause = ''
        rowLen = 0
        for j in range(len(AMatrix[i])):
            if (AMatrix[i][j] == matrixFalse):
                continue
            rowLen += 1
            if (yVector[i] == matrixTrue):
                clause += "+1 x"+str(j+1)+" "
            else:
                clause += "+1 ~x"+str(j+1)+" "
        if (yVector[i] == matrixFalse):
            kValue = rowLen-mValue+1
        clause += " >= "+str(kValue)+ ";\n"
        clauseList += str(-noise)+":"+clause
        f = open(tempPBFile,'w')
        f.write("* #variable= "+str(xSize)+" #constraint= 1\n*\n")
        f.write(clause)
        f.close()
        cmd = "pbencoder "+tempPBFile+" -auxVar="+str(auxVariableStart)+"  > "+str(tempOutFile)
        os.system(cmd)
        (addedClauses,addedNumClauses,auxVariableStart) = ExtractClausesFromCNFFile(tempOutFile,topWeight,noise, auxVariableStart)
        cnfClauses += addedClauses
        numClauses += addedNumClauses
       #if (yVector[i] == 1):
        #    exit(0)
    for rowNum in groupRowNoise:
        for group in groupRowNoise[rowNum]:
            numClauses += 1
            cnfClauses += str(gamma)+' '+str(-groupRowNoise[rowNum][group])+' 0\n'
    for group in groupList:
        clause = ''
        if (len(group) <= 1):
            continue
        for element in group:
            clause += "+1 ~x"+str(element)+" "
        clause += " >= "+str(len(group) -1)+";\n"
        #clauseList += clause
        f = open(tempPBFile,'w')
        f.write("* #variable= "+str(max(group))+" #constraint= 1\n*\n")
        f.write(clause)
        f.close()
        cmd = "pbencoder "+tempPBFile+" -auxVar="+str(auxVariableStart)+" > "+str(tempOutFile)
        os.system(cmd)
        cnfClauses += addedClauses
        numClauses += addedNumClauses
    opbFileName = wCNFFileName[:-5].replace("/tmp","PBFiles")+"_"+str(mValue)+".opb"
    #f = open(opbFileName,'w')
    #f.write("* #variable= "+str(xSize+len(yVector))+" #constraint= "+str(len(yVector))+"\n")
    #f.write("* origVar = "+str(xSize)+"\n*\n")
    #f.write(opbClauses)
    #f.close()
    header = 'p wcnf '+str(auxVariableStart-1)+" "+str(numClauses)+" "+str(topWeight)+'\n'
    cmd = 'rm '+tempOutFile+" "+tempPBFile
    os.system(cmd)
    f = open(wCNFFileName,'w')
    f.write(header)
    f.write(cnfClauses)
    f.close()
def usage():
    print("python LearnRules.py <AMatrixFile> <yVectorFile> <mValue> <alpha> <beta>")
    exit(0)

def LearnRules(datafile,mValue,alpha,beta, gamma, timeoutSec, rule_type, level, groupNoiseFlag,runIndex,assignList):
    wCNFFileName = datafile[:-3]+"_"+str(runIndex)+"_maxsat_rule.wcnf"
    outFileName = datafile[:-3]+"_"+str(runIndex)+"_out.txt"
    startTime= time.time()
    AMatrix,yVector,groupList,groupMap,xSize = ParseFiles(datafile)
    endTime = time.time()
    print("Time taken to parse:"+str(endTime-startTime))
    startTime = time.time()
    noiseMultiplyFactor = 1
    if (rule_type == 'and'):
        noiseMultiplyFactor = 1
    if (False and (mValue == 1)):
        GenerateWCNFFileImplication(AMatrix,yVector,alpha,beta,xSize,wCNFFileName)
    else:
        print("Calling PB Encoder")
        GenerateWCNFFileForPB(AMatrix,yVector,alpha,beta,gamma,mValue,xSize,wCNFFileName,groupList,
                    groupMap, groupNoiseFlag,level,runIndex,assignList)
    endTime = time.time()
    print("Time taken to model:"+str(endTime-startTime))
    #cmd = 'open-wbo_release '+wCNFFileName+' > '+outFileName
    cmd = 'maxhs -printBstSoln -cpu-lim='+str(timeoutSec)+' '+wCNFFileName+' > '+outFileName
    #cmd = 'LMHS '+wCNFFileName+' > '+outFileName
    #command = ['open-wbo_release', wCNFFileName, ' > ',outFileName]
    #command =['maxhs', '-printBstSoln', '-cpu-lim='+str(timeoutSec), wCNFFileName,' > ',outFileName]
    startTime = time.time()
    os.system(cmd)
    #print(command)
    #output = check_output(command, stderr=STDOUT, timeout=timeoutSec+20)
    endTime = time.time()
    print("Time taken to find the solution:"+str(endTime-startTime))
    f = open(outFileName,'r')
    lines = f.readlines()
    f.close()
    optimumFound = False
    bestSolutionFound = False
    solution = ''
    for line in lines:
        if (line.strip().startswith('v') and optimumFound):
            solution = line.strip().strip('v ')
            break
        if (line.strip().startswith('c ') and bestSolutionFound):
            solution = line.strip().strip('c ')
            break
        if (line.strip().startswith('s OPTIMUM FOUND')):
            optimumFound = True
            print("Optimum solution found")
        if (line.strip().startswith('c Best Model Found:')):
            bestSolutionFound = True
            print("Best solution found")
    fields = solution.split()
    TrueRules = []
    TrueErrors = []
    zeroOneSolution = []
    for field in fields:
        if (int(field) > 0):
            zeroOneSolution.append(1.0)
        else:
            zeroOneSolution.append(0.0)
        if (int(field) > 0):
            
            if (abs(int(field))<=level*xSize):
                TrueRules.append(field)
            
            elif (abs(int(field)) <= level*xSize+len(yVector)):
                TrueErrors.append(field)
    #print(solution)
    print("Cost of the best rule:"+str(alpha*len(TrueErrors)+beta*len(TrueRules)))
    print("The number of True Rule are:"+str(len(TrueRules)))
    print("The number of errors are: "+str(len(TrueErrors))+" out of "+str(len(yVector)))
    #print("The True Rules are: "+str(TrueRules))
    #print("True Error are "+str(TrueErrors))
    xhat = []
    for i in range(level):
        xhat.append(numpy.array(zeroOneSolution[i*xSize:(i+1)*xSize]))
    err = numpy.array(zeroOneSolution[xSize: xSize+len(yVector)])
    print("xSize:"+str(xSize))
    return xhat,err,fields[:level*xSize]
def runTool():
    parser=argparse.ArgumentParser()
    parser.add_argument("dataFile",help="datafile")
    parser.add_argument("--k", type=int, help="give value of k",default=1)
    parser.add_argument("--alpha",type=int, help="alpha",default=10)
    parser.add_argument("--beta",type=int,help="beta",default=1)
    parser.add_argument("--lambda", type=int, help="lambda", default=10)
    parser.add_argument("--timeout",type=int,help="timeout in seconds",default=300)
    parser.add_argument("--runIndex",type=int,help="run Index", default=1)
    args = parser.parse_args()
    datafile = args.dataFile
    mValue = args.m
    alpha = args.alpha
    beta = args.beta
    timeoutSec=args.timeout
    groupNoiseFlag = False
    runIndex = args.runIndex
    if (mValue < 1):
        usage()
    LearnRules(datafile,mValue,alpha,beta,timeoutSec,'or', level, groupNoiseFlag, runIndex)
if __name__ == '__main__':
    runTool()
 
