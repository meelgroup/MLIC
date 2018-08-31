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



def GenerateWCNFFileImplication(AMatrix, yVector, alpha, beta, xSize, level, wCNFFileName, assignList):
    cnfClauses = ''

    # print(numpy.sum(map(int, yVector)))

    topWeight = alpha * len(yVector) + 1 + beta * xSize * level
    numClauses = 0
    # print("xSize",xSize)
    for i in range(1, level * xSize + 1):
        numClauses += 1
        cnfClauses += str(beta) + ' ' + str(-i) + ' 0\n'
    # print(cnfClauses)
    for i in range(level * xSize + 1, level * xSize + len(yVector) + 1):
        numClauses += 1
        cnfClauses += str(alpha) + ' ' + str(-i) + ' 0\n'
        # print(cnfClauses)
    for each_assign in assignList:
        numClauses += 1
        cnfClauses += str(topWeight) + ' ' + str(int(each_assign)) + ' 0\n'
        # print(str(topWeight) + ' ' + each_assign)
    for i in range(len(yVector)):
        noise = level * xSize + i + 1
        implyClause = ''
        reverseImplyClauses = ''
        implyClause += str(topWeight) + ' ' + str(noise) + ' '
        # print(len(AMatrix[i]))
        for each_level in range(level):
            for j in range(len(AMatrix[i])):
                if (AMatrix[i][j] == 1):
                    continue
                if (yVector[i] == 1):
                    numClauses += 1
                implyClause += str(int(j + each_level * xSize + 1) * int(1 - AMatrix[i][j])) + ' '
                reverseImplyClauses += str(topWeight) + ' ' + str(noise) + ' ' + str(
                    -1 * int(j + each_level * xSize + 1) * int(1 - AMatrix[i][j])) + ' 0\n'
                # print("im: "+implyClause)
                # print("ri: "+reverseImplyClauses)

            implyClause += ' 0\n'
            if (yVector[i] == 0):
                numClauses += 1
                cnfClauses += implyClause
                # print(cnfClauses)
            else:
                cnfClauses += reverseImplyClauses
                # print(cnfClauses)
    header = 'p wcnf ' + str(xSize * level + (len(yVector))) + ' ' + str(numClauses) + ' ' + str(topWeight) + '\n'
    # print("header", header)
    # print("cnfClauses", cnfClauses)
    f = open(wCNFFileName, 'w')
    f.write(header)
    # print(cnfClauses)
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
    GenerateWCNFFileImplication(AMatrix,yVector,alpha,beta,xSize,level,wCNFFileName,assignList)
    endTime = time.time()
    print("Time taken to model:"+str(endTime-startTime))
    #cmd = 'open-wbo_release '+wCNFFileName+' > '+outFileName


    # tool_path = "../Tools/"
    # cmd = tool_path + './maxhs -printBstSoln -cpu-lim=' + str(
    #     timeoutSec) + ' ' + wCNFFileName + ' > ' + outFileName


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
    print(len(fields))
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
    print("The True Rules are: "+str(TrueRules))
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
    parser.add_argument("--m", type=int, help="give value of M",default=1)
    parser.add_argument("--alpha",type=int, help="alpha",default=10)
    parser.add_argument("--beta",type=int,help="beta",default=1)
    parser.add_argument("--gamma", type=int, help="gamma", default=10)
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
 
