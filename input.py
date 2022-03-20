import sys
import os
import numpy as np

def rot(theta,data,database):
    ret1 = np.cos(theta)*float(data[0]) + np.sin(theta)*float(data[1])
    ret2 = -np.sin(theta)*float(data[0]) + np.cos(theta)*float(data[1])
    return str(ret1) + "\t" + str(ret2) + "\t"

def rotY(theta,data,database):
    ret1 = np.cos(theta)*float(data[0]) + np.sin(theta)*float(data[2])
    ret2 = -np.sin(theta)*float(data[0]) + np.cos(theta)*float(data[2])
    return str(ret1) + "\t" +str(data[1]) + "\t" + str(ret2) + "\t"


def work(inp,out):
    database = ""
    inp1 = inp + "_location.txt"
    inp2 = inp + "_face.txt"
    inp3 = inp + "_velocity.txt"
    inp4 = inp + ".gait"
    i = 61
    with open(inp1,'r') as fi:
        openfile1 = fi.read().split('\n')
    with open(inp2,'r') as fi:
        openfile2 = fi.read().split('\n')
    with open(inp3,'r') as fi:
        openfile3 = fi.read().split('\n')
    with open(inp4, 'r') as fi:
        openfile4 = fi.read().split('\n')
    with open(out,'w') as fi:
        fi.write('')
    if(os.path.isfile(inp1) and os.path.isfile(inp2) and os.path.isfile(inp3)):
        while((i + 51 < len(openfile1)) and (i + 51 < len(openfile2))):
            if((len(openfile1[i+51]) != 0) and (len(openfile2[i+51]) != 0)):
                t = 0
                currunt = openfile1[i].split('\t')
                face = openfile2[i].split('\t')
                face_p = openfile2[i-1].split('\t')
                theta = np.arctan2(float(face[1]),float(face[0]))
                theta_p = np.arctan2(float(face_p[1]),float(face_p[0]))
                for x in range(12):
                    tem = openfile1[i - 60 + x*10].split('\t')
                    database += rot(theta,[(float(tem[0])-float(currunt[0])),(float(tem[2])-float(currunt[2]))],database)
                for x in range(12):
                    database += rot(theta, openfile2[i-60+x*10].split('\t'),database)
                for x in range(12):
                    database += openfile4[i - 60 + x*10] + '\t'
                pastP = openfile1[i-1].split('\t')
                while(3*t+2 < len(pastP)):
                    database += rotY(theta_p,[float(pastP[3*t])-float(pastP[0]),pastP[3*t+1],float(pastP[3*t+2])-float(pastP[2])],database)
                    t += 1
                t = 0
                pastV = openfile3[i-2].split('\t')
                while(3*t+2 < len(pastV)):
                    database += rotY(theta_p,[pastV[3*t],pastV[3*t+1],pastV[3*t+2]],database)
                    t += 1
                write_data(database,out)
                database = ""
            i += 1

def write_data(data, out):
    data = data[:-1]
    data += '\n'
    with open(out, 'a') as f:
        f.write(data)

def search():
    filepath = "C:\\Users\\YJ Jung\\bvh"
    for (path, dir, files) in os.walk(filepath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.bvh':
                stri = path + "\\" + filename[:-4] 
                out =  path + "\\" + filename[:-4] + ".input"
                work(stri,out)

search()                