import sys
import os
import numpy as np

def rot(theta,data):
    ret1 = np.cos(theta)*float(data[0]) + np.sin(theta)*float(data[1])
    ret2 = -np.sin(theta)*float(data[0]) + np.cos(theta)*float(data[1])
    return str(ret1) + "\t" + str(ret2) + "\t"

def rotY(theta,data):
    ret1 = np.cos(theta)*float(data[0]) + np.sin(theta)*float(data[2])
    ret2 = -np.sin(theta)*float(data[0]) + np.cos(theta)*float(data[2])
    return str(ret1) + "\t" +str(data[1]) + "\t" + str(ret2) + "\t"

def rotate_matrix(theta,zz,yy,xx):
    z = zz/180.*np.pi
    y = yy/180.*np.pi
    x = xx/180.*np.pi
    Z = np.array([[np.cos(z),-np.sin(z),0.],[np.sin(z),np.cos(z),0.],[0.,0.,1.]])
    Y = np.array([[np.cos(y),0.,np.sin(y)],[0.,1.,0.],[-np.sin(y),0.,np.cos(y)]])
    X = np.array([[1.,0.,0.],[0.,np.cos(x),-np.sin(x)],[0.,np.sin(x),np.cos(x)]])
    seta = np.array([[np.cos(theta),0.,np.sin(theta)],[0.,1.,0.],[-np.sin(theta),0.,np.cos(theta)]])
    ret = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

    ret = X@ret
    ret = Y@ret
    ret = Z@ret
    ret = seta@ret

    retZ = np.arctan2(ret[1][0],ret[0][0])*180./np.pi
    retY = np.arcsin(-ret[2][0])*180./np.pi
    retX = np.arctan2(ret[2][1],ret[2][2])*180./np.pi

    if(retZ <= 0.):
        retZ += 360.
    if(retX <= 0.):
        retX += 360.
    return [retZ,retY,retX]

def angle(theta,op):
    openfile = op.split('\t')
    T = rotate_matrix(theta,float(openfile[0]),float(openfile[1]),float(openfile[2]))
    ret = str(T[0]) + '\t' + str(T[1]) + '\t' + str(T[2]) + '\t'
    i = 3
    while(i < len(openfile)):
        ret += openfile[i] + '\t'
        i += 1
    return ret

def work(inp,out):
    database = ""
    inp1 = inp + "_location.txt"
    inp2 = inp + "_face.txt"
    inp3 = inp + "_velocity.txt"
    inp4 = inp + ".phase"
    inp5 = inp + "_label.txt"
    inp6 = inp + ".angle"
    i = 61
    with open(inp1,'r') as fi:
        openfile1 = fi.read().split('\n')
    with open(inp2,'r') as fi:
        openfile2 = fi.read().split('\n')
    with open(inp3,'r') as fi:
        openfile3 = fi.read().split('\n')
    with open(inp4,'r') as fi:
        openfile4 = fi.read().split('\n')
    with open(inp5,'r') as fi:
        openfile5 = fi.read().split('\n')
    with open(inp6,'r') as fi:
        openfile6 = fi.read().split('\n')
    with open(out,'w') as fi:
        fi.write('')
    if(os.path.isfile(inp1) and os.path.isfile(inp2) and os.path.isfile(inp3)):
        while((i + 51 < len(openfile1)) and (i + 51 < len(openfile2))):
            if((len(openfile1[i+51]) != 0) and (len(openfile2[i+51]) != 0)):
                t = 0
                currunt = openfile1[i+1].split('\t')
                face = openfile2[i+1].split('\t')
                face_p = openfile2[i].split('\t')
                face_pp = openfile2[i-1].split('\t')
                theta = np.arctan2(float(face[1]),float(face[0]))
                theta_p = np.arctan2(float(face_p[1]),float(face_p[0]))
                theta_pp = np.arctan2(float(face_pp[1]),float(face_pp[0]))
                for x in range(12):
                    tem = openfile1[i - 59 + x*10].split('\t')
                    database += rot(theta,[(float(tem[0])-float(currunt[0])),(float(tem[2])-float(currunt[2]))])
                for x in range(12):
                    database += rot(theta, openfile2[i-59+x*10].split('\t'))
                pastP = openfile1[i].split('\t')
                while(3*t+2 < len(pastP)):
                    database += rotY(theta_p,[float(pastP[3*t])-float(pastP[0]),pastP[3*t+1],float(pastP[3*t+2])-float(pastP[2])])
                    t += 1
                t = 0
                pastV = openfile3[i-1].split('\t')
                while(3*t+2 < len(pastV)):
                    database += rotY(theta_p,[pastV[3*t],pastV[3*t+1],pastV[3*t+2]])
                    t += 1
                database += angle(theta_p,openfile6[i])
                openfile3_tem = openfile3[i].split('\t')
                database += rot(theta_p,[openfile3_tem[0],openfile3_tem[2]])
                database += str((theta_p-theta_pp)*180./np.pi) + '\t'
                database += str(float(openfile4[i])-float(openfile4[i-1])) + '\t'
                database += openfile5[i]
                write_data(database,out)
                database = ""
            i += 1



def write_data(data, out):
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
                out =  path + "\\" + filename[:-4] + ".output"
                work(stri,out)

search()                