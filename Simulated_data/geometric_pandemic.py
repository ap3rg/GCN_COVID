
def prob(a, b):

   if a == b:
       return 0.999
   else:
       return 0.001
#####


import os
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Graph tooks
import graph_tool as gt
# from graph_tool.all import *
import graph_tool.dynamics as pand

# Local imports
import constants as cons

def create_graph(num_centers, num_individuals, sigma, mu, alpha):

    x1=[]
    y1=[]

    # Create centers
    centers=[]
    centy=[]
    for i in range(0, num_centers):
        centers.append(list(5*np.random.rand(1))[0])
        centy.append(list(5*np.random.rand(1))[0])

    clr=[]
    
    # Create individuals
    for j in range(len(centers)):
        
        theta =  math.pi*(np.random.rand(num_individuals))
        s = np.random.normal(mu, sigma, num_individuals)

        clr0=np.random.rand(1)
        for i in range(0, num_individuals):
            x1.append(math.cos(theta[i])*s[i] + centers[j])
            y1.append(math.sin(theta[i])*s[i] + centy[j])
            clr.append(j)

    G=nx.Graph()

    # Build graph 
    for i in range(0, len(x1)):
        G.add_node(i)

    Gcc=[]
    gcprev=0
    count=0
    
    while len(Gcc)!=1:
        #Check if size hasn't changed
        if len(Gcc)==gcprev:
            count+=1
        else:
            count=0
            gcprev=len(Gcc)
            
        if count > cons.MAX_ITER_GRAPHBUILD:
            break
        
        for j in range(0,1000):
            in1=random.choice(list(G.nodes()))
            in2=in1
            while (in2==in1):
                in2=random.choice(list(G.nodes()))
            d=math.sqrt( (x1[in1]-x1[in2])**2 + (y1[in1]-y1[in2])**2 )
            r=np.random.rand(1)
            if (math.exp(-d/alpha)>r):
                G.add_edge(in1,in2)
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)


    # Connect remainning .
    cont=0
    while len(Gcc)!=1:
        if cont%10000==0:
            print(cont)
            print(len(Gcc))
        cont+=1
        gcc1 = random.randrange(len(Gcc))
        gcc2 = random.randrange(len(Gcc))
        if gcc1!=gcc2:
            in1=random.sample(Gcc[gcc1],1)[0]
            in2=random.sample(Gcc[gcc2],1)[0]
            d=math.sqrt( (x1[in1]-x1[in2])**2 + (y1[in1]-y1[in2])**2 )
            r=np.random.rand(1)
            if (math.exp(-d/alpha)>r):
                G.add_edge(in1,in2)
                cont=0
                Gcc = sorted(nx.connected_components(G), key=len, reverse=True)        

    G2 = gt.Graph(directed=False)

    for i in range(0,len(x1)):
        G2.add_vertex()

    for i in G.edges():
        print(i[0],i[1])
        G2.add_edge(G2.vertex(i[0]),G2.vertex(i[1]))

    return centers, clr, G2


def simulate_disease_spread(centers, clr, G2, beta, out_path):

    # Initial disease state
    state = pand.SIState(G2, beta=beta)

    alpha={}
    plt.show(block=False)
    fl=open(out_path, "w")
    for j in range(len(centers)):
        print ("Cluster%02d"%(j),end="",file=fl)
        if j<len(centers)-1:
            print(";",end="",file=fl)
            
    print("",file=fl)

    X=[]                                        # Number of daily infections
    num_days = 1500                             # Number of days 
    for t in range(num_days):
        sac = state.get_state()                 # State of individuals
        ret = state.iterate_sync()
        X.append(state.get_state().fa.sum())    # Number of infected

        counts=[]
        for i in range(0,len(clr)):
            if sac[i]:
                counts.append(clr[i])
                if i in alpha.keys():
                    alpha[i]=min(alpha[i]+0.025,1)
                else:
                    alpha[i]=0.025

        for j in range(len(centers)):
            k=counts.count(j)
            if k:
                print ("%02d"%(k),end="",file=fl)
            else:
                print ("00",end="",file=fl)
            if j<len(centers)-1:
                print(";",end="",file=fl)
        print("",file=fl)
    fl.close()

def save_graph_edgelist(G):
    