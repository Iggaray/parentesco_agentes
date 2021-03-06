from pyvis.network import Network
import random as rd
import numpy as np
import networkx as nx

def load_distr(filename, has_headers=True):
    '''
    Function that uploads the probability distribution and returns a dictionary with it. 
    Archive containing the information must have the structure:

    n       probability
    (int)   (float)
    ....    ......

    Output is in the form:
    distr[n]=p(n) (given n as key, returns the probability of having it)
    '''
    distr={}
    types=[int, float]
    with open(filename, 'rt') as rows:   
        if has_headers:
            next(rows)
            for line in rows:
                line=line.split('\t')
                n_p=['', '']
                try:
                    n_p=[func(val) for func, val in zip(types, line)]
                except:
                    n_p[0]=types[0](float(line[0]))
                    n_p[1]=types[1](float(line[1][:-1]))
                distr[n_p[0]]=n_p[1]
        else:
            for line in rows:
                
                line=line.split('\t')
                #print(line)
                n_p=['', '']
                try:
                    n_p=[func(val) for func, val in zip(types, line)]
                except:
                    #print(line)
                    n_p[0]=types[0](float(line[0]))
                    n_p[1]=types[1](float(line[1][:-1]))
                    #print(n_p)
                distr[n_p[0]]=n_p[1]
               
    return distr

def acumulate_prob_dict(given_dict):
    '''
    Returns a dictionary with the cummulative probability distribution of a given PDF d[n]=f_n:
    A[n]=sum of all d[i] with i<=n
    n must be type integer
    probabilities must be type float

    '''
    A={}
    suma=0
    for n in given_dict.keys():
        suma+=given_dict[n]
        A[n]=suma
    return A

def linear_search_list(L, x):
    '''
    Given a sorted list L, returns the position where the element x is using a linear search. 
    If x is not L, returns the position where it must be inserted.
    '''
    
    for i,z in enumerate(L):
        if x<=z:
            return i 
        
def binary_search_list(L, x):
    pos=-1
    left=0
    right=len(L)-1
    while left<=right:
        middle=(left+right)//2
        if L[middle]==x:
            pos=middle
            break
        
        if L[middle]>x:
            right=middle-1
        else:
            left=middle+1
    
    if not (left<=right):
        pos=left
    
    return pos
 
def create_F1(N, f_n, acum):
    '''
    Creates a population of M couples. Each copules has n children with probability f_n. 
    Returns a dictionary containing the list of children per couple and a dictionary containing 
    the gender of each child (assigned at random).
    '''

    #M is the number of couples that we want
    #f_n is the probability distribution that we want those couples to have

    

    #parents couples to children list contains the information of   couple_ID: [child_ID, child_ID]
    pc2cl={}

    #c2p[i]: m_i   indicates at which parent couple is linked the chidl i
    c2p={}



    acum_list_values=[acum[k] for k in list(acum.keys())]  
    acum_list_keys=[int(k) for k in list(acum.keys())]
    
    child_ID_counter=0

    N_dyn=0
    M_dyn=0

    while N_dyn<2*(N):
        # generate a random number 'r' between 0 and 1
        r=rd.random()       
        
        # find out where 'r' is in the accumulate curve, and that gives us the number 'n' of children added to couple i
        k=binary_search_list(acum_list_values, r)
        
        while k==len(acum_list_keys):
            r=rd.random()
            k=binary_search_list(acum_list_values, r)
        
        n=acum_list_keys[k]
        N_dyn+=n
        
        if N_dyn>2*(N):
            n-=N_dyn-2*(N)
            N_dyn=2*N
        

        if n:
            for aux in range(n):
                try:
                    pc2cl[M_dyn].append(child_ID_counter)
                    
                except:
                    pc2cl[M_dyn]=[child_ID_counter]

                c2p[child_ID_counter]=M_dyn
                child_ID_counter+=1
        
        elif n==0:
            pc2cl[M_dyn]=[]

        M_dyn+=1
    #print('Ndyn=', N_dyn)
    #print('M_dyn=', M_dyn)
    #print('max_ID=', child_ID_counter)
    

    
    return pc2cl, c2p, M_dyn


def marriages_F1(N, p2c, c2p):
    # make a list of all people in F1 and reverse the p2c
    # (make a dictionary of child: parent_couple)
    F1_population=[i for i in range(2*N)]

    #print('Length of population used for marriages=', len(F1_population))
    
    #empty list for engaged people
    #engaged=[]
    spouse={}
    gender={}

    
    count=0
    sucess=1
    while(len(F1_population)!=0 and count<10*N*2):
        ok=0
        count=0
        
        while(ok==0):
            #pick 2 different individuals x and y from the F1 population
            x, y=rd.choices(F1_population, k=2)
            count+=1
            #condition: not a blood sibling
            if c2p[x]!=c2p[y]:
                spouse[x]=y
                spouse[y]=x
                gender[x]='m'
                gender[y]='f'
                #add x and y to the list of engaded people
                F1_population.pop(binary_search_list(F1_population, x))
                F1_population.pop(binary_search_list(F1_population, y))
                #engaged.append(x)
                #engaged.append(y)
                ok=1
                
            if count>=10*N*2:
                sucess=0
                break

    del  count, ok

    return sucess, spouse, gender     

def generate_network(N, f_n, acum):
    
    '''
    Creates a kinship network of a population with M men and M women. 
    
    Parents couples are dynamically created according to the given f_n function and his cummulative PDF.

    Marriages within the populaton are assigned fully random between nodes in population of children if:
        -the nodes are not blood siblings (brother-sister)
        -the nodes has different gender (heterosexual couples mating)
        -both nodes are single (monogamous couples)
    
    Input: 
            -number of people in children population N
            - f(n), namely probability that a couple of parents has 'n' children
            -acum   is the Cummulative PDF of f(n)
    
    Output:
            Returns the graph of the population
            
    '''  
    
    #Creating the F1 generation

    p2c, c2p, M=create_F1(N, f_n, acum)

    del M


    #Assignation of random marriages

    x=0

    sucess, spouses_dict, gender=marriages_F1(N, p2c, c2p)

    #print(len(spouses_dict.keys()))

    while not sucess:
        x+=1
        p2c, c2p, M=create_F1(N, f_n, acum)
        del M
        sucess, spouses_dict, gender=marriages_F1(N,p2c, c2p)
        #print('Hizo falta volver a asignar matrimonios')
    
    del x


    
    ######################Build graph#################
    G=nx.Graph()
    #print(popul_F1)
    G.add_nodes_from([i for i in range(2*N)])      #Add nodes
    

    #Add blood siblings edges
    for e in list(p2c.keys()):
        L=p2c[e]
        
        for i in range(0, len(L)-1):
            for j in range(i+1, len(L)):
                G.add_edge(L[i], L[j])
  
        
    #Add marriage links
    #print(gender)
    gdr='m'
    for p in range(2*N):
        #print(p, 'gender[p]==gdr?', gender[p]==gdr)
        if gender[p]==gdr:
            G.add_edge(p, spouses_dict[p])
            
     #Add in law siblings
    
    
    
    
    
    
    #print(neigh)



            

    return G



class Kinship_net(object):
    """Clase que representa un grafo de parentesco
    
    Se pueden generar representaciones como diccionario de nodos y sus
    respectivos vecinos. Tambi??n es posible graficar la red.
    
    Par??metros
    ----------
    N_total : n??mero de agentes en la poblaci??n
    n_mean : n??mero medio de hijos por pareja de padres
    
    Atributos    ####  Update this
    ---------
    grafo : objeto Graph de la libreria networkx
        
    N : int
        n??mero de nodos de la red
    n_max : int
        m??ximo n??mero permitido de hijes por pareja
    
    M??todos
    -------
    __init__ : constructor que genera la red de N_total agentes
    
    generar_dic_vecinos : m??todo que devuelve un diccionario donde cada key es
        un nodo y tiene una lista de vecinos asociada
        
    show : m??todo para graficar la red
    """
    def __init__(self, N_total, n_mean):
        """Construir una red de parentesco con par??metros N y n_mean=2*alpha
        
        Input
        -----
        N_tot: numero total de agentes
        n_mean: numero medio de hijos por pareja de padres
        """       
        
        
        self.N = N_total // 2           # En el modelo se trabaja con N personas de cada genero, 
                                        #haciendo 2N personas en total
        self.alpha = n_mean / 2
        
        self.n_max = 25                 #Esto es un top que ponemos para que sea m??s realista el modelo
                                        #Parejas con m??s de 25 hijos son imposibles
        self.nombre = f"Red_N{N_total:d}_nmean{n_mean:d}"
        
        #########Build f_n  and acum  ####################
        
        a=1+1/(2*self.alpha)
        f_n={i: (1-1/a)*a**(-i) for i in range(0,self.n_max)}
        acum=acumulate_prob_dict(f_n)
        
        #######################
        
        #construyo la red
        self.grafo = generate_network(self.N, f_n, acum)
        
       
            
    def generar_dic_vecinos(self):
        """Devolver un diccionario de nodos y listas de vecinos asociados
        """ 
        
        neigh={i: [v for v in self.grafo.neighbors(i)] for i in range(2*self.N)}
        
        return neigh

    def segregate_per_neighbours(self, k_list):
        '''
        Devuelve un diccionario de la forma:

        k : [agente1, agente2, ...]

        O sea, agrupa una lista de agentes que tienen k vecinos

        1 <= k <= 26

        '''
        k_agents = {i:[] for i in k_list}
        node_k = self.generar_dic_vecinos()
        for j in node_k.keys():
            L = len(node_k[j])
            if L in k_agents:
                k_agents[L].append(j)
            
        #paso cada lista a np.array int32
        for key in k_agents.keys():
            k_agents[key] = np.array(k_agents[key], dtype=np.int32)
        return k_agents
    
    def show(self, resources={}):
        """
        Shows a visual representation of the kinship network. 
        Node sizes are proportional to agent resource (averaged during the stationary observation time). 
        If no specific information on resource is given, then node sizes are default ones.

        Input:
            -Dictionary  "resources"   containing the resources of each agent. (default value is empty list)
        """
        
        #Initialization
        nt = Network('500px', '500px')
        

        
        
        
        if resources is {}:
            
            nt.from_nx(self.grafo)
            
            nt.show(self.nombre + '.html')
        
        else:
            nodes=list(self.grafo.nodes)
            edges = list(self.grafo.edges)
            nt.add_nodes(nodes, value=[resources[i] for i in nodes])
            for element in edges:
                nt.add_edge(element[0], element[1])
            
            nt.show(self.nombre + '.html')


