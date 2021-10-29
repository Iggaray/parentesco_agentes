import random as rd
import pyvis


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
            headers=next(rows)
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
 
def create_F1(M, f_n):
    '''
    Creates a population of M couples. Each copules has n children with probability f_n. 
    Returns a dictionary containing the list of children per couple and a dictionary containing 
    the gender of each child (assigned at random).
    '''

    #M is the number of couples that we want
    #f_n is the probability distribution that we want those couples to have

    acum=acumulate_prob_dict(f_n)

    #parents couples to children list contains the information of   couple_ID: [child_ID, child_ID]
    pc2cl={}

    #gender dictionary contains the information of gender for child population inf the form    child_ID: gender
    # 'm' holds for males and 'f' for females
    gender={}

    acum_list_values=[acum[k] for k in list(acum.keys())]  
    acum_list_keys=[int(k) for k in list(acum.keys())]
    
    child_ID_counter=1


    for i in range(M):
        # generate a random number 'r' between 0 and 1
        r=rd.random()       
        
        # find out where 'r' is in the accumulate curve, and that gives us the number 'n' of children added to couple i
        k=binary_search_list(acum_list_values, r)
        if k==len(acum_list_keys):
            k-=1
        n=acum_list_keys[k]
        #print(n)
        
        # add 'n' children to cuple i
        if n:
            for aux in range(n):
                try:
                    pc2cl[i].append(child_ID_counter)
                except:
                    pc2cl[i]=[child_ID_counter]
                child_ID_counter+=1
        elif n==0:
            pc2cl[i]=[]
    
    # asign random gender to those n children and write it in gender dictionary
    m=0
    for c in range(1,child_ID_counter+1):
        r=rd.random()
        if r<0.5:
            gender[c]='m'
            m+=1
        else:
            gender[c]='f'
    

    
    return pc2cl, gender


def marriages_F1(p2c, gender):
    '''
    Given a dict parent[i]:[child1, child2, ...] and a gender[i]: male/female dict, assings marriage at
    random in population of children, preventing sibling marriages and polygamous couples.

    returns a dict  person: parents, a dict person: spouse, list of people getting married

    '''

    #make a list of all people in F1 and reverse the p2c  (make a dictionary of child: parent_couple)
    F1_population=[]
    c2p={}
    for k in p2c.keys():
        F1_population+=p2c[k]
        for e in p2c[k]:
            c2p[e]=k
    
    #empty list for engaged people
    engaged=[]
    spouse={}

    
    count=0
    while(len(engaged)<len(F1_population) and count<10*len(F1_population)):
        ok=0
        count=0
        
        while(ok==0):
            #pick 2 different individuals x and y from the F1 population
            x, y=rd.choices(F1_population, k=2)
            count+=1
            #condition: y must not be of the same gender, and not a blood sibling
            if gender[x]!=gender[y] and c2p[x]!=c2p[y] and x not in engaged and y not in engaged:
                spouse[x]=y
                spouse[y]=x
                #add x and y to the list of engaded people
                engaged.append(x)
                engaged.append(y)
                ok=1
                
            if count>=10*len(F1_population):
                break

    return c2p, spouse, engaged      






def generate_population(M, a, start_zero=True):
    
    '''
    Creates a population descendant of M couples of parents.
    
    The f(n) function, namely the probability that a couple of parents has 'n' children, is exponential.
    Gender is asigned fully at random to each person (male and female only).
    Marriages are assigned fully random between nodes in population of children if:
        -the nodes are not blood siblings (brother-sister)
        -the nodes has different gender (heterosexual couples mating)
        -both nodes are single (monogamous couples)
    
    Input: 
            -number of couples of parents M
            -'a' to compute f(n)=A*a^(-n)
            -start_zero=True means that n starts in zero or one (false), this changes A
    
    Output:
            Returns a list of tuples [(node1, node2), (node1, node7), ...] representing the edges of a kinship network on population.
            
            Links are builded by chossing:
                -in blood relations (brothers and sisters of a node)
                -marriage relation (spouse of the node)

    '''
    
    ##################Definition of the f(n) distribution###########################
    f_n={}

    if start_zero:
        for i in range(0,60):
            f_n[i]=(1-1/a)*a**(-i)
    elif not start_zero:
        for i in range(1,60):
            f_n[i]=(a-1)*a**(-i)
    

    
    ########################Create population##########################################
    
    #Creating the F1 generation

    p2c, gender=create_F1(M, f_n)


    #Assignation of random marriages

    c2p, spouses_dict, popul_F1=marriages_F1(p2c, gender)

    
    ######################Build graph (list of tuples)#################
    T=[]
    

    #Add blood siblings edges
    for e in list(p2c.keys()):
        L=p2c[e]
        for i in range(0, len(L)-1):
            for j in range(i+1, len(L)):
                if L[i] in popul_F1 and L[j] in popul_F1:
                    
                    T.append((L[i], L[j]))
                   
        
    #Add marriage links
    gdr=gender[popul_F1[0]]
    for p in popul_F1:
        if gender[p]==gdr:
            
            T.append((p, spouses_dict[p]))
    
    del gdr, spouses_dict, popul_F1, p2c, c2p, gender, f_n


    

    return T

class Kinship_net(object):
    """Clase que representa un grafo
    
    Permite instanciar grafos a partir de una lista de tuplas. Se pueden
    generar representaciones como diccionario de nodos y sus respectivos
    vecinos. También es posible graficar la red.
    
    Parámetros
    ----------
    M : número inicial de parejas
    a : parámetro de la probabilidad de n hijos por pareja: f(n)=A*a^(-n)
    
    Atributos
    ---------
    lista : list
        lista de tuplas con aristas entre nodos [(n1, n2), ..., (ni, nj)]
    nodos : set
        conjunto de nodos de la red
    net : pyvis.network.Network
        objeto que representa la red
    
    Métodos
    -------
    __init__ : constructor que genera la red a partir de M y a
    generar_dic_vecinos : método que devuelve un diccionario donde cada key es
        un nodo y tiene una lista de vecinos asociada
    show : método para graficar la red
    """
    def __init__(self, M, a):
        """Construir una red de parentesco con parámetros M y a
        
        Input
        -----
        M: numero de parejas inicial
        a: parámetro de la probabilidad de n hijos por pareja: f(n)=A*a^(-n)
        """       
        #construyo la red
        self.lista = generate_population(M, a)
        self.M = M
        self.a = a
        
        #genero un objeto Network con la lista de tuplas de nodos
        self.nodos = set()
        for tupla in self.lista:
            self.nodos.add(tupla[0])
            self.nodos.add(tupla[1])
        
        for i in range(max(self.nodos)): #completo los nodos sin edge
            self.nodos.add(i)
            
        self.net = pyvis.network.Network()
        for nodo in self.nodos:
            self.net.add_node(nodo)
        for edge in self.lista:
            self.net.add_edge(*edge)
            
    def generar_dic_vecinos(self):
        """Devolver un diccionario de nodos y listas de vecinos asociados
        """ 
        grafo = []
        #completo los pares recíprocos y quito la unidad para que arranque en 0
        for tupla in self.lista[:len(self.lista)]:
            x, y = tupla
            grafo.append((x - 1, y - 1))
            grafo.append((y - 1, x - 1))
        
        #ordeno la lista de nodos
        grafo.sort()     
        #genero el diccionario de vecinos para cada nodo
        dic_vecinos = {}
        for tupla in grafo:
            i, j = tupla
            if i not in dic_vecinos:
                dic_vecinos[i] = [j]
            else:
                dic_vecinos[i].append(j)
        
        for nodo in self.nodos: #los nodos estan referidos a si mismos
            if nodo not in dic_vecinos:
                dic_vecinos[nodo] = [nodo]
            else:
                dic_vecinos[nodo].append(nodo)
        
        self.dic_vecinos = dic_vecinos
        return dic_vecinos
    
    def show(self):
        """Mostrar la red
        """
        #pendiente: estaria buenisimo que grafique el tamaño de los nodos en
        #función del promedio temporal de recursos de cada uno. computar eso 
        #igual tomaria una banda de tiempo. GPU con cupy?
        self.net.show('my_net.html')