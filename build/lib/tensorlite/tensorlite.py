import numpy as np

"""
Welcome, to tensorlite . This project is made to make tensorflow simple. Here , we will be discussing about the different aspects of tensorflow. This project can be considered as a subset of tensorflow , to help everyone understand tensorflow and its concepts . 

We have some basic concepts in Tensorflow. 

1) Tensors - N-Dimentional Array, much like the ndArray of numpy

2) Operation - Any operation that is to be performed i.e. addition , multiplication, matrix-mltiplication

3) Graph - A global variable , which connects all these variables and placeholders in to a specific operation.

4) Variables - A variable or a changeble parameter of a graph. For example, weights of a neural network.

5) Placeholders - a placeholder for any unknown value. It can be also thought of as an empty node , which needs a value to compute output.

so, lets start by making a graph.

"""

class Graph():
    
    def __init__(self):
        
        self.operations = []
        self.placeholders = []
        self.variables = []
        
    def set_as_default(self):
        global _default_graph
        _default_graph = self

"""
What we did, here , is that we constructed a graph class , which is nothing but a global variable which will connect all the variables , placeholders and operations together. Also, since we will be having multiple graphs , we will need to know , which operations and variables belong to which graph. Thus, we made a method named set_as_default, which makes a *_default_graph* global variable, and sets it to the reference of the graph.

for example,
if 

y = Mx + C ( which is standard equation of a straight line ) , we can classify this.

M = Variable & C = Variable
x = Placeholder

M(x) = Multiplication of M with x = Multiplication Operation
M(x) + c = Addition of M(x) with C = Addition Operation

now , why are M and C called variables , is because , these are the values that are being calculated. For example, in a simple regression problem, __y__ is the dependent variable which we have to predict. __x__ is the indepent variable or features, based on which the values of __y__ will be predicted. This is why __x__ is a placeholder since , we will feed the equation with __x__ to give the result. But , since M and C are to be calculated i.e. slope and intercept every time the the equation has to give a result, we define them as a variable.

In the case of a neural network, the weights of a neural network are the variables since , after each iteration the weight changes .

Now, all these things happen differently , thus we need some sort of a pipeline to connect all these things into one equation , thus we need a graph.
"""

class Placeholder():
    
    def __init__(self):
        
        self.output_nodes = []
        
        #append the instance of the placeholder, to a global variable (graph) to be later get connected to a final operation
        _default_graph.placeholders.append(self)

class Variable():
    
    def __init__(self, initial_values=None): # we are having some initial values, such as random weights for a neural network
        
        self.values = initial_values
        self.output_nodes = []
        
        #append the instance of the Variable, to a global variable (graph) to be later get connected to a final operation
        _default_graph.variables.append(self)

"""
Now comes the operation class , which is nothing but basic simple mathematical operations. Basically , an operation is based on 4 things - 

1) Input Nodes - The inputs passed onto the operation

2) Output Nodes - The ouput of the operation

3) Graph - A global variable , since it will be connecting the operations to other aspects of the equation (Placeholders and Variables)

4) Compute Method - Whihc will actually perform the mathematical calculation

"""

class Operation():
    
    def __init__(self, input_nodes = []):
        
        self.input_nodes = input_nodes
        self.output_nodes = []
        
        for node in input_nodes:
            # for each node in input node, assign itself to the correct output node.
            node.output_nodes.append(self) 
                      
    def compute(self):
        pass

class add(Operation):
    
    def __init__(self, x, y):
        # we initialize the Main Operation class by the calling the Operation method and then by passing the list of input nodes
        # x and y
        super().__init__([x,y])
        
    def compute(self, x_variable, y_variable):
        self.inputs = [x_variable, y_variable]
        
        return x_variable + y_variable

class multiply(Operation):
    
    def __init__(self, x, y):
        # we initialize the Main Operation class by the calling the Operation method and then by passing the list of input nodes
        # x and y
        super().__init__([x,y])
        
    def compute(self, x_variable, y_variable):
        self.inputs = [x_variable, y_variable]
        
        return x_variable * y_variable

class matmul(Operation):
    
    def __init__(self, x, y):
        # we initialize the Main Operation class by the calling the Operation method and then by passing the list of input nodes
        # x and y
        super().__init__([x,y])
        
    def compute(self, x_variable, y_variable):
        self.inputs = [x_variable, y_variable]
        
        return x_variable.dot(y_variable)

"""
Now for the equation , we know which element corresponds to what in our version of tensorflow, so what we can do is try to write the equation , based on what we kwow.So,

y = M(x) + C

=> let us assume M = 10 and C = 4

=> y = 10x + 4 , where 10 & 4 are variables and __x__ is a placeholder.
"""

def traverse_postorder(operation):
    ## operation argument is the argument for the function, and not the Operation
    ## class . Just to be clear.
    
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
        
    recurse(operation)
    return nodes_postorder

class Session():
    
    def run(self, operation, feed_dict={}):
        
        nodes_postorder = traverse_postorder(operation)
        
        for node in nodes_postorder:
            
            if type(node) == Placeholder:
                
                node.output = feed_dict[node]
                
            elif type(node) == Variable:
                
                node.output = node.values
                
            else:
                
                #Operations
                node.inputs = [input_node.output for input_node in node.input_nodes]
                
                node.output = node.compute(*node.inputs)
                
            if type(node.output) == list:
                node.output = np.array(node.output)
                
        return operation.output

class sigmoid(Operation):
    
    def __init__(self, z):
        super().__init__([z])
        
    def compute(self, z_val):
        return 1 / (1 + np.exp(-z_val))

