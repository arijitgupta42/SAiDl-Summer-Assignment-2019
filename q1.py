import numpy

def tanh(x):
    return (1.0 - numpy.exp(-2*x))/(1.0 + numpy.exp(-2*x))

def tanh_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))

class NeuralNetwork:
   
    def __init__(self, net_arch):
        numpy.random.seed(0)
        
        self.activity = tanh
        self.activity_derivative = tanh_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 1
        self.arch = net_arch
        self.weights = []
        
           
        for layer in range(self.layers - 1):
            w = 2*numpy.random.rand(net_arch[layer] + 1, net_arch[layer+1]) - 1
            self.weights.append(w)

            
    def _forward_prop(self, x):
        y = x

        for i in range(len(self.weights)-1):
           
            activation = numpy.dot(y[i], self.weights[i])
            activity = self.activity(activation)

       
            activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
            y.append(activity)

       
        activation = numpy.dot(y[-1], self.weights[-1])
        activity = self.activity(activation)
        y.append(activity)
        
        return y
    
    def _back_prop(self, y, target, learning_rate):
        error = target - y[-1]
        delta_vec = [error * self.activity_derivative(y[-1])]

       
        for i in range(self.layers-2, 0, -1):
            error = delta_vec[-1].dot(self.weights[i][1:].T)
            error = error*self.activity_derivative(y[i][1:])
            delta_vec.append(error)

       
        delta_vec.reverse()
        
       
        for i in range(len(self.weights)):
            layer = y[i].reshape(1, self.arch[i]+1)
            delta = delta_vec[i].reshape(1, self.arch[i+1])
            self.weights[i] += learning_rate*layer.T.dot(delta)
    
    
    def fit(self, data, labels, learning_rate=0.1, epochs=100):
        
       
        ones = numpy.ones((1, data.shape[0]))
        Z = numpy.concatenate((ones.T, data), axis=1)
        
        for k in range(epochs):
            if (k+1) % 10000 == 0:
                print('epochs: {}'.format(k+1))
        
            sample = numpy.random.randint(X.shape[0])

       
            x = [Z[sample]]
            y = self._forward_prop(x)

       
            target = labels[sample]
            self._back_prop(y, target, learning_rate)
    
    def predict_data(self, x):
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activity(numpy.dot(val, self.weights[i]))
            val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))
        return [numpy.absolute(val[1]), numpy.absolute(val[2])]
    
    
    def predict(self, X):
        Y = numpy.array([]).reshape(0, self.arch[-1])
        for x in X:
            y = numpy.array([[self.predict_single_data(x)]])
            Y = numpy.vstack((Y,y))
        return Y
numpy.random.seed(0)

nn = NeuralNetwork([5,4,3,2])

X = numpy.array([[0,0,0,0,0], [0,0,1,0,0],[0,0,0,1,0], [0,0,1,1,0],
				 [0,1,0,0,0], [0,1,1,0,0],[0,1,0,1,0], [0,1,1,1,0],
				 [1,0,0,0,0], [1,0,1,0,0],[1,0,0,1,0], [1,0,1,1,0],
				 [1,1,0,0,0], [1,1,1,0,0],[1,1,0,1,0], [1,1,1,1,0],
				 [0,0,0,0,1], [0,0,1,0,1],[0,0,0,1,1], [0,0,1,1,1],
				 [0,1,0,0,1], [0,1,1,0,1],[0,1,0,1,1], [0,1,1,1,1],
				 [1,0,0,0,1], [1,0,1,0,1],[1,0,0,1,1], [1,0,1,1,1],
				 [1,1,0,0,1], [1,1,1,0,1],[1,1,0,1,1], [1,1,1,1,1]])

y = numpy.array([[0,0],[0,1],[1,0],[1,1],
				 [0,1],[0,1],[1,1],[1,0],
				 [1,0],[1,1],[0,0],[0,1],
				 [1,1],[1,0],[0,1],[0,0],
				 [1,1],[1,0],[0,1],[0,0],
				 [1,0],[1,0],[0,0],[0,1],
				 [0,1],[0,0],[1,1],[1,0],
				 [0,0],[0,1],[1,0],[1,1]])

nn.fit(X, y, epochs=475000)

print("Final predictions :")
accuracy=0
i=0
for s in X:
	accuracy+=100*(1-(numpy.absolute(y[i][0]-nn.predict_data(s)[0])+numpy.absolute(y[i][0]-nn.predict_data(s)[0]))/2)
	print(s,"Predicted ==>",nn.predict_data(s),"Accuracy =", 100*(1-(numpy.absolute(y[i][0]-nn.predict_data(s)[0])+numpy.absolute(y[i][0]-nn.predict_data(s)[0]))/2),'%')
	i = i + 1
print("\n\nNet Accuracy ==>", accuracy/(i))