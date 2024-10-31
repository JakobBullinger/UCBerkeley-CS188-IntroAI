import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        weights = self.get_weights()
        return nn.DotProduct(x, weights)



    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dot_product = self.run(x)

        # if nn.as_scalar(dot_product) >= 0: 
        #     return 1
        # else: 
        #     return -1

        return 1 if nn.as_scalar(dot_product) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        weights = self.get_weights()
        batchSize = 1
        # learningRate = 1

        while True:
            numberMisclassified = 0
            for x,y in dataset.iterate_once(batchSize):
                yprediction = self.get_prediction(x)
                yValue = nn.as_scalar(y)
                if yprediction != yValue:
                    numberMisclassified += 1
                    weights.update(x, yValue)
        
            if numberMisclassified == 0:
                break
    


        # Update weights: weights = weights + x * y
        # y= true label


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    # Hidden layer size: 512
    # Batch Size: 200
    # Learning Rate: 0.05
    # One hidden layer
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.HiddenLayerSize = 550
        self.BatchSize = 200
        self.learningRate = 0.03
        self.inputSize = 1  #input vairbale is a single dimensional real-world number
        self.outputSize = 1  #output vairbale is a single dimensional real-world number

        # define weights and bias of layers
        self.HiddenLayerWeights = nn.Parameter(self.inputSize, self.HiddenLayerSize)
        self.HiddenLayerBias = nn.Parameter(1, self.HiddenLayerSize)
        self.OutputLayerWeights = nn.Parameter(self.HiddenLayerSize, self.outputSize)
        self.OutputLayerBias = nn.Parameter(1, self.outputSize)

    def run(self, x):  #forward pass
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #linear transformation
        hiddenLayer = nn.Linear(x, self.HiddenLayerWeights)
        hiddenLayerComplete = nn.AddBias(hiddenLayer, self.HiddenLayerBias)
        
        #activation
        activation_relu = nn.ReLU(hiddenLayerComplete)

        #second linear tranformation from hidden layer to output layer
        outputLayer = nn.Linear(activation_relu, self.OutputLayerWeights)
        outputLayerComplete = nn.AddBias(outputLayer, self.OutputLayerBias)

        return outputLayerComplete

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        LossScalar = 1

        while LossScalar > 0.02:
            for x,y in dataset.iterate_once(self.BatchSize):
                Loss = self.get_loss(x,y)
                LossScalar = nn.as_scalar(Loss)
                gradient = nn.gradients(Loss, [self.HiddenLayerWeights, self.HiddenLayerBias, self.OutputLayerWeights, self.OutputLayerBias])          

                #updates
                self.HiddenLayerWeights.update(gradient[0], -self.learningRate)
                self.HiddenLayerBias.update(gradient[1], -self.learningRate)
                self.OutputLayerWeights.update(gradient[2], -self.learningRate)
                self.OutputLayerBias.update(gradient[3], -self.learningRate)
        return

     
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hiddenLayerSize = 200
        self.batchSize = 100
        self.learningRate = 0.5
        self.inputSize = 784  
        self.outputSize = 10  

        # define weights and bias of layers
        self.hiddenLayerWeights = nn.Parameter(self.inputSize, self.hiddenLayerSize)
        self.hiddenLayerBias = nn.Parameter(1, self.hiddenLayerSize)
        self.outputLayerWeights = nn.Parameter(self.hiddenLayerSize, self.outputSize)
        self.outputLayerBias = nn.Parameter(1, self.outputSize)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #linear transformation
        hiddenLayer = nn.Linear(x, self.hiddenLayerWeights)
        hiddenLayerComplete = nn.AddBias(hiddenLayer, self.hiddenLayerBias)
        
        #activation
        activation_relu = nn.ReLU(hiddenLayerComplete)

        #second linear tranformation from hidden layer to output layer
        outputLayer = nn.Linear(activation_relu, self.outputLayerWeights)
        outputLayerComplete = nn.AddBias(outputLayer, self.outputLayerBias)

        return outputLayerComplete
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)
    
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        validation_accuracy = 0.00

           
        while validation_accuracy < 0.98:
            for x,y in dataset.iterate_once(self.batchSize): 
                Loss = self.get_loss(x,y)
                gradient = nn.gradients(Loss, [self.hiddenLayerWeights, self.hiddenLayerBias, self.outputLayerWeights, self.outputLayerBias])          

                #updates
                self.hiddenLayerWeights.update(gradient[0], -self.learningRate)
                self.hiddenLayerBias.update(gradient[1], -self.learningRate)
                self.outputLayerWeights.update(gradient[2], -self.learningRate)
                self.outputLayerBias.update(gradient[3], -self.learningRate)
            validation_accuracy = dataset.get_validation_accuracy()
        return
    
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hiddenLayerSize = 200
        self.learningRate = 0.05
        self.batchSize = 100
        self.inputSize = self.num_chars
        self.outputSize = len(self.languages)

        #weights & biases
        self.inputLayerWeights = nn.Parameter(self.inputSize, self.hiddenLayerSize)
        self.inputLayerBias = nn.Parameter(1, self.hiddenLayerSize)        
        self.hiddenLayerWeights = nn.Parameter(self.hiddenLayerSize, self.hiddenLayerSize)
        self.hiddenLayerBias = nn.Parameter(1, self.hiddenLayerSize)
        self.outputLayerWeights = nn.Parameter(self.hiddenLayerSize, self.outputSize)
        self.outputLayerBias = nn.Parameter(1, self.outputSize)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # #linear transformation
        # hiddenLayer = nn.Linear(x, self.hiddenLayerWeights)
        # hiddenLayerComplete = nn.AddBias(hiddenLayer, self.hiddenLayerBias)
        
        # #activation
        # activation_relu = nn.ReLU(hiddenLayerComplete)

        # #second linear tranformation from hidden layer to output layer
        # outputLayer = nn.Linear(activation_relu, self.outputLayerWeights)
        # outputLayerComplete = nn.AddBias(outputLayer, self.outputLayerBias)


        hiddenLayer = nn.Linear(xs[0], self.inputLayerWeights)
        hiddenLayer = nn.AddBias(hiddenLayer, self.inputLayerBias)
        hiddenLayer = nn.ReLU(hiddenLayer)
        for x in xs[1:]:
            hiddenLayer = nn.Add(nn.Linear(x, self.inputLayerWeights), nn.Linear(hiddenLayer, self.hiddenLayerWeights))
            hiddenLayer = nn.AddBias(hiddenLayer, self.hiddenLayerBias)
            hiddenLayer = nn.ReLU(hiddenLayer)
        outputLayer = nn.Linear(hiddenLayer, self.outputLayerWeights)
        outputLayer = nn.AddBias(outputLayer, self.outputLayerBias)
        return outputLayer

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)
    
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        validation_accuracy = 0

        while validation_accuracy < 0.89:
            for x,y in dataset.iterate_once(self.batchSize): 
                Loss = self.get_loss(x,y)
                gradient = nn.gradients(Loss, [self.inputLayerWeights, self.inputLayerBias, self.hiddenLayerWeights, self.hiddenLayerBias, self.outputLayerWeights, self.outputLayerBias])          

                #updates
                self.inputLayerWeights.update(gradient[0], -self.learningRate)
                self.inputLayerBias.update(gradient[1], -self.learningRate)
                self.hiddenLayerWeights.update(gradient[2], -self.learningRate)
                self.hiddenLayerBias.update(gradient[3], -self.learningRate)
                self.outputLayerWeights.update(gradient[4], -self.learningRate)
                self.outputLayerBias.update(gradient[5], -self.learningRate)
            validation_accuracy = dataset.get_validation_accuracy()