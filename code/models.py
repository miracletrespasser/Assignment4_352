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
        self.batch_size = 1

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
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        scalar = nn.as_scalar(self.run(x))
        if scalar >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        error_loop = 1
        # terminate when error_loop = 0
        while error_loop:
            error_loop = 0
            # iterate throught the data and make the prediction
            for x, y in dataset.iterate_once(self.batch_size):
                label = nn.as_scalar(y)
                # if the prediction is wrong loop again, and update parameter
                if self.get_prediction(x) != label:
                    error_loop = 1
                    parameter = self.get_weights()
                    parameter.update(x, label)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    '''
    We tried for 2 layers and 3 layers, it turns out that 3 layers would allow the computation to 
    go faster but can create ambiguity in some test cases, therefore we decide to do a 2 layer demonstration 
    '''
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(1, 200)
        self.b1 = nn.Parameter(1, 200)
        self.w2 = nn.Parameter(200, 1)
        self.b2 = nn.Parameter(1, 1)
        #we choose -0.01 because the final graph it produces is more smoother compare to other values
        self.learningrate=-0.01
        self.batch_size=1

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #initilize the variables of f(x) = relu(x * W1 + b1) * W2 + b2
        x1 = nn.Linear(x, self.w1)
        relu = nn.ReLU(nn.AddBias(x1, self.b1))
        x2 = nn.Linear(relu, self.w2)
        f_x= nn.AddBias(x2, self.b2)
        #return the result of f(x) = relu(x * W1 + b1) * W2 + b2
        return f_x

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
        pass_test= False
        while not pass_test:
            for x, y in dataset.iterate_once(self.batch_size):
                #calculate the gradients
                getloss=self.get_loss(x,y)
                gradients = nn.gradients(getloss, [self.w1, self.b1,self.w2, self.b2])
                #update the bias and weights
                self.w1.update(gradients[0], self.learningrate)
                self.b1.update(gradients[1], self.learningrate)
                self.w2.update(gradients[2], self.learningrate)
                self.b2.update(gradients[3], self.learningrate)

            #the test shall be passed with a loss less than 0.02
            pass_test= (nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02)
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
        self.w1 = nn.Parameter(784, 200)
        self.b1 = nn.Parameter(1, 200)
        self.w2 = nn.Parameter(200, 10)
        self.b2 = nn.Parameter(1, 10)
        self.learning_rate = -0.1
        self.batch_size = 100


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
        # LAYER 1
        # L1 = ReLU(x * W1 + b1)
        x1 = nn.Linear(x, self.w1)
        L1 = nn.ReLU(nn.AddBias(x1, self.b1))
        # ---------------------------------------
        # LAYER 2
        # L2 = L1 * W2 + b2
        x2 = nn.Linear(L1, self.w2)
        L2 = nn.AddBias(x2, self.b2)
        # ---------------------------------------
        # FINAL OUTPUT
        # f(x) = ReLU(x * W1 + b1) * W2 + b2
        return L2

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
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        passed = False
        while not passed:
            for x, y in dataset.iterate_once(self.batch_size):
                # run model, compute loss and gradients
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])

                # update the model using gradients
                self.w1.update(gradients[0], self.learning_rate)
                self.b1.update(gradients[1], self.learning_rate)
                self.w2.update(gradients[2], self.learning_rate)
                self.b2.update(gradients[3], self.learning_rate)

            # test passes when validation accuracy of 97.75%
            # higher validation threshold so testing accuracy will be higher than 97%
            passed = dataset.get_validation_accuracy() > 0.9775

