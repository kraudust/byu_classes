import numpy as np
import numpy as np
import pandas
import matplotlib.pyplot as plt
import copy

class grad_desc_loss():
    def __init__(self,features,labels):
        self.x = features
        self.lab = labels
        self.num_features, self.num_samples = np.shape(self.x)
        self.num_classes = np.shape(self.lab)[1]
        self.lab_onehot = np.zeros((self.num_classes,self.num_samples))
        self.W = np.random.randn(self.num_classes, self.num_features)
        self.scores = np.zeros((self.num_classes,self.num_samples))

    def calc_loss_softmax(self):
        Lmat = np.zeros((self.num_classes,self.num_samples))
        self.scores = self.W*self.x
        b = np.max(self.scores,0) # used for the exp-normalize trick (see timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
        Lmat = -np.log(np.divide(np.exp(np.subtract(self.scores,b)),np.sum(np.exp(np.subtract(self.scores,b)),0))) #loss matrix
        self.Loss = np.sum(np.sum(self.lab_onehot*Lmat,0))

    def one_hot_encode(self):
        for i in range(self.num_samples):
            self.lab_onehot[self.l[0,i],i] = 1.0

if __name__ == '__main__':
    #Load CIFAR-10 Data Set--------------------------------
    def unpickle( file ):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict
    #------------------------------------------------------
    data = unpickle( 'cifar-10-batches-py/data_batch_1' )

    features = data['data'].T #size is #features x #samples
    labels = data['labels']
    labels = np.atleast_2d( labels ) #size is 1 x number of samples
