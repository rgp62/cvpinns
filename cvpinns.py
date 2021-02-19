import numpy as np
import tensorflow as tf

class PDE:
    def __init__(self,L,nx,quad,F):
        self.F = F
        
        self.L = L
        self.nx = nx
        self.quad = quad


        xi0  = quad['0'] [0]
        xi1  = quad['1'] [0]
        xi01 = quad['01'][0]
        self.wi0  = quad['0'] [1]
        self.wi1  = quad['1'] [1]
        self.wi01 = quad['01'][1]


        x0 = np.linspace(L[0][0],L[0][1],nx[0]+1)
        x1 = np.linspace(L[1][0],L[1][1],nx[1]+1)

        x0c = (x0[0:-1]+x0[1:])/2
        x1c = (x1[0:-1]+x1[1:])/2
        x01c = np.transpose(np.meshgrid(x0c,x1c,indexing='ij'),(1,2,0))


        dx = x01c[1,1] - x01c[0,0]
        self.dx = dx
        
        self.x0  = tf.constant(np.expand_dims(np.transpose(np.meshgrid(x0,x1c,indexing='ij'),(1,2,0)),2) \
                     + np.stack([np.zeros(len(xi0)),dx[1]/2*xi0],1))
        self.x1  = tf.constant(np.expand_dims(np.transpose(np.meshgrid(x0c,x1,indexing='ij'),(1,2,0)),1) \
                     + np.expand_dims(np.stack([dx[0]/2*xi1,np.zeros(len(xi1))],1),1))
        self.x01 = tf.constant(np.expand_dims(x01c,[1,3]) + np.expand_dims(xi01*dx/2,1))
        
        self.x = x01c
       
        
        @tf.function        
        def F0(x0i):
            return tf.concat([
                        self.F['0L'](tf.expand_dims(x0i[0],0)),   
                        self.F['0I'](x0i[1:-1]),
                        self.F['0H'](tf.expand_dims(x0i[-1],0))
                    ],0)

        
            
        @tf.function
        def F1(x1i):
            return tf.concat([
                        self.F['1L'](tf.expand_dims(x1i[:,:,0],2)),
                        self.F['1I'](x1i[:,:,1:-1]),
                        self.F['1H'](tf.expand_dims(x1i[:,:,-1],2))
                    ],2)

        
        @tf.function
        def F01(x01i):
            return self.F['01'](x01i)
        
        self.F0 = F0
        self.F1 = F1
        self.F01 = F01


    
    @tf.function
    def getRES(self):
        F0int  = self.dx[1]/2*tf.tensordot(self.F0(self.x0),self.wi0,[2,0])
        F1int  = self.dx[0]/2*tf.tensordot(self.F1(self.x1),self.wi1,[1,0])
        F01int = self.dx[0]*self.dx[1]/4*tf.tensordot(self.F01(self.x01),self.wi01,[[1,3],[0,1]])
        return  (F0int[1::] - F0int[0:-1])  \
               +(F1int[:,1::] - F1int[:,0:-1])  \
               -F01int
 