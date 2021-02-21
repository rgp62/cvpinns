import numpy as np
import tensorflow as tf

class PDE:
    def __init__(self,L,nx,quad,F0,F1,IC,BCl,BCr,u):
        self.F0 = F0
        self.F1 = F1
        self.IC = IC
        self.BCl = BCl
        self.BCr = BCr
        self.u = u
        
        self.L = L
        self.nx = nx
        self.quad = quad


        xi0  = quad['0'] [0]
        xi1  = quad['1'] [0]

        self.wi0  = quad['0'] [1]
        self.wi1  = quad['1'] [1]



        x0 = np.linspace(L[0][0],L[0][1],nx[0]+1)
        x1 = np.linspace(L[1][0],L[1][1],nx[1]+1)

        x0c = (x0[0:-1]+x0[1:])/2
        x1c = (x1[0:-1]+x1[1:])/2
        xc = np.transpose(np.meshgrid(x0c,x1c,indexing='ij'),(1,2,0))


        dx = xc[1,1] - xc[0,0]
        self.dx = dx
        
        self.x0  = tf.constant(np.expand_dims(np.transpose(np.meshgrid(x0,x1c,indexing='ij'),(1,2,0)),2) \
                     + np.stack([np.zeros(len(xi0)),dx[1]/2*xi0],1))
        self.x1  = tf.constant(np.expand_dims(np.transpose(np.meshgrid(x0c,x1,indexing='ij'),(1,2,0)),1) \
                     + np.expand_dims(np.stack([dx[0]/2*xi1,np.zeros(len(xi1))],1),1))
        
        self.x = xc
        
        x0r = tf.reshape(self.x0,(nx[0]+1,nx[1]*len(xi0),2))

        jrmax = nx[1]*len(xi0)

        if xi1[0]<-1.+1e-10:
            jl = 1
        else:
            jl = 0
        if xi1[-1]>1-1e-10:
            jr = jrmax-1
        else:
            jr = jrmax

        self.x0_I = x0r[1::,jl:jr]
        self.x0_1L = x0r[:,0:jl]
        self.x0_1R = x0r[:,jr:jrmax]
        self.x0_0L = tf.expand_dims(x0r[0,jl:jr],0)


        x1r = tf.reshape(self.x1,(nx[0]*len(xi1),nx[1]+1,2))

        jrmax = nx[0]*len(xi1)

        if xi0[0]<-1.+1e-10:
            jl = 1
        else:
            jl = 0
        if xi0[-1]>1-1e-10:
            jr = jrmax-1
        else:
            jr = jrmax


        self.xl_I = x1r[1::,1:-1]
        self.x1_1L = tf.expand_dims(x1r[:,0],1)
        self.x1_1R = tf.expand_dims(x1r[:,-1],1)

        self.xl_0L = tf.expand_dims(x1r[0,1:-1],0)


    
    @tf.function
    def getRES(self):
        F0h = tf.concat([self.F0(self.BCl(self.x0_1L)),
                                    tf.concat([self.F0(self.IC(self.x0_0L)),
                                           self.F0(self.u(self.x0_I))],0),
                              self.F0(self.BCr(self.x0_1R))],1)
        F0i = tf.reshape(F0h,(self.nx[0]+1,self.nx[1],len(self.quad['0'][0]),F0h.shape[-1]))
        
        F1h = tf.concat([self.F1(self.BCl(self.x1_1L)),
                                    tf.concat([self.F1(self.IC(self.xl_0L)),
                                           self.F1(self.u(self.xl_I))],0),
                              self.F1(self.BCr(self.x1_1R))],1)
        F1i = tf.reshape(F1h,(self.nx[0],len(self.quad['1'][0]),self.nx[1]+1,F0h.shape[-1]))
        
        F0int  = self.dx[1]/2*tf.tensordot(F0i,self.wi0,[2,0])
        F1int  = self.dx[0]/2*tf.tensordot(F1i,self.wi1,[1,0])
        return  (F0int[1::] - F0int[0:-1])  \
               +(F1int[:,1::] - F1int[:,0:-1])
 