import jax 
import jax.numpy as jnp 
from jax import Array,jit

class compute : 

    def __init__(self,n): 
        self.w=jnp.zeros(1,n)
        self.b=0


    def lr_model(self,X)-> Array : 
        '''the represntation of the multiple linear regression model that we are going to be using '''
        return self.w@X+self.b
    

    @jit
    def MSE_loss(self,w,b,X,y)-> float  :
        pred=w@X+b
        return jnp.mean((pred -y)**2) 



    def gradient_descent(self,X,y,iter=100,lr=0.001) : 

        grad=jax.jit(jax.grad(self.MSE_loss,argnums=(0,1)))

        for i in range(iter): 
            dw,db=grad(self.w,self.b,X,y)
            self.w-=lr*dw
            self.b-=lr*db 
            
            if i%10==0 : 
                print(f"iter {i} : loss = {self.MSE_loss(self.w,self.b,X,y)}")



