import jax 
import jax.numpy as jnp 
from jax import Array,jit
import matplotlib.pyplot as plt  

class Compute : 

    def __init__(self,n): 
        self.w=jnp.zeros((n,1),dtype=jnp.float32)
        self.b = jnp.array(0.0, dtype=jnp.float32)



    def lr_model(self,X)-> Array : 
        '''the represntation of the multiple linear regression model that we are going to be using '''
        return X@self.w+self.b
    

    @staticmethod
    @jit
    def MSE_loss(w,b,X,y)-> float  :
        pred=X@w+b
        return jnp.mean((pred -y)**2) 



    def gradient_descent(self,X,y,iter=100,lr=0.001) : 

        grad=jax.jit(jax.grad(Compute.MSE_loss,argnums=(0,1)))
        loss=[]

        for i in range(iter): 
            dw,db=grad(self.w,self.b,X,y)
            self.w-=lr*dw
            self.b-=lr*db 

            loss.append(float(self.MSE_loss(self.w, self.b, X, y)))
            if 0<i<10 : 
                print(f"iter {i}: loss = {loss[i]:.6f}")
            if i % 10 == 0:
                print(f"iter {i}: loss = {loss[i]:.6f}")

        # plot the training progress
        plt.figure(figsize=(8, 5))
        plt.plot(range(iter), loss, label="Training Loss", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.title("Loss Curve During Training")
        plt.grid(True)
        plt.legend()
        plt.show()
            



