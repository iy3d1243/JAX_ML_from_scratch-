#we will generate synthetic data with 10 features 
#the goal will be later to estimate back the value of the synthetic weights for a regressoin model 
#the weights should have a meaningfull values and reperesent a context  


#the context will be about the prediction of a car's price from a specific characteristics 


import jax 
import jax.numpy as jnp 
import warnings 
warnings.filterwarnings('ignore')

key=jax.random.PRNGKey(int(input("Give a PRNG key :")))
n=input("input the numbre of samples (default=10_000)")
n= int(n) if n.strip() else 10_000 



