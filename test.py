import jax 
import jax.numpy as jnp 

key=jax.random.PRNGKey(3)

print(jax.random.normal(key,(1,1)))