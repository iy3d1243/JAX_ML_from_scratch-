import jax 
import jax.numpy as jnp 
import numpy as np
import warnings 
warnings.filterwarnings('ignore')


#we will generate synthetic data with 10 features 
#the goal will be later to estimate back the value of the synthetic weights for a regressoin model 
#the weights should have a meaningfull values and reperesent a context  


#the context will be the prediction of a car's price from a specific characteristics 

'''

features : 
x₁ — Age :	                    How old the car is (in years)	0---15
x₂ — Mileage :	                Total kilometers/miles driven	0---300,000
x₃ — Engine size :	            Engine displacement (liters or cc)	1.0---5.0
x₄ — Horsepower :	            Engine power output	50---500 HP
x₅ — Fuel type :	            Encoded as numbers (0 = Petrol, 1 = Diesel, 2 = Electric, etc.)	0---2
x₆ — Transmission type :	    0 = Manual, 1 = Automatic	0---1
x₇ — Number of doors :	        Total doors on the car	2---5
x₈ — Brand reputation score :	Average reliability/resale reputation (normalized ---0)	0---10
x₉ — Accident history :	        Number of recorded accidents	0---5
x₁₀ — Fuel efficiency :	        Average km/l or mpg	5---40
Y : the Price 

'''


key=input("Give a PRNG key (default=22) :")
key=jax.random.PRNGKey(int(key) if key.strip() else 22)

n=input("input the numbre of samples (default=100_000)")
n= int(n) if n.strip() else 100_000 

age_key,mileage_key,engine_size_key,horsepower_key,fuel_type_key,transmission_type_key,number_of_doors_key,\
brand_reputation_score_key,accident_history_key,fuel_efficiency_key,price_key=jax.random.split(key,11)

age=jax.random.randint(age_key,(n,1),minval=0,maxval=35) 
mileage=jnp.clip((1+0.3*jax.random.normal(mileage_key,(n,1)))*age*25_000,a_min=0) 
engine_size = jax.random.uniform(engine_size_key, shape=(n,1), minval=1.0, maxval=5.0)
horsepower = jnp.clip(engine_size * 100 + 50 + 30 * jax.random.normal(horsepower_key, (n,1)), a_min=50, a_max=500)
fuel_type = jax.random.randint(fuel_type_key, (n,1), minval=0, maxval=3)
transmission_type = jax.random.randint(transmission_type_key, (n,1), minval=0, maxval=2)
number_of_doors = jax.random.randint(number_of_doors_key, (n,1), minval=2, maxval=6)
brand_reputation_score = jnp.clip(5 + 2 * jax.random.normal(brand_reputation_score_key, (n,1)), a_min=0, a_max=10)
accident_history = jax.random.randint(accident_history_key, (n,1), minval=0, maxval=6)
fuel_efficiency = jnp.clip(40 - 3*engine_size - 0.02*horsepower + 2*jax.random.normal(fuel_efficiency_key, (n,1)), a_min=5, a_max=40)


#now this what matters 
#we will try to predict this equation later 

base_price = 15000
price= base_price+3000*engine_size + 40*horsepower + 200*fuel_efficiency + 1000*fuel_type +2000*brand_reputation_score + 10*number_of_doors + 500*transmission_type - 1500*accident_history -1500*age -0.05*mileage+500*jax.random.uniform(price_key,(n,1))
price=jnp.clip(price,a_min=1000,a_max=500_000)

data=jnp.concatenate([age,mileage,engine_size,horsepower,fuel_type,transmission_type,number_of_doors,brand_reputation_score,accident_history,fuel_efficiency,price],axis=1 )

np.savetxt("synthetic_car_data.csv",np.round(data),fmt='%.0f',delimiter=",",header="age,mileage,engine_size,horsepower,fuel_type,transmission_type,number_of_doors,brand_reputation_score,accident_history,fuel_efficiency,price",comments='')
print("synthetic car dataset generated and saved to synthetic_car_data.csv")
