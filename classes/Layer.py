import numpy as np
import ActivationFunctions as acfunc

##The following formulas are used in the backpropagation stage
# x -> input of first layer
# z -> expected output of last layer
# H -> last layer
# zᴸ = (Wᴸ)ᵗ * aᴸ⁻¹ + bᴸ
# aᴸ = h(zᴸ)
# δᴸ = h'(zᴸ) ⊙ Wᴸ⁺¹ * δᴸ⁺¹ unless L = H, then δᴸ = (aᴸ - y) ⊙  h'(zᴸ)
# ∂J/∂Wᴸ = aᴸ⁻¹ * (δᴸ)ᵗ unless L = 0, then ∂J/∂Wᴸ = x * (δᴸ)ᵗ
# ∂J/∂bᴸ = δᴸ
class Layer:

    ##There are n neurons in layer i and m neurons in layer i - 1
    def __init__(self, m, n):
        self.W = np.random.rand(n, m)
        self.b = np.zeros(n)
        self.m = m
        self.n = n
        self.W_error = np.zeros((n, m))
        self.b_error = np.zeros(n)
   
    def calculate_z(self, input):
        
        self.z = np.matmul(self.W, input) + self.b
        #print(self.z)
        return self.z

    def calculate_a(self, input):
        self.calculate_z(input)
        self.a = acfunc.logistic_f(self.z)
        return self.a

    def calculate_a_der(self):
        return acfunc.logistic_d(self.z)

    def calculate_delta_normal(self, W_next, delta_next):
        self.delta = np.multiply(self.calculate_a_der(), np.matmul(delta_next, W_next).T)

    def calculate_delta_last(self, y):
        self.delta = np.multiply((self.a - y), self.calculate_a_der()) 
    
    #if current layer is first layer, then a_before is the input vector
    #only the errors of weights from CURRENT LAYER are returned
    def calculate_error(self, a_before):
        self.W_error = np.outer(self.delta, a_before)
        self.b_error = self.delta



        