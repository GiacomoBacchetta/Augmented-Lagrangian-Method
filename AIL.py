#!/usr/bin/env python
# coding: utf-8

# In[1]:

import autograd.numpy as np
from autograd import grad
from autograd import hessian
import time


# In[2]:

# PHI
def phi(f, x, d):
    return mxv(d, f, x) + grad(f)(x)


# PRODOTTO MATRICE PER VETTORE
def mxv(v, f, x):
    eta = 1e-6
    x_succ = x + eta*v
    return (grad(f)(x_succ) - grad(f)(x))/eta


# CONDIZIONI DI KKT STANDARD, LE PIU UTILIZZATE
def verify_KKT(f, g, h, x, lam, mu):
    epsilon = 1e-3
    for i in range(len(g)):
        if g[i](x) > epsilon or lam[i]*g[i](x) > epsilon or lam[i]*g[i](x) < -epsilon:
            return False
    for j in range(len(h)):
        if h[j](x) > epsilon or h[j](x) < -epsilon:
            return False

    if np.linalg.norm(grad(f)(x) + sum(grad(g[i])(x)*lam[i] for i in range(len(g))) + sum(grad(h[j])(x)*mu[j] for j in range(len(h)))) > epsilon:
        return False

    return True


# CONDIZIONI DI KKT MODIFICATE DA FABIO
def verify_KKT_mod(f, g, h, x, lam, mu):
    epsilon = 1e-3
    for i in range(len(g)):
        if g[i](x) > epsilon or lam[i]*g[i](x) > epsilon or lam[i]*g[i](x) < -epsilon:
            return [False, np.linalg.norm(kkt), compl]
    for j in range(len(h)):
        if h[j](x) > epsilon or h[j](x) < -epsilon:
            return [False, np.linalg.norm(kkt), compl]
    kkt = grad(f)(x)+sum(lam[i]*grad(g[i])(x) for i in range(len(g))
                         ) + sum(mu[j]*grad(h[j])(x) for j in range(len(h)))
    lb = np.zeros(len(kkt))
    ub = np.zeros(len(kkt))

    for i in range(len(kkt)):
        lb[i] = -1e-3
        ub[i] = 1e-3

    if (kkt > ub).any() or (kkt < lb).any():
        return [False, np.linalg.norm(kkt), compl]

    return [True, np.linalg.norm(kkt), compl]


# CONDIZIONE DI ARMIJO, PASSO DI DISCESA
def armijo(f, x_k, d):
    alpha = 1
    gamma = 1e-3

    while True:
        if f(x_k + alpha*d) > f(x_k) + alpha*gamma*(grad(f)(x_k) @ d):
            alpha = 0.5*alpha
        else:
            return alpha



# METODO DEL GRADIENTE
def gradient_descent(f, x_0, delta):
    x = x_0
    #k = 0

    while True:
    
        #k += 1
        if np.linalg.norm(grad(f)(x)) < delta:
            print('Il valore della funzione obiettivo è:',f(x))
            return x
        else:
            d = -(grad(f)(x))
            a = armijo(f, x, d)
            x = x + a*d

            
# METODO DEL GRADIENTE MODIFICATO
def gradient_descent_mod(f, x0, eps = 1e-5):
    x = [x0]
    k = 0
    
    while True:
        print (k)
        print('Il valore della f.o. ', f(x[k]))
        print('Il valore della norma', np.linalg.norm(grad(f)(x[k])))
        print('il punto corrente', x[k])
        if np.linalg.norm(grad(f)(x[k]))<= eps:
            return [x[k], f(x[k])]
        d = -grad(f)(x[k])
        a = armijo(f, x[k], d)
        x.append(x[k]+a*d)
        
        
        if f(x[k+1]) > (f(x[k]) - 1e-6) and f(x[k+1]) < (f(x[k]) + 1e-6):
            return newton_method_mod(f, x[k+1])
        
        k = k+1
            


# STESSA FUNIONE DI PHI ALL'INIZIO DEL FILE
def gradfi(d,f,x):
    return mxv(d,f,x) + grad(f)(x)


# In[5]:

# DIREZIONE TRONCATA
def direzione_troncato(f, x, k):

    epsilon_1 = 0.5
    epsilon_2 = 0.5
    p = 0

    s = -(gradfi(p, f, x))
    
    if (s @ mxv(s, f, x)) < (epsilon_1 * (np.linalg.norm(s))**2):
        d = -(grad(f)(x))
        return d
    
    while True:
        
        if (s @ mxv(s, f, x)) <= 1e-6:              #default = 1e-9
            return -grad(f)(x)
        
        #print('K1')
        alfa = -((gradfi(p, f, x) @ s) / (s @ mxv(s, f, x)))
        #print('K2')
#         ref = p
        p = p + alfa * s
#         if (np.linalg.norm(ref - p))**2 < 1e-5:
#             return p

        if np.linalg.norm(gradfi(p, f, x)) <= (1/(k+1))*epsilon_2*(np.linalg.norm(grad(f)(x))):
            d = p
            return d
        else:
            #print('K3')
            beta = (gradfi(p, f, x) @ mxv(s, f, x)) / (s @ mxv(s, f, x))
            s = -(gradfi(p, f, x)) + beta * s

            if (s @ mxv(s, f, x)) < (epsilon_1 * (np.linalg.norm(s))**2):
                d = p
                return d

            
# GRADIENTE CONIUGATO
def gradiente_coniugato(f, x_k, k):
    eps_1 = 0.5
    eps_2 = 0.5
    i = 0
    p = [np.zeros(len(x_k))]
    s = [- grad(f)(x_k)]

    if s[0] @ mxv(s[0], f, x_k) < eps_1 * (np.linalg.norm(s[0])**2):
        d = - grad(f)(x_k)
        return d

    while True:

        if abs(s[i]@mxv(s[i], f, x_k)) <= 1e-9:
            return -grad(f)(x_k)

        alpha = -((phi(f, x_k, p[i])@s[i])/(s[i]@mxv(s[i], f, x_k)))
        p.append(p[i]+alpha*s[i])

        if (np.linalg.norm(p[i]-p[i-1]))**2 < 1e-11:  # e-11
            return p[i]

        if np.linalg.norm(phi(f, x_k, p[i+1])) <= (1/(k+1)) * eps_2 * np.linalg.norm(grad(f)(x_k)):
            d = p[i+1]
            return d

        i = i+1

        beta = (phi(f, x_k, p[i])@mxv(s[i-1], f, x_k)) / \
            (s[i-1]@mxv(s[i-1], f, x_k))
        s.append(-phi(f, x_k, p[i])+beta*s[i-1])

        if s[i]@mxv(s[i], f, x_k) < eps_1 * (np.linalg.norm(s[i])**2):
            d = p[i]
            return d
            

            
# NEWTON TRONCATO
def newton_troncato(f, x_0, eps=1e-5):
    x = [x_0]
    k = 0

    if np.linalg.norm(grad(f)(x[k])) < eps:
#         print('Punto già stazionario con norma:', np.linalg.norm(grad(f)(x[k])))
        return [x[k], f(x[k])]

    while True:

#         print('NT: Iterazione Newton Troncato:', k)
#         print('NT: Norma del gradiente:', np.linalg.norm(grad(f)(x[k])))

#         if np.linalg.norm(grad(f)(x[k])) > 1e6:    #condizione aggiunta successivamente
#             print(x[k])                            
#             return [x[k], f(x[k])]                 #condizione inutile, basta giocare gli con epsilon in PS

        d = gradiente_coniugato(f, x[k], k)
        a = armijo(f, x[k], d)

        x.append(x[k]+a*d)
        k = k+1

        if np.linalg.norm(grad(f)(x[k])) < eps:
#             print('NT: Il punto calcolato è:', x[k])
            return [x[k], f(x[k])]
        
        if abs(f(x[k-1])-f(x[k])) < 1e-9:
#             print('NT: Il punto calcolato è:', x[k])
            return [x[k], f(x[k])]        
            
            
# In[6]:
        
        
def newton_method(f, x_0, delta):
    x = [x_0]
    k = 0
   
    if np.linalg.norm(grad(f)(x[0])) < 1e-5:
        return [x[0], f(x[0])]
   
    while True:
        d = - np.linalg.inv(hessian(f)(x[k])) @ grad(f)(x[k])
        x.append(x[k]+d)
       
        if np.linalg.norm(grad(f)(x[k+1])) < delta:
            return [np.round(x[k+1]), round(f(x[k+1]))]
       
        k = k+1
        print(k)
        print(np.linalg.norm(grad(f)(x[k])))
        print(f(x[k]))


def newton_method_mod(f, x_0, eps):
    x = [x_0]
    k = 0
    
    if np.linalg.norm(grad(f)(x[0])) < eps:
        return [x[0], f(x[0])]
    
    while True:
        print('ECCO NEWTON', k)
        if abs(np.linalg.det(hessian(f)(x[k])))>1e-3:
            d = - np.linalg.inv(hessian(f)(x[k])) @ grad(f)(x[k])
            x.append(x[k]+d)
        else:
            d = -grad(f)(x[k])
            a = armijo(f, x[k], d)
            x.append(x[k]+a*d)
        
        if np.linalg.norm(grad(f)(x[k+1])) < eps:
            return [x[k+1], f(x[k+1])]
        
        k = k+1
        print(k)
        print(np.linalg.norm(grad(f)(x[k])))
        print(f(x[k]))