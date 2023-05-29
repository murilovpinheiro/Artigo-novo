import numpy as np
from copy import copy
from  classes.plugin import *
from  classes.classifier import *

PERF_CONS_MAP = {'hmean': 'cov', 'qmean': 'nae', 'fmeasure': 'kld', 'microF1':'cov', 'err': 'dp'}
CONS_UPPER = {'cov': 1.0, 'kld': 1.0, 'dp': 1.0, 'nae': 1.0}

def frank_wolfe(x, y, z, classifier, cpe_model, gamma, epsilon, lr, num_inner_iter, M, lambdas):
    plugin = BinaryPluginClassifier(cpe_model, protected_present=True)
    plugin.set_thresh([0.5] * M)

    C, CC = plugin.evaluate_conf(x, y, z, use_stored_prob=True)
    norm_const = 1.0

    probs = np.zeros((M, 1))
    for i in range(M):
        probs[i] = y[z == i].mean()

    for i in range(num_inner_iter):
        gamma0 = [0.0] * M

        for j in range(M):
            lamda = lambdas[j] * (1 + 1.0/M) # teste eu basicamente supus isso do outro valor
            # Compute costs from objective gradient
            wt_on_neg = gamma - lamda * probs[j] / (probs[j] + C[0, 1] - C[1, 0])\
                            + lamda * (1 - probs[j]) / (1 - probs[j] - C[0, 1] + C[1, 0])
            # dividir por M
            wt_on_pos = 2 - gamma + lamda * probs[j] / (probs[j] + C[0, 1] - C[1, 0])\
                            - lamda * (1 - probs[j]) / (1 - probs[j] - C[0, 1] + C[1, 0])
            # print("%"*200)
            # print(j)
            # print(wt_on_neg, wt_on_pos)
            # print("%"*200)
            gamma0[j] = wt_on_neg * 1.0 / (wt_on_pos + wt_on_neg)
                        
        plugin.set_thresh(gamma0)

        C_hat, CC_hat = plugin.evaluate_conf(x, y, z, use_stored_prob=True)
        C = (1 - 2.0 / (i + 2)) * C + 2.0 / (i + 2) * C_hat

        for j in range(M):
            CC[j, :, :] = (1 - 2.0 / (i + 2)) * CC[j, :, :] + 2.0 / (i + 2) * CC_hat[j, :, :]

        if i == 0:
            classifier.append(1.0, copy(plugin))
        else:
            norm_const *= 1 - 2.0 / (i + 2)
            classifier.append(2.0 / (i + 2) / norm_const, copy(plugin))
        # Update confusion matrix iterate
    
    classifier.weights[-num_inner_iter:-1] = [x * norm_const for x in classifier.weights[-num_inner_iter:-1]]
    classifier.weights[-1] *= norm_const
    
    ## frankwolf ok
    return C, CC, classifier

def coco(x, y, z, classifier, cpe_model, gamma, epsilon, lr, num_outer_iter, num_inner_iter):
    # p = y.mean() # Probabilidade de ser classificado como positivo
    M = 2 # M igual a dois pois temos Z como binário

    s = np.ones((M,)) # não sei o que é isso, ele recebe mas não utiliza para nada

    alpha = np.zeros((M,)) # multiplicadores de Lagrange, um para cada classe
    # quando ele se referia a restrição eu entendia que se referia tipo "dp, kld ou sla" 
    # mas no caso é para cada restrição mesmo, se desejamos otimizar 2 grupos se tratam de 2 restrições
    
    for t in range(num_outer_iter):
        C, CC, _ = frank_wolfe(x, y, z, classifier, cpe_model, gamma, epsilon, lr, num_inner_iter, M, alpha * s)
        ###########################################

        C_mean = np.zeros((2, 2))
        for j in range(M):
            C_mean += CC[j, :, :].reshape((2, 2)) * 1.0/M

        jstar = np.argmax(np.abs(CC[:, 0, 1] + CC[:, 1, 1] - C_mean[0, 1] - C_mean[1, 1]) - epsilon)

        for j in range(M):
            s[j] = np.sign(CC[jstar, 0, 1] + CC[jstar, 1, 1] - C_mean[0, 1] - C_mean[1, 1])

        # Gradient update to alpha
        for j in range(M):
            alpha[j] = alpha[j] + lr * 1.0 / np.sqrt(t + 1) \
                       * (np.abs(CC[jstar, 0, 1] + CC[jstar, 1, 1] - C_mean[0, 1] - C_mean[1, 1]) - epsilon)
            # print("="*200)
            # print(alpha[j])
            # print(lr)
            # print((np.abs(CC[jstar, 0, 1] + CC[jstar, 1, 1] - C_mean[0, 1] - C_mean[1, 1]) - epsilon))
            # print((CC[jstar, 0, 1] + CC[jstar, 1, 1] - C_mean[0, 1] - C_mean[1, 1]) - epsilon)
            # print("="*200)
            # Projection step
            if alpha[j] < 0:
                alpha[j] = 0
        
        obj = 2.0 * (1 - gamma) * C[1, 1] - gamma * (C[1, 0] + C[0, 1])
        #obj = 2.0 * (1 - gamma) * CC[:,1, 1] - gamma * (CC[:,1, 0] + CC[:,0, 1])
    
    classifier.normalize_weights()

    return classifier, obj


def fraco(x, y, z, classifier, cpe_model, epsilon, lr, num_outer_iter, num_inner_iter=1):
    lwr = 0
    upr = 1

    # método da Bissecção, sem segredos
    while upr - lwr > 0.01:
        # Pega o meio
        gamma = (lwr + upr) / 2.0 # BISSECTION

        # chama o COCO com todos os dados enviados e o GAMMA, para que ele leve em consideração ele
        classifier, obj = coco(
            x, y, z, classifier, cpe_model, gamma, epsilon, lr, num_outer_iter, num_inner_iter)
            # esse obj vai ser o que vai ser utilizado para saber para onde ir no método da bissecção

        # dá update nos limites de acordo com a saída
        if obj < 0:
            upr = gamma
        else:
            lwr = gamma

    return classifier
