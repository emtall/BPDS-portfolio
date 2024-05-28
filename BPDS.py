class BPDS_TVAR(object):
    """
    A minimal model class for use in BPDS portfolio examples that is fed in relevant attributes from a full model

    ...

    Attributes
    ----------
    name : str
        name/description of model
    opt_x : array_like
        2D (T x n) array containing n-dimensional model optimal decision made at t-1 targeting time t
    samples : array_like
        3D (T x n_rep x q) containing n_rep samples of the q-dimensional outcome of interest y_{t}, here asset returns between time t-1 and t
    likelihoods : array_like
        1D array of length T containing the log-likelihood of y_{t-1}|D_{t-2}, with the realized value of y_{t-1} observed at t-1
    rs : array_like
        1D array of length T containing the target portfolio return at each time point


    Methods
    -------
    predict(t)
        Accesses relevant samples for time t
    
    get_opt_x(t)
        Accesses model-specific optimal decision for time t
        
    """
    
    def __init__(self,model, name):
        
        self.name = name
        self.opt_x = model.ws 
        self.samples = model.samples 
        self.likelihoods = model.likelihoods
        self.rs = model.rs
                
    def predict(self, t):
        return(self.samples[t])

    def get_opt_x(self, t):
        return(self.opt_x[t])
        
class BPDS_port(object):

    """
    A class to run and evaluate BPDS for financial portfolio with target return r_t + \tau_1/\tau_2 and constraint \tau_1/\tau_2 < r_t

    ...

    Attributes
    ----------
    models : list
        J-length list of models, including a baseline
    r : float
        Initial portfolio return target
    s_imp : array_like
        Percent improvement targets for setting the BPDS target score, for example s_imp = [1.05, .9] 
        indicates a 5% improvement in return and a 10% reduction in the variance
    alpha : float
        Default is 1, defines how much the BMA weights should decay over time 
    h : int
        Default is None, defines which horizon of interest, if any, to evaluate

    Methods
    -------
    
    fit(self, T = 1, verbose = False)
        Runs BPDS up to time T, set verbose to True to have a progress bar displayed
        
    score_f(self, samples, opt_x, j = None)
        Scores the samples of outcome y_t under decision opt_x for model j
        
    pi_calc(self, models, t, pis)
        Calcualtes the initial probabilities using discounted BMA probabilities
    
    x_calc(self, tau, opt_x, pis)
        Calculates the optimal portoflio using Markowitz optimization using BPDS estimates of the mean and covariance of returns, with target r_t + \tau_1/\tau_2
    
    s_calc(self, scores, pis)
        Calculates the BPDS target score
    
    get_returns(self, data)
        After running fit, run to calculate the returns of the BPDS model (returns DataFrame of model returns + BPDS returns)
    
    _tau_optimize(self,tau, s, pis)
        Internal function to optimize for tau
    
    _update_params(self, pis, t)
        Updates target return
    """
     
    def __init__(self, models, r, s_imp,  alpha=1, h=None):
       
        self.models = models
        self.J = len(models)
        self.r = r
        self.s_imp = s_imp
        self.alpha = alpha
        self.rs = np.zeros(models[0].rs.shape)

    def fit(self, T = 1, verbose = False):
        # T is number of time steps
        self.m = self.models[0].get_opt_x(0).shape
        self.decisions = np.zeros((T, *self.m))
        self.x = np.zeros(self.m)
        b = self.score_f(self.models[0].predict(0), self.models[0].get_opt_x(0), 0).shape[0]
        
        self.tau = np.array([.01, .1])
        self.bounds = tuple([(-25, 25) for i in range(len(self.tau))]) # bounds needed to ensure optimization doesn't wander off
        self.cons =  {'type': 'ineq','fun': lambda x: self.r - x[0]/x[1]}
        self.taus = np.zeros((T,b))
        self.s_store = {"prior": np.zeros((T,b)), "actual": np.zeros((T,b))}

        pis = np.ones(self.J)/(self.J)
        self.pi_store = {"prior": np.zeros((T,self.J)), "BPDS": np.zeros((T,self.J))}
        self.score_means = np.zeros((T, b))
        self.score_V = np.zeros((T, b, b))
        
        if verbose:
            l = tqdm(range(T))
        else:
            l = range(T)
            
        for t in l:        
                        
            pis = self.pi_calc(self.models, t, pis)
            self.pi_store["prior"][t] = pis
            self._update_params(pis, t)
            
            self.samples = np.array([self.models[j].predict(t) for j in range(self.J)])
            self.opt_x = np.array([self.models[j].get_opt_x(t) for j in range(self.J)])
            self.scores = np.array([self.score_f(self.samples[j], self.opt_x[j], j) for j in range(self.J)])
            self.s = self.s_calc(self.scores, pis)
            
            s0 = np.multiply(pis[:, None, None], self.scores).sum(axis=0).mean(axis=1)
            means = np.array([np.vstack(self.scores[i].mean(axis=1)) for i in range(len(pis))])
            V = np.array([pis[i]*(np.cov(self.scores[i])+means[i]@means[i].T) for i in range(len(pis))]).sum(axis=0)
            V = V - np.vstack(s0)@np.vstack(s0).T
            self.score_V[t] = V
            self.score_means[t] = s0
            
            if self.cons is not None and self.tau[0]/self.tau[1] > self.r:
                #resets initial value for tau to ensure constraint is satisfied
                self.tau[0] = self.tau[0]*self.r/(self.tau[0]/self.tau[1])
                
            if any(self.tau > 5) or any(np.round(self.tau, 5) == 0):
                #if any of the previous values for tau were fairly large/small, try two starting points
                opt = minimize(self._tau_optimize,self.tau,args=(self.s, pis), bounds = self.bounds, constraints = self.cons)
                start2 = np.array([.1, 1])
                if self.cons is not None:
                    start2[0] = start2[0]*self.r/(start2[0]/start2[1])
                opt2 = minimize(self._tau_optimize,start2,args=(self.s, pis), bounds = self.bounds, constraints = self.cons)
                if opt.fun < opt2.fun:
                    self.tau = opt.x.reshape(-1)
                else:
                    self.tau = opt2.x.reshape(-1)
            else:
                opt = minimize(self._tau_optimize,self.tau,args=(self.s, pis), bounds = self.bounds, constraints = self.cons)
                self.tau = opt.x.reshape(-1)

            a_s = np.mean(np.exp(self.tau@self.scores), axis=1)
            a_deriv = np.mean(self.scores*np.exp(self.tau@self.scores)[:, None, :], axis=2)
            
            self.pi_store["BPDS"][t] = pis*a_s/(pis@a_s)
            self.x = self.x_calc(self.tau, self.opt_x, pis)
            
            self.decisions[t] = self.x
            self.taus[t] = self.tau
            self.s_store["prior"][t] = self.s
            self.s_store["actual"][t] = 1/(pis@a_s)*pis@a_deriv
    
    def score_f(self, samples, opt_x, j = None):
        returns = np.multiply(samples,opt_x[None, :]).sum(axis=1)
        means = returns
        sq = (np.subtract(returns, self.r)**2)
        return(np.array([means, -.5*sq]))
    
    def pi_calc(self, models, t, pis):
        for j in range(len(pis)):
            pis[j] = pis[j]**self.alpha*np.exp(self.models[j].likelihoods[t, 0] - min([self.models[i].likelihoods[t, 0] for i in range(len(self.models))] ))
        return(pis/sum(pis))
        
    def x_calc(self, tau, opt_x, pis):
        a_s = np.mean(np.exp(tau@self.scores), axis=1)
        k = 1/(pis@a_s)
        opt_x = np.zeros(self.m)

        fs = (np.exp(tau@self.scores)[:, :, None]*self.samples).mean(axis=1)
        f = np.vstack(k*(pis@fs))

        Vs = [np.cov(np.exp(tau@self.scores[j])[ :, None]*self.samples[j], rowvar=False) for j in range(len(pis))]

        V = np.sum([k*pis[j]* (Vs[j] + np.vstack(fs[j])@np.vstack(fs[j]).T) for j in range(len(pis))], axis=0)
        V = V-f@f.T
        opt_x = target_return_port(self.r+tau[0]/tau[1], f, V).T[0]
            
        return(opt_x)
        
    def s_calc(self, scores, pis):
        s = np.multiply(pis[:, None, None], scores).sum(axis=0).mean(axis=1)*self.s_imp
        return(s)
        
    def get_returns(self, data):
        self.rets = pd.DataFrame(index = data.index[1:len(self.decisions)])
        for j in range(len(self.models)):
            self.rets[self.models[j].name] = self.models[j].returns[:-1]
        self.rets["BPDS"]= (self.decisions[:-1]*data.iloc[1:len(self.decisions)]).sum(axis=1)
        return(self.rets)
        
    def _tau_optimize(self,tau, s, pis):
        a_s = np.mean(np.exp(tau@self.scores), axis=1)
        a_deriv = np.mean(self.scores*np.exp(tau@self.scores)[:, None, :], axis=2)
        g = s - 1/(pis@a_s)*pis@a_deriv
        return(sum(abs(g)))
    
    def _update_params(self, pis, t):
        self.r = pis@np.array([self.models[j].rs[t] for j in range(len(self.models))])
        self.rs[t] = self.r

        

"""
Example Code after reading in relevant data and models

Attributes
----------
mods : list
    list of BPDS_TVAR models, including a baseline at index 0
r : float
    Initial portfolio return target
s_imp : array_like
    Percent improvement targets for setting the BPDS target score, for example s_imp = [1.05, .9] 
    indicates a 5% improvement in return and a 10% reduction in the variance
alpha : float
    Default is 1, defines how much the BMA weights should decay over time 
data_pct: pandas DataFrame
    Contains asset returns up to time T
"""

BPDS = BPDS_port(mods, r=.1, s_imp=np.array([1.05, .9]), alpha=.8)
BPDS.fit(T = len(data_pct), verbose = True)
returns = BPDS.get_returns(data_pct)
        
