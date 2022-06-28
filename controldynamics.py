import numpy as np
from qiskit.quantum_info import Operator
from qiskit_dynamics import Solver
from qiskit_dynamics import DiscreteSignal
from qiskit_dynamics.array import Array
from qiskit_dynamics.signals import Convolution

v = 5.
r = 0.02

static_hamiltonian = 2 * np.pi * v * Operator.from_label('Z') / 2
drive_term = 2 * np.pi * r * Operator.from_label('X') / 2

ham_solver = Solver(
    hamiltonian_operators=[drive_term],
    static_hamiltonian=static_hamiltonian,
    rotating_frame=static_hamiltonian,
)



# define convolution filter
def gaus(t):
    sigma = 15
    _dt = 0.1
    return 2.*_dt/np.sqrt(2.*np.pi*sigma**2)*np.exp(-t**2/(2*sigma**2))

convolution = Convolution(gaus)

# define function mapping parameters to signals
def signal_mapping(params):
    samples = Array(params)

    # map samples into [-1, 1]
    bounded_samples = np.arctan(samples) / (np.pi / 2)

    # pad with 0 at beginning
    padded_samples = np.append(Array([0], dtype=complex), bounded_samples)

    # apply filter
    output_signal = convolution(DiscreteSignal(dt=1., samples=padded_samples))

    # set carrier frequency to v
    output_signal.carrier_freq = v

    return output_signal



signal = signal_mapping(np.ones(80) * 1e8)
signal.draw(t0=0., tf=signal.duration * signal.dt, n=1000, function='envelope')



X_op = Array(Operator.from_label('X'))

def fidelity(U):
    U = Array(U)

    return np.abs(np.sum(X_op * U))**2 / 4.



def objective(params):

    # apply signal mapping and set signals
    signal = signal_mapping(params)
    solver_copy = ham_solver.copy()
    solver_copy.signals = [signal]

    # Simulate
    results = solver_copy.solve(
        y0=np.eye(2, dtype=complex),
        t_span=[0, signal.duration * signal.dt],
        method='LSODA',
        atol=1e-8,
        rtol=1e-8
    )
    U = results.y[-1]

    # compute and return infidelity
    fid = fidelity(U)
    return 1. - fid.data



from scipy.optimize import minimize

#jit_grad_obj =np.asarray(np.gradient(objective())) 
initial_guess = np.random.rand(80) - 0.5
jit_grad_obj =np.asarray(np.gradient(objective(initial_guess)))

opt_results = minimize(fun=jit_grad_obj, x0=initial_guess, jac=True, method='BFGS')
print(opt_results.message)
print('Number of function evaluations: ' + str(opt_results.nfev))
print('Function value: ' + str(opt_results.fun))



from scipy.optimize.optimize import MemoizeJac

class MemoizeJacHess(MemoizeJac):
    """ Decorator that caches the return vales of a function returning
        (fun, grad, hess) each time it is called. """

    def __init__(self, fun):
        super().__init__(fun)
        self.hess = None

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None or self.hess is None:
            self.x = np.asarray(x).copy()
            self._value, self.jac, self.hess = self.fun(x, *args)

    def hessian(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.hess
def obj_and_grad_and_hess(x):
    obj = np.exp(x) * x**2
    grad = obj + 2*np.exp(x)*x
    hess = obj + 4*np.exp(x)*(x) + 2*np.exp(x)
    return obj, grad, hess



obj = MemoizeJacHess(objective)
grad = obj.derivative
hess = obj.hessian

res = minimize(obj, x0=[1.0], jac=grad, hess=hess)
res




n_iters=500
for i in range(n_iters):
    gradient = derivative




opt_signal = signal_mapping(opt_results.x)

opt_signal.draw(
    t0=0,
    tf=opt_signal.duration * opt_signal.dt,
    n=1000,
    function='envelope',
    title='Optimized envelope'
)






def minimize_opt_fun(self,x):
        # minimization function called by scipy in each iteration
        self.l,self.rl,self.grads,self.metric,self.g_squared=self.get_error(np.reshape(x,(len(self.sys_para.ops_c),len(x)/len(self.sys_para.ops_c))))
        
        if self.l <self.conv.conv_target :
            self.conv_time = time.time()-self.start_time
            self.conv_iter = self.iterations
            self.end = True
            print 'Target fidelity reached'
            self.grads= 0*self.grads # set zero grads to terminate the scipy optimization
        
        self.update_and_save()
        
        if self.method == 'L-BFGS-B':
            return np.float64(self.rl),np.float64(np.transpose(self.grads))
        else:
            return self.rl,np.reshape(np.transpose(self.grads),[len(np.transpose(self.grads))])

    
    def bfgs_optimize(self, method='L-BFGS-B',jac = True, options=None):
        # scipy optimizer
        self.conv.reset_convergence()
        self.first=True
        self.conv_time = 0.
        self.conv_iter=0
        self.end=False
        print "Starting " + self.method + " Optimization"
        self.start_time = time.time()
        
        x0 = self.sys_para.ops_weight_base
        options={'maxfun' : self.conv.max_iterations,'gtol': self.conv.min_grad, 'disp':False,'maxls': 40}
        
        res = minimize(self.minimize_opt_fun,x0,method=method,jac=jac,options=options)

        uks=np.reshape(res['x'],(len(self.sys_para.ops_c),len(res['x'])/len(self.sys_para.ops_c)))

        print self.method + ' optimization done'
        
        g, l,rl = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss])
            
        if self.sys_para.show_plots == False:
            print res.message
            print("Error = %1.2e" %l)
            print ("Total time is " + str(time.time() - self.start_time))
            
        self.get_end_results()          