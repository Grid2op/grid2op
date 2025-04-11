import time

import scipy

import jax.experimental
import grid2op
import warnings
from lightsim2grid import LightSimBackend
from scipy.sparse import coo_matrix
import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import sparse
import optax
    
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    env = grid2op.make("l2rpn_case14_sandbox", test=True, backend=LightSimBackend())
    
grid = env.backend._grid
time_nr = grid.get_computation_time()
# NB get_computation_time returns "time_total_nr", which is
# defined in the powerflow algorithm and not on the linear solver.
# it takes into account everything needed to solve the powerflow
# once everything is passed to the solver.
# It does not take into account the time to format the data in the 
# from the GridModel 
print(f"Time to perform 1 powerflow : {time_nr}s on 1 cpu {1./time_nr:.2e} powerflow / s")
# numpy
Ybus = coo_matrix(grid.get_Ybus())
Sbus = grid.get_Sbus()
pv = grid.get_pv()
pq = grid.get_pq()
v_init = np.ones(Ybus.shape[0], dtype=complex)
v_final = grid.get_V()
pvpq = np.concatenate((pv, pq))
slack_id = np.array([0])
pv_and_slack = np.concatenate((slack_id, pv))

# jax
Ybus_jax = sparse.BCOO.from_scipy_sparse(Ybus)
Sbus_jax = jnp.asarray(Sbus, copy=True)
pv_jax = jnp.asarray(pv, copy=True)
pq_jax = jnp.asarray(pq, copy=True)
v_init_jax = jnp.asarray(v_init, copy=True)

v_final_jax = jnp.asarray(v_final, copy=True)
pvpq_jax = jnp.asarray(pvpq, copy=True)
slack_id_jax = jnp.asarray(slack_id, copy=True)
pv_and_slack_jax = jnp.asarray(pv_and_slack, copy=True)
init_v_jax = np.abs(v_final_jax[pv_and_slack_jax])

# numpy version
def get_error(v, Ybus, Sbus, pvpq, pq):
    mis = v * np.conj((Ybus @ v)) - Sbus
    # p mismatch
    tmp_p = mis[pvpq].real
    mis_p = tmp_p * tmp_p
    
    # q mismatch
    tmp_q = mis[pq].imag
    mis_q = tmp_q * tmp_q
    
    return 0.5 * (mis_p.sum() + mis_q.sum())

# jax version
def get_vcplx_from_v_theta(v_theta):
    return v_theta[:28] * (jnp.cos(v_theta[28:]) + 1j * jnp.sin(v_theta[28:]))

def get_vcplx_from_v_theta_2(v, theta):
    return v * (jnp.cos(theta) + 1j * jnp.sin(theta))

def get_v_theta_from_vcplx(vcplx):
    return jnp.concatenate((jnp.abs(vcplx), jnp.angle(vcplx))) 

def get_error_jax(v_theta, Ybus, Sbus, pvpq, pq):
    # compute complex voltage
    v = get_vcplx_from_v_theta(v_theta)
    
    # compute mismatch
    mis = v * jnp.conj((Ybus @ v)) - Sbus
    
    # p mismatch
    tmp_p = jnp.real(mis[pvpq])
    mis_p = tmp_p * tmp_p
    
    # q mismatch
    tmp_q = jnp.imag(mis[pq])
    mis_q = tmp_q * tmp_q
    
    return 0.5 * (mis_p.sum() + mis_q.sum())

def get_error_jax_2(v, theta, Ybus, Sbus, pvpq, pq):
    # compute complex voltage
    v = get_vcplx_from_v_theta_2(v, theta)
    
    # compute mismatch
    mis = v * jnp.conj((Ybus @ v)) - Sbus
    
    # p mismatch
    tmp_p = jnp.real(mis[pvpq])
    mis_p = tmp_p * tmp_p
    
    # q mismatch
    tmp_q = jnp.imag(mis[pq])
    mis_q = tmp_q * tmp_q
    
    return 0.5 * (mis_p.sum() + mis_q.sum())

get_error_jax_jit = jax.jit(get_error_jax)
get_error_jax_2_jit = jax.jit(get_error_jax_2)
derr_dv = jax.grad(get_error_jax_jit)

# numpy
error_init_np = get_error(v_init, Ybus, Sbus, pvpq, pq)
error_final_np = get_error(v_final, Ybus, Sbus, pvpq, pq)
    
# jax
error_init_jax = get_error_jax(get_v_theta_from_vcplx(v_init_jax), Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
error_final_jax = get_error_jax(get_v_theta_from_vcplx(v_final_jax), Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
error_final_jax_jit = get_error_jax_jit(get_v_theta_from_vcplx(v_final_jax), Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
print(f"Ref error : {error_final_jax_jit}")

# solve just by the gradient
# v_theta = get_v_theta_from_vcplx(1. * v_init_jax)
# v_theta = v_theta.at[pv_and_slack_jax].set(init_v_jax)
# lambda_ = 3e-4
# for i in range(1000):
#     grad_ = derr_dv(v_theta, Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
#     # I do not modify magnitude of pv bus
#     grad_ = grad_.at[pv_and_slack_jax].set(0.)
#     # I do not modify the slack theta
#     grad_ = grad_.at[slack_id_jax + 28].set(0.)
#     # perform the update
#     v_theta -= lambda_ * grad_
#     v_theta = get_v_theta_from_vcplx(get_vcplx_from_v_theta(v_theta))
    
#     # v_ = v_.at[pv_and_slack_jax].set(v_[pv_and_slack_jax] / np.abs(v_[pv_and_slack_jax]) * init_v_jax)
#     # print(f"angle: {jnp.angle(v_theta[slack_id_jax + 28])}")
#     # print(f"magn: {jnp.abs(v_theta[pv_and_slack_jax])}")
    
#     this_err = get_error_jax_jit(v_theta, Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
#     if i % 10 == 0:
#         print(f"\terror at iteration {i}: {this_err}")
# print(f"Final error {this_err}")

# use optax
# for 1 powerflow
learning_rate = 3e-2
optimizer = optax.adam(learning_rate)
params = {'v': jnp.abs(v_init_jax), "theta": jnp.angle(v_init_jax)}
opt_state = optimizer.init(params)
compute_loss = lambda params, Ybus_jax, Sbus_jax, pvpq_jax, pq_jax: get_error_jax_2(params['v'], params['theta'], Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
dloss_dparm_jit = jax.jit(jax.grad(compute_loss))
beg_ = time.perf_counter()
for i in range(200):
    grads = dloss_dparm_jit(params, Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
#     if i % 10 == 0:
#         this_err = get_error_jax_2_jit(params["v"], params["theta"], Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
#         print(f"\terror at iteration {i}: {this_err}")
# print(f"Final error {this_err}")
end_ = time.perf_counter()
print(f"Time to perform 1 powerflow : {end_ - beg_:.2e}s on cpu {1. / (end_ - beg_):.2e} pf/s")

# for lots of powerflow
# TODO
nb_powerflows = 10_000

# scipy world
nb_buses_one = Ybus.shape[0]
Ybuses = scipy.sparse.block_diag([Ybus for _ in range(nb_powerflows)])
Sbuses = np.concatenate([Sbus for _ in range(nb_powerflows)])
v_inits = np.concatenate([v_init for _ in range(nb_powerflows)])
pvs = np.concatenate([pv + i*nb_buses_one for i in range(nb_powerflows)] )
pqs = np.concatenate([pq + i*nb_buses_one for i in range(nb_powerflows)] )
pvpqs = np.concatenate((pvs, pqs))

# back to jax
Ybuses_jax = sparse.BCOO.from_scipy_sparse(Ybuses)
Sbuses_jax = jnp.asarray(Sbuses, copy=True)
pqs_jax = jnp.asarray(pqs, copy=True)
pvpqs_jax = jnp.asarray(pvpqs, copy=True)
v_inits_jax = jnp.asarray(v_inits, copy=True)
# jax.lax.concatenate([v_init_jax for _ in range(nb_powerflows)], dimension=0)
optimizer_s = optax.adam(learning_rate)
params_s = {'v': jnp.abs(v_inits_jax),
            "theta": jnp.angle(v_inits_jax)}
opt_state_s = optimizer_s.init(params_s)


def training_loop_jax(nb_it,
                      dloss_dparm,
                      optimizer_s,
                      opt_state_s,
                      params_s,
                      Ybuses_jax,
                      Sbuses_jax,
                      pvpqs_jax,
                      pqs_jax):
    
    def body_fun(it_num,
                 args):
        (dloss_dparm,
        optimizer_s,
        opt_state_s,
        params_s,
        Ybuses_jax,
        Sbuses_jax,
        pvpqs_jax,
        pqs_jax) = args
        grads = dloss_dparm(params_s, Ybuses_jax, Sbuses_jax, pvpqs_jax, pqs_jax)
        updates, opt_state_s = optimizer_s.update(grads, opt_state_s)
        params_s = optax.apply_updates(params_s, updates)
        res = (dloss_dparm,
               optimizer_s,
               opt_state_s,
               params_s,
               Ybuses_jax,
               Sbuses_jax,
               pvpqs_jax,
               pqs_jax)
        return res
    
    init = (dloss_dparm,
            optimizer_s,
            opt_state_s,
            params_s,
            Ybuses_jax,
            Sbuses_jax,
            pvpqs_jax,
            pqs_jax)
    res = jax.lax.fori_loop(0, 2,  body_fun, init)
    (dloss_dparm,
     optimizer_s,
     opt_state_s,
     params_s,
     Ybuses_jax,
     Sbuses_jax,
     pvpqs_jax,
     pqs_jax) = res
    return params_s

training_loop_jax_jit = jax.jit(training_loop_jax,
                                static_argnames=('dloss_dparm',
                                                 'optimizer_s',
                                                 'nb_it',
                                                #  'Ybuses_jax',
                                                #  'Sbuses_jax',
                                                #  'pvpqs_jax',
                                                #  'pqs_jax'
                                                 ))

beg_ = time.perf_counter()
training_loop_jax_jit(200, 
                      dloss_dparm_jit,
                      optimizer_s,
                      opt_state_s,
                      params_s,
                      Ybuses_jax,
                      Sbuses_jax,
                      pvpqs_jax,
                      pqs_jax)
end_ = time.perf_counter()
print(f"Time to perform {nb_powerflows} powerflow : {end_ - beg_:.2e}s on cpu  {nb_powerflows / (end_ - beg_):.2e} pf/s")


# newton raphson (does not work ! need to really invert !)
# v_theta = get_v_theta_from_vcplx(1. * v_init_jax)
# v_theta = v_theta.at[pv_and_slack_jax].set(init_v_jax)
# this_err = get_error_jax_jit(v_theta, Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
# for i in range(10):
#     grad_ = derr_dv(v_theta, Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
    
#     grad_ = grad_.at[pvpq_jax + 28].set(jnp.where(jnp.abs(grad_[pvpq_jax + 28]) > 1e-3, grad_[pvpq_jax + 28], 1.))
    
#     # perform the update
#     v_theta = v_theta.at[pq_jax].set(v_theta[pq_jax] - this_err / grad_[pq_jax]) # |v| of pv does not change
#     v_theta = v_theta.at[pvpq_jax + 28].set(v_theta[pvpq_jax + 28] - this_err / grad_[pvpq_jax + 28])
#     v_theta = get_v_theta_from_vcplx(get_vcplx_from_v_theta(v_theta))
    
#     # v_ = v_.at[pv_and_slack_jax].set(v_[pv_and_slack_jax] / np.abs(v_[pv_and_slack_jax]) * init_v_jax)
#     # print(f"angle: {jnp.angle(v_theta[slack_id_jax + 28])}")
#     # print(f"magn: {jnp.abs(v_theta[pv_and_slack_jax])}")
    
#     this_err = get_error_jax_jit(v_theta, Ybus_jax, Sbus_jax, pvpq_jax, pq_jax)
#     if i % 1 == 0:
#         print(f"\terror at iteration {i}: {this_err}")
# print(f"Final error {this_err}")
