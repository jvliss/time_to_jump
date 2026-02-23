from numba import jit, prange
from numba.extending import get_cython_function_address
import scipy.stats as stats
import numpy as np
import ctypes
import bayesflow as bf

# Get a pointer to the C function levy.c
addr_levy= get_cython_function_address("levy_noise", "levy_noise")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
levy_noise = functype(addr_levy)

def levy_prior(fix_alpha, batch_size):
    
    # --- Mean RT --- #
    #Voss et al. (2019; multi-speed): 
    ##correct: M = 1110, SD = 420
    ##incorrect: M = 1258, SD = 617
    #Evans et al. (2017, speed): M = 0.69, SD = 0.15 
    #Evans et al. (2017, accuacy): M = 1.29, SD = 0.47

    # --- Accuracy --- #
    #Voss et al. (2019; multi-speed): M = .83, SD = .08
    #Evans et al. (2017, speed): M = .81, SD = .08 
    #Evans et al. (2017, accuacy): M = .86, SD = .07

    # --- Nondecision time --- #
    #Voss et al. (2019; multi-speed): M = 0.51, SD = 0.10
    #Hato et al. (2025; speed): M = 0.42, SD = 0.06
    #Hato et al. (2025; accuracy): M = 0.46, SD = 0.08
    mean_t0 = 0.5
    std_t0 = 0.25
    clip_a_t0 = 0.25
    clip_b_t0 = 1.5
    a_t0, b_t0 = (clip_a_t0 - mean_t0) / std_t0, (clip_b_t0 - mean_t0) / std_t0
    t0 = stats.truncnorm.rvs(loc=mean_t0, scale=std_t0, a=a_t0, b=b_t0, size=batch_size)

    # --- Intertrial variabiliy of nondecision time --- #
    #Voss et al. (2019; multi-speed): M = 0.30, SD = 0.14
    #Hato et al. (2025; speed): M = 0.13, SD = 0.06
    #Hato et al. (2025; accuracy): M = 0.10, SD = 0.08
    st0 = np.random.uniform(0.0, clip_a_t0*2, size=batch_size)

    # --- Threshold separation --- #
    #Voss et al. (2019; multi-speed): M = 1.92, SD = 0.76
    #Hato et al. (2025; speed): M = 0.88, SD = 0.30
    #Hato et al. (2025; accuracy): M = 2.14, SD = 0.56
    a = np.random.gamma(4.0, 0.5, size=batch_size) 

    # --- Relative starting point --- #
    #Hato et al. (2025; speed): M = 0.54, SD = 0.07
    #Hato et al. (2025; accuracy): M = 0.54, SD = 0.06
    mean_zr = 0.5
    std_zr = 0.2
    clip_a_zr = 0.2
    clip_b_zr = 0.8
    a_zr, b_zr = (clip_a_zr - mean_zr) / std_zr, (clip_b_zr - mean_zr) / std_zr
    zr = stats.truncnorm.rvs(loc=mean_zr, scale=std_zr, a=a_zr, b=b_zr, size=batch_size)

    # --- Drift rate --- #
    #Voss et al. (2019; multi-speed): 
    ##Difficult stimuli: M = 1.97, SD = 0.50
    ##Easy stimuli: M = 1.03, SD = 0.29
    #Hato et al. (2025; speed): M = -1.37 (1.31), SD = 0.45 (0.39)
    #Hato et al. (2025; accuracy): M = -1.41 (1.54), SD = 0.45 (0.33)
    mean_v = 1.5
    std_v = 0.75
    clip_a_v = 0.01
    clip_b_v = 18.51
    a_v, b_v = (clip_a_v - mean_v) / std_v, (clip_b_v - mean_v) / std_v
    v1 = stats.truncnorm.rvs(loc=mean_v, scale=std_v, a=a_v, b=b_v, size=batch_size)
    v0 = stats.truncnorm.rvs(loc=mean_v*-1, scale=std_v, a=b_v*-1, b=a_v*-1, size=batch_size)

    # --- Stability parameter --- #
    #Voss et al. (2019; multi-speed): M = 1.73, SD = 0.36
    #Hato et al. (2025; speed): M = 1.48, SD = 0.15
    #Hato et al. (2025; accuracy): M = 1.78, SD = 0.13
    if fix_alpha:  
        alpha = np.repeat(2.0, repeats=batch_size)
    else:  # LFM
        mean_alpha = 1.7
        std_alpha = 0.6
        clip_a_alpha = 1.0
        clip_b_alpha = 2.0
        a_alpha, b_alpha = (clip_a_alpha - mean_alpha) / std_alpha, (clip_b_alpha - mean_alpha) / std_alpha
        alpha = stats.truncnorm.rvs(loc=mean_alpha, scale=std_alpha, a=a_alpha, b=b_alpha, size=batch_size)

    # --- Intertrial variability of drift --- #
    #Voss et al. (2019; multi-speed): M = 0.99, SD = 0.58
    #Hato et al. (2025; speed): 0.48, SD = 0.23
    #Hato et al. (2025; accuracy): M = 0.99, SD = 0.30
    #bis 2
    mean_sv = 1.0
    std_sv = 0.75
    clip_a_sv = 0.0
    clip_b_sv = 3.45
    a_sv, b_sv = (clip_a_sv - mean_sv) / std_sv, (clip_b_sv - mean_sv) / std_sv
    sv = stats.truncnorm.rvs(loc=mean_sv, scale=std_sv, a=a_sv, b=b_sv, size=batch_size)

    # --- Intertrial variability of relative starting point --- #
    #Voss et al. (2019; multi-speed): M = 0.47, SD = 0.19
    #Hato et al. (2025; speed): M = 0.14, SD = 0.07
    #Hato et al. (2025; accuracy): M = 0.12, SD = 0.02
    szr = np.random.uniform(0.0, clip_a_zr*2, size=batch_size)
        
    p_samples = np.c_[t0, st0, zr, v0, v1, a, alpha, szr, sv]
    return p_samples.astype(np.float32)

@jit(nopython=True)
def generate_condition_matrix(num_obs, num_conditions=2):
    """Draws a design matrix for each simulation in a batch."""
    obs_per_condition = int(np.ceil(num_obs / num_conditions))
    condition = np.arange(num_conditions)
    condition = np.repeat(condition, obs_per_condition)
    return condition[:num_obs]

# For a single trial
@jit(nopython=True)
def diffusion_trial(v, a, ndt, sndt, zr, alpha, 
                    szr, sv,
                    deadline):    
    n_steps = 0.
    dt = 0.001
    max_steps = 10000
    
    ndt = ndt - 0.5*sndt + sndt * np.random.uniform(0, 1) 
    zr = zr - 0.5*szr + szr * np.random.uniform(0, 1) 
    v = v + sv * np.random.normal()
    x = a * zr

    while (x > 0 and x < a and n_steps < max_steps):
        x += v*dt + levy_noise(alpha, dt)
        n_steps += 1.0

    rt = n_steps * dt
    rt = rt + ndt if x > 0. else -rt - ndt
    
    if deadline:     
        if abs(rt) > 0.900:
                rt = 0
    return rt

# For an entire subject
@jit(nopython=True)
def ddm_simulator(theta, n_obs, deadline):
    design_matrix = generate_condition_matrix(n_obs, 2)
    
    v = theta[3:5]

    out = np.zeros(n_obs)
    for n in range(n_obs):
        out[n] = diffusion_trial(v[design_matrix[n]], a=theta[5], ndt=theta[0],sndt=theta[1], zr=theta[2], alpha=theta[6], szr=theta[7], sv=theta[8], deadline=deadline)

    rts = np.expand_dims(out, 1)
    condition_labels = design_matrix.reshape(-1, 1)
    rts_with_labels = np.hstack((rts, condition_labels))

    return rts_with_labels

def batch_simulator(prior_samples, deadline=False, n_obs=None): 
    if not n_obs: # i.e., for parameter estimation
        n_obs = random_num_obs()
    
    n_sim = prior_samples.shape[0]

    sim_data = np.empty((n_sim, n_obs, 2), dtype=np.float32)

    for i in range(n_sim):
        sim_data[i] = ddm_simulator(prior_samples[i], n_obs, deadline)
    
    return sim_data

def random_num_obs(min_obs=299, max_obs=352):
    """Draws a random number of observations for all simulations in a batch."""
    return np.random.randint(low=min_obs, high=max_obs + 1)