import numpy
from scipy.special import spherical_jn
from IPython.display import clear_output
import time
from scipy import integrate
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import os

def helm_form_factor(q2, A, Z):
    """
    Input:
    q2 = Charge squared

    Important Variables:
    A = atomic mass number, 183.84 for Tungsten (W), and 55.845 for Iron (Fe)
    s = Lorents Invarient Quantity
    R1 = Sets the charge distribution in the nucleus
    j1 = First Spherical Bessel Function of the first kind

    Output:
    F = Helm Form Factor
    """
    #A = 55.845 #For Iron
    s = 0.9 #fm
    R1 = numpy.sqrt((1.23*A**(1/3) - 0.6)**2 + (7/3)*numpy.pi**2 * 0.52**2 - 5* s**2) / (2.99792458e23*6.582119569e-25) #GeV
    j1 = spherical_jn(1,numpy.sqrt(q2)*R1)

    F = 3*j1/(numpy.sqrt(q2)*R1) * numpy.exp(-(numpy.sqrt(q2)*(s/ (2.99792458e23*6.582119569e-25)))**2/(2))
    return F

def tsai_form_factor(q2, A, Z):
    t = q2
    inelastic1 = 1.9276
    inelastic2 = 1.40845
    aval = 111.0/(0.0005111*numpy.power(Z,1./3.))
    apval = 773.0/(0.0005111*numpy.power(Z,2./3.))
    dval = 0.164/numpy.power(A,2./3.)

    return (1./Z) * ((Z**2*(aval**2*t/(1+aval**2*t))**2*(1/(1+t/dval))**2)+Z*(apval**2*t/(1+apval**2*t))**2*((1+t*inelastic1)/(1+t*inelastic2)**4))**0.5

def form_factor(q2, A, Z):
    #return helm_form_factor(q2, A, Z)
    return tsai_form_factor(q2, A, Z)

def Energy(M, m1, m2):
    """
    Computes energies of daughter particles in a two-body decay
    Args:
        M = Mass of the decaying particle
        m1 = mass of one of the daughter particles
        m2 = mass of the other daughter particle

    Returns:
        E1 = Energy of one of the daughter particles with mass m1
        E2 = Energy of the other daughter particle with mass m2
    """
    E1 = (M/2) * (1 + (m1**2/M**2) - (m2**2/M**2))
    E2 = (M/2) * (1 + (m2**2/M**2) - (m1**2/M**2))
    return E1, E2

def Lorentz_Matrix(lorentz_factor, v):
    """
    ### This produces matrices for particles traveling in any direction ###
    Input:
    lorentz_factor = The Lorentz Factor of Lorentz transformations (the gamma)
    v = Velocity of the decaying particle

    Output:
    transform_matirx = Lorentz Transformation Matrix, to go from Lab to Rest frame of the partcile
    inverse_matrix = Inverse Lorentz transformation matrix, to go from rest to lab frame
    """
    vx = v[0]
    vy = v[1]
    vz = v[2]
    vabs = numpy.sqrt(sum(v*v))
    g = lorentz_factor #For ease of typing
    transform_matrix = numpy.array([[g    , -g*vx                   , -g*vy                 , -g*vz], \
                                 [-g*vx, 1+(g-1)*(vx*vx/vabs**2) ,  (g-1)*(vx*vy/vabs**2),  (g-1)*(vx*vz/vabs**2)], \
                                 [-g*vy,   (g-1)*(vy*vx/vabs**2) ,1+(g-1)*(vy*vy/vabs**2),  (g-1)*(vy*vz/vabs**2)], \
                                 [-g*vz,   (g-1)*(vz*vx/vabs**2) ,  (g-1)*(vz*vy/vabs**2),1+(g-1)*(vz*vz/vabs**2)]])


    inverse_matrix = numpy.linalg.inv(transform_matrix)

    return transform_matrix, inverse_matrix

def find_Lorentz_Factor(M,p):
    """
    Compute the Lorentz factor gamma E/m
    Args:
        M = Mass of the decaying particle
        p = 4-momenta of the decaying particle in the Lab frame
    Returns:
        gamma = The Lorentz Factor
    """
    """
    Ep = p[0] # Energy of the Particle
    v = p[1:]/Ep #Velocity of the Particle (In the Rest Frame)
    vabs = numpy.sqrt(sum(v*v)) #Magnitude of the v vector
    pabs = numpy.sqrt(sum(p[1:]*p[1:])) #Magnitude of the p3 vector
    gamma = M/(Ep - vabs*pabs) #Lorentz Factor
    return gamma
    """
    return p[0]/M

def generate_random_angles():
    """
    Randomly samples two-body decay angles
    Args:
    Returns:
        phi = Azimuth angle of event
        theta = Zenith angle of event
    """
    phi = numpy.random.uniform(0,2*numpy.pi)
    costheta = numpy.random.uniform(-1,1)
    theta = numpy.arccos(costheta)
    return phi,theta

def generate_random_time(gamma, tau):
    """
    Randomly samples from the decay time distribution exp(-t/tau)/tau
    Args:
        gamma = Lorentz Factor
        tau = Lifetime of the axion-like particle

    Returns:
        t = Decay time of the particle
    """
    u = numpy.random.uniform(0,1) #Generate a random number in a uniform distribution
    t = -tau*gamma*numpy.log(1-u) #Transform the number so it fits a (1/(tau*gamma)) * exp(-t/(tau*gamma)) distribution
    return t

def find_daughter_momenta_4(M, p, tau, m1, m2):
    """
    Generates decay products of a two-body decay
    Args:
        M = Mass of the decaying particle
        p = 4-momenta vector of the decaying particle
        m1 = mass of one of the daughter particles
        m2 = mass of the other daughter particle
        theta = Theta angle from the MC integration
        phi = Phi angle from the MC integration

    Returns:
        k1 = 4 momenta vector, [Energy, 3D k1 vector]
        k2 = 4 momenta vector, [Energy, 3D k2 vector]
    """
    phi, theta = generate_random_angles()

    E1, E2 = Energy(M,m1,m2) #Calculate energies of daughter particles
    k1_mag = numpy.sqrt(E1**2 - m1**2) #Calculate magnitude of k1

    k1_3 = k1_mag*numpy.array([numpy.sin(theta)*numpy.cos(phi),numpy.sin(theta)*numpy.sin(phi),numpy.cos(theta)]) #Creating 3D momentum Vector
    k2_3 = -1*k1_3   #Because vec{k2} = -vec{k1}

    k1 = numpy.insert(k1_3,0,E1) #Create 4-momenta vectors
    k2 = numpy.insert(k2_3,0,E2)

    Ep = p[0] # Energy of the Particle
    v = p[1:]/Ep # Calculate Rest frame Velocity of the Decaying particle

    gamma = find_Lorentz_Factor(M,p) #Find the Lorentz Factor

    transform_matrix, inverse_matrix = Lorentz_Matrix(gamma, v) #Find transformation matrices

    k1 = numpy.dot(inverse_matrix,k1) #Inverse Lorentz transform back to the lab frame
    k2 = numpy.dot(inverse_matrix,k2)

    return k1,k2

def find_4_position(M, p, tau):
    """
    Randomly generates a decay four position for a particle
    Args:
        M = Mass of the Decaying Particle
        p = 4-momenta of the Decaying Particle in the lab frame
        tau = Lifetime of the axion-like particle

    Returns:
        x = displacement of the particle from creation to decay
    """
    gamma = find_Lorentz_Factor(M,p) #Find the lorentz factor
    decay_time = generate_random_time(gamma, tau) #Generate decay time
    v = p[1:]/(gamma*M) #Find lab frame velocity of the particle
    x3 = v*decay_time #Calculate the displacement
    x = numpy.array([decay_time, x3[0], x3[1], x3[2]])
    return x

def t_distribution(t, ma, Mn, A, Z, Egamma):
    """
    Primakoff differential cross-section dsigma/dt with various numerical constants stripped off
    Args:
        t = Mandelstam t
        ma = Mass of the outgoing Axion
        Egamma = Energy of the incoming Photon
        Mn = nuclear mass
        A = atomic number
        Z = nuclear charge
    Returns:
        disigmadt = distribution of t
    """

    s = s_term(Mn,Egamma)
    F = form_factor(abs(t), A, Z)

    dsigmadt = 1/(s-Mn**2)**2 * (F**2)/(t**2) * (ma**2 *t *(Mn**2 + s) - ma**4 * Mn**2 - t*((Mn**2-s)**2 + s*t))
    return dsigmadt

def s_term(Mn,Egamma):
    """
    Mandelstam s for the lab-frame scattering of a photon on a nucleus
    Args:
        Mn = Mass of the target nucleus
        Egamma = Energy of the Incoming Photon

    Returns:
        s = Mandelstam s
    """
    s = Mn**2 + 2*Mn*Egamma
    return s

def t_bounds(ma, Mn, Egamma):
    """
    Kinematic boundaries for the Mandelstam t variable in the Primakoff process gamma + N > ALP + N
    Args:
        ma : ALP mass
        Mn : nucleus mass
        Egamma : photon energy
    """

    s = s_term(Mn,Egamma)
    pgcm = (s-Mn**2) / (2*numpy.sqrt(s))
    pacm = numpy.sqrt( ((s + ma**2 - Mn**2)/(2*numpy.sqrt(s)))**2 - ma**2)
    t0 = ma**4/(4*s) - (pgcm - pacm)**2
    t1 = ma**4/(4*s) - (pgcm + pacm)**2
    return t1, t0

#Function 12
def generate_t(ma, Mn, A, Z, Egamma, small_t_cut_over_t0 = 100):
    """
    Samples the Mandelstam t parameter from the Primakoff differential distribution using naive accept/reject method
    Args:
        ma = Mass of the axion-like particle
        Egamma = Energy of the Incoming Photon

    Returns:
        t = randomly generated Mandelstam t
    """
    s = s_term(Mn,Egamma)
    t1, t0 = t_bounds(ma, Mn, Egamma)

    t_max = (-2*ma**4 *Mn**2) / (-ma**2 * Mn**2 + Mn**4 - ma**2 *s -2 * Mn**2 * s + s**2)
    c = t_distribution(t_max, ma, Mn, A, Z, Egamma)*1.1
    while True:
        # the integrand is peaked close to 0, so avoid generating t far from the peak
        t = numpy.random.uniform(small_t_cut_over_t0*t0,t0)
        ft = t_distribution(t, ma, Mn, A, Z, Egamma)
        u = numpy.random.uniform(0,1)
        if ft > c:
            c = ft
            continue
        if ft > u*c:
            return t

def ALP_energy(Egamma, mN, t):
    """
    Egamma = Energy of the incoming Photon
    mN = Mass of the Nucleus
    t = Lorentz Invarient, Mandelstam Variable

    Output:
    Ea = Energy of the outgoing ALP
    """
    Ea = (t+2*mN*Egamma)/(2*mN)
    return Ea

def ngamma(pgamma):
    """
    Input:
    pgamma = 3 vector momentum of the incoming photon

    Output:
    ng = unit vector in the direction of the incoming photon
    """
    pgabs = numpy.linalg.norm(pgamma) 
    ng = pgamma/pgabs
    return ng

def ALP_parallel(mN, ma, Ea, pabs, Egamma, pgamma):
    """
    Computes the component of the outgoing ALP momentum parallel to the incoming photon momentum
    Args:
        ma = Mass of the ALP
        Ea = Energy of the Alp
        pabs = Magnitude of the ALP's momenta
        Egamma = Energy of the incoming photon
        pgamma = 3 vector momentum of the incoming photon

    Returns:
        p_parallel = Parallel Component of the ALP's three momentum
    """

    # cosine of the angle between the photon and the ALP
    cos_theta = (2*Ea*Egamma - 2*mN*(Egamma- Ea) - ma**2) / (2*Egamma*pabs) 
    p_parallel = cos_theta*pabs*ngamma(pgamma)
    return p_parallel

def Gram_Schmidt_Process(ngamma):
    """
    Inputs:
    ngamma = unit vector in the direction of the incoming photon

    Outputs:
    n1, n2 = Orthonormal basis vectors, perpendicular to pgamma
    """
    x2 = numpy.array([1, 0, 0]) #ngamma, x1, & x3 are our starting Vectors
    x3 = numpy.array([0, 1, 0])
    v2 = x2 - ( numpy.dot(x2, ngamma)/numpy.dot(ngamma,ngamma) )*ngamma  #Product Orthogonal Vectors with the Gram Schmidt Process
    v3 = x3 - ( numpy.dot(x3, ngamma)/numpy.dot(ngamma,ngamma) )*ngamma - ( numpy.dot(x3, v2)/numpy.dot(v2,v2) )*v2

    n1 = v2 / (numpy.sqrt(sum(v2**2))) #Normalize the vectors to make Orthonormal Vectors
    n2 = v3 / (numpy.sqrt(sum(v3**2)))
    return n1,n2

def ALP_perpendicular(pabs, p_parallel, pgamma):
    """
    Computes the component of the outgoing ALP momentum perpendicular to the incoming photon momentum
    Args:
        pabs = Magnitude of the ALP's momenta
        p_parallel = 3 vector of the parallel component of the ALP momentum
        pgamma = 3 vector momentum of the incoming photon

    Returns:
        p_perpendicular = Perpendicular Component of the ALP's momentum
    """
    p_perpabs_sq = pabs**2 - numpy.dot(p_parallel,p_parallel)
    
    if numpy.fabs(p_perpabs_sq) < 1e-8:
        return numpy.zeros(3)

    assert p_perpabs_sq >= 0.

    p_perpabs = numpy.sqrt(p_perpabs_sq)

    phi = numpy.random.uniform(0,2*numpy.pi)
    n1, n2 = Gram_Schmidt_Process(ngamma(pgamma))

    p_perpendicular = p_perpabs * (numpy.cos(phi)*n1 + numpy.sin(phi)*n2)

    return p_perpendicular

def ALP_momentum_from_t(ma, mN, Egamma, pgamma, t):
    """
    Construct an ALP 4 momentum provided a value of the Mandelstam t in the gamma + N > ALP + N scattering 
    Args:
        ma = Mass of the Axion-like Particle
        mN = Mass of the Nucleus
        Egamma = Energy of the incoming Photon
        pgamma = 3 vector momentum of the incoming photon

    Returns:
        pa = 4 momenta of the outgoing ALP
    """
    Ea = ALP_energy(Egamma, mN, t)

    pabs = numpy.sqrt(Ea**2 - ma**2)

    p_para = ALP_parallel(mN, ma, Ea, pabs, Egamma, pgamma)
    p_perp = ALP_perpendicular(pabs, p_para, pgamma)

    p3 = p_para + p_perp

    pa = numpy.array([Ea, p3[0], p3[1], p3[2]])
    return pa

def ALP_momenta(ma, mN, A, Z, Egamma, pgamma, small_t_cut_over_t0 = 100):
    """
    Samples an ALP four-momentum from the Primakoff differential distribution
    Args:
        ma = Mass of the Axion-like Particle
        mN = Mass of the Nucleus
        Egamma = Energy of the incoming Photon
        pgamma = 3 vector momentum of the incoming photon

    Returns:
        pa = 4 momenta of the outgoing ALP
    """
    t = generate_t(ma, mN, A, Z, Egamma, small_t_cut_over_t0)

    return ALP_momentum_from_t(ma, mN, Egamma, pgamma, t)

def generate_event(p_gamma, ma, mN, A, Z, tau, small_t_cut_over_t0 = 100):
    """
    Generates a single Primakoff (gamma + N > ALP + N) event for a photon with a given initial momentum
    Args:
        p_gamma = 4-momentum of the incoming photon
        ma : Mass of the produced ALP
        mN : mass of the target nucleus
        tau : Lifetime of the particle
        small_t_cut_over_t0: smallest Mandelstam t to consider relative to the kinematic cutoff t0. Default 100. i.e. t 
            values are smaples in the range [t0*small_t_cut_over_t0, t0]
    
    Returns: 
        pa = four-momentum of the ALP
        k1, k2 = 4-momenta of the 2 daughter photons of the axion decay
        x = decay four-position of the ALP

    """
    ### Scattering ###
    Egamma = p_gamma[0]
    pgamma = p_gamma[1:]
    pa = ALP_momenta(ma, mN, A, Z, Egamma, pgamma, small_t_cut_over_t0)

    ### Decay ###
    k1, k2 = find_daughter_momenta_4(ma, pa, tau, m1=0, m2=0)
    x = find_4_position(ma, pa, tau)

    return pa, k1, k2, x


def nucleus_4_momenta(mN, pgamma, pa):
    """
    Reconstructs the scattered nucleus momentum for the two body collision gamma + N > N + ALP
    Args:
        mN = Mass of the target nucleus
        pgamma = 3-vector of the incoming photon
        pa = 3-vector of the outgoing ALP
    Returns:
        pN_prime = 4-momenta of the Nucleus after the interaction and creation of the ALP
    """
    pN3 = numpy.zeros(3)
    pN_prime3 = (pN3 + pgamma) - pa
    pN_prime_abs = numpy.linalg.norm(pN_prime3)
    EN = numpy.sqrt(mN**2 + pN_prime_abs**2 )
    pN_prime = numpy.array([EN, pN_prime3[0],pN_prime3[1],pN_prime3[2]])
    return pN_prime


def generate_primakoff_events(photons, ma, mN, A, Z, tau, small_t_cut_over_t0 = 100, print_output = False):
    """
    Generates Primakoff (gamma + N > ALP + N)  events from a list of incoming photons
    Args:
        photons: list of incoming photon 4-momenta, shape (N,4)
        ma : Mass of the produced ALP
        mN : mass of the target nucleus
        tau : Lifetime of the particle
        small_t_cut_over_t0: smallest Mandelstam t to consider relative to the kinematic cutoff t0. Default 100. i.e. t 
            values are smaples in the range [t0*small_t_cut_over_t0, t0]
    
    Returns:
        output = (N',6,4) array containing a list of events that produced an ALP. N' can be less than N if not 
        all photons have enough energy to produce an ALP. Each event is a list of five four-momenta for the 
        incoming photon, ALP, scattered nucleus, ALP decay photon1, ALP decay photon2. The last entry in 
        each event is the decay four-position generated based on the lifetime tau.
    """
    output = []

    start = time.time()
    for i in range(len(photons)):
        
        event = numpy.zeros((6,4))
    
        if photons[i][0] < ma + (ma**2)/(2*mN):
            continue
        
        event[0] = photons[i]
        pa, k1, k2, x = generate_event(photons[i], ma, mN, A, Z,tau, small_t_cut_over_t0)
    
        pN = nucleus_4_momenta(mN, photons[i][1:], pa[1:])
    
        event[1] = pa
        event[2] = pN
        event[3] = k1
        event[4] = k2
        event[5] = x
        
        output.append(event)
        if print_output:
            print(i)
            clear_output(wait=True)
    
    end = time.time()
    if print_output:
        print("This took me", (end-start)/60, "minutes to process", len(photons), "events")
    
    return numpy.array(output)

def parallel_helper(params, photon):
    ma = params['ma']
    mN = params['mN']
    A = params['A']
    Z = params['Z']
    tau = params['tau']
    small_t_cut_over_t0 = params['small_t_cut_over_t0']

    event = numpy.zeros((6,4))

    if photon[0] < ma + (ma**2)/(2*mN):
        return np.array([photon])
    
    event[0] = photon
    pa, k1, k2, x = generate_event(photon, ma, mN, A, Z,tau, small_t_cut_over_t0)

    pN = nucleus_4_momenta(mN, photon[1:], pa[1:])

    event[1] = pa
    event[2] = pN
    event[3] = k1
    event[4] = k2
    event[5] = x

    return event

def generate_primakoff_events_in_parallel(photons, ma, mN, A, Z, tau, small_t_cut_over_t0 = 100, print_output = False, cpu_count=os.cpu_count(), chunksize=1):
    """
    Generates Primakoff (gamma + N > ALP + N)  events from a list of incoming photons
    Args:
        photons: list of incoming photon 4-momenta, shape (N,4)
        ma : Mass of the produced ALP
        mN : mass of the target nucleus
        tau : Lifetime of the particle
        small_t_cut_over_t0: smallest Mandelstam t to consider relative to the kinematic cutoff t0. Default 100. i.e. t 
            values are smaples in the range [t0*small_t_cut_over_t0, t0]
    
    Returns:
        output = (N',6,4) array containing a list of events that produced an ALP. N' can be less than N if not 
        all photons have enough energy to produce an ALP. Each event is a list of five four-momenta for the 
        incoming photon, ALP, scattered nucleus, ALP decay photon1, ALP decay photon2. The last entry in 
        each event is the decay four-position generated based on the lifetime tau.
    """
    output = []

    start = time.time()

    params = {}
    params['ma'] = ma 
    params['mN'] = mN 
    params['A'] = A
    params['Z'] = Z 
    params['tau'] = tau
    params['small_t_cut_over_t0'] = small_t_cut_over_t0 

    pool = Pool(cpu_count)
    for out in tqdm(pool.imap_unordered(func=partial(parallel_helper, params), iterable=photons, chunksize=chunksize), total=len(photons)):
        output.append(out)
    pool.close()
    pool.join()
    
    return numpy.array(output)


