#This project was created by Dr. Prof. Arya Vishala. It is a very enjoyable project and 
# I recommend you try and solve the problem tasks. 

#All function defined below are complete and can used as you please.
import numpy as np

def MRP_to_DCM(sigma):
    def skew_mat(sigma):
        return np.array([[0, -sigma[2], sigma[1]],
                         [sigma[2], 0, -sigma[0]],
                         [-sigma[1], sigma[0], 0]])
    
    dcm = np.eye(3) + (8*skew_mat(sigma)@skew_mat(sigma) - 4*(1-np.inner(sigma,sigma))*skew_mat(sigma)) / (1+np.inner(sigma,sigma))**2

    return dcm

def DCM_to_MRP(dcm):
    taljare = np.trace(dcm) + 1 + 2*np.sqrt(np.trace(dcm)+1)

    s1 = (dcm[1,2]-dcm[2,1])/taljare
    s2 = (dcm[2,0]-dcm[0,2])/taljare
    s3 = (dcm[0,1]-dcm[1,0])/taljare

    return np.array([s1,s2,s3])

def position_and_velocity(r,euler_angle): #Task 1
    mu = 42828.3 #km^3 / sec^2
    theta_dot = np.sqrt(mu / (r**3)) #rad/sec

    def dcm_313(t1,t2,t3): #ON-dcm 
        c3 = np.cos(t3)
        s3 = np.sin(t3)
        c2 = np.cos(t2)
        s2 = np.sin(t2)
        c1 = np.cos(t1)
        s1 = np.sin(t1)

        dcm = np.array([[c3*c1-s3*c2*s1, c3*s1+s3*c2*c1, s3*s2],
                        [-s3*c1-c3*c2*s1, -s3*s1+c3*c2*c1, c3*s2],
                        [s2*s1, -s2*c1, c2]])
        return dcm
    
    
    #get DCM from the euler angles
    OMEGA = euler_angle[0]
    i = euler_angle[1]
    theta = euler_angle[2]

    dcm_ON = dcm_313(OMEGA,i,theta) #orbit frame relative inertial
    dcm_NO = np.transpose(dcm_ON)

    N_r = dcm_NO@np.array([[r],[0],[0]]) #position vector in N-frame
    N_r_dot = dcm_NO@np.array([[0],[theta_dot*r],[0]]) #velocity vector in N-frame

    return N_r.flatten(), N_r_dot.flatten()

def dcm_HN(t): #Task 2 
    r = 400 + 3396.19 #km LMO
    mu = 42828.3 #km^3 / sec^2
    theta_dot = np.sqrt(mu / (r**3)) #rad/sec, orbital rate of LMO

    intial_euler_LMO = np.deg2rad( np.array([20, 30, 60]) ) #initial euler angles
    theta_dot_vec = np.array([0,0,theta_dot]) #vector form of orbital rate of LMO i Hill-frame

    euler_angs = intial_euler_LMO + t*theta_dot_vec

    r_LM, r_dot_LMO = position_and_velocity(r,euler_angs) #position/velolcity vectors of H-frame relative N-frame in N-frame

    cross_prod = np.cross(r_LM.flatten(),r_dot_LMO.flatten()).reshape(3,1)

    i_r = r_LM/np.linalg.norm(r_LM)
    i_h = cross_prod/np.linalg.norm(cross_prod)
    i_theta = np.cross(i_h.flatten(),i_r.flatten()).reshape(3,1)

    HN = np.zeros((3,3))
    HN[0,:] = i_r.T
    HN[1,:] = i_theta.T
    HN[2,:] = i_h.T

    return HN

def dcm_RsN(t): #task 3
    dcm = np.array([[-1, 0, 0], #dcm R_s frame relative to N-frame
                    [0, 0, 1],
                    [0, 1, 0]])
    return dcm

def N_omega_RsN(t):
    return np.array([0, 0, 0])


def dcm_RnN(t): #task 4
    def dcm_RnH(): #task 4 dcm of R_n-frame relative to H-frame
        dcm = np.array([[-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]])
        return dcm
    
    dcm = dcm_RnH()@dcm_HN(t)

    return dcm

def N_omega_Rn_N(t): #task 4
    r = 400 + 3396.19 #km LMO
    mu = 42828.3 #km^3 / sec^2
    theta_dot = np.sqrt(mu / (r**3)) #rad/sec, orbital rate of LMO

    omega_Rn_N = np.transpose(dcm_HN(t))@np.array([[0],[0],[theta_dot]]) #angular velocity of R_n relative to N in N-frame
    
    return omega_Rn_N.flatten()

def dcm_RcN(t): #Task 5
    r_LMO = 400 + 3396.19 #km LMO
    r_GMO = 20424.2 #km

    mu = 42828.3 #km^3 / sec^2
    theta_dot_LMO = np.sqrt(mu / (r_LMO**3)) #rad/sec, orbital rate of LMO
    theta_dot_GMO = np.sqrt(mu / (r_GMO**3)) #rad/sec, orbital rate of GMO

    intial_euler_LMO = np.deg2rad( np.array([20, 30, 60]) )
    intial_euler_GMO = np.deg2rad( np.array([0, 0, 250]) )

    curr_euler_LMO = intial_euler_LMO + np.array([0,0,theta_dot_LMO])*t
    curr_euler_GMO = intial_euler_GMO + np.array([0,0,theta_dot_GMO])*t


   
    N_r_LMO = position_and_velocity(r_LMO,curr_euler_LMO)[0]
    N_r_GMO = position_and_velocity(r_GMO, curr_euler_GMO)[0]
    
    delta_r = N_r_GMO - N_r_LMO
    
    r1 = (-1*delta_r/np.linalg.norm(delta_r)).reshape(3,1)


    cross_prod = np.cross(delta_r,np.array([0,0,1])).reshape(3,1)
    r2 = cross_prod/np.linalg.norm(cross_prod)
    r3 = np.cross(r1.flatten(),r2.flatten()).reshape(3,1)

    dcm = np.zeros((3,3))

    dcm[0,:] = r1.T
    dcm[1,:] = r2.T
    dcm[2,:] = r3.T

    return dcm

def N_omega_RcN(t): #task 5 finding the angular veloctiy

    dt = 0.0001
    RcN_2 = dcm_RcN(t+dt)
    RcN_1 = dcm_RcN(t-dt)

    RcN_dot = (RcN_2 - RcN_1)/(2*dt)
    RcN = dcm_RcN(t)

    skew_omega = -RcN_dot@np.linalg.inv(RcN)

    w1 = skew_omega[2,1]
    w2 = skew_omega[0,2]
    w3 = skew_omega[1,0] 

    omega = np.array([w1,w2,w3]).reshape(3,1) #omega expressed in Rc-frame

    N_omega = RcN.T@omega


    return N_omega.flatten()

def Attitude_error(t,sig,B_w_bn,refenece_frame,n_W_rn): #task 6
    BN_dcm = MRP_to_DCM(sig)
    RN_dcm = refenece_frame(t)

    BR = BN_dcm@RN_dcm.T #error dcm

    error_sigma = DCM_to_MRP(BR) #attitude between body and refernece frame

    B_omega_BR = B_w_bn.reshape(3,1) - BN_dcm@(n_W_rn(t).reshape(3,1))
    return error_sigma, B_omega_BR.flatten()

def Attitude_Simulation(t): #task 7

    def f(X,u): # derivative of state-vector
        def skew_mat(x): #function that return skew-symmetric matrix
            return np.array([[0, -x[2], x[1]],
                             [x[2], 0, -x[0]],
                             [-x[1], x[0], 0]])
        
        def MRP_derivative(sig,ome): #Funcation calculated attitude time derivative
            s1 = sig[0]
            s2 = sig[1]
            s3 = sig[2]
            s_square = s1**2 + s2**2 + s3**2
            B = np.array([[1-s_square+2*s1**2,2*(s1*s2-s3),2*(s1*s3+s2)],
                  [2*(s1*s2+s3),1-s_square+2*s2**2,2*(s2*s3-s1)],
                  [2*(s1*s3-s2),2*(s2*s3+s1),1-s_square+2*s3**2]])
    
            sigma_dot = (1/4)*B@ome.reshape(3,1)

            return sigma_dot.flatten()
        
        def w_dot(w,u): #omega, control vector. Funcation calculated omega time derivative
            B_I = np.array([[10,0,0],
                            [0,5,0],
                            [0,0,7.5]]) #inertial matrix in B-frame 
            
            omega_dot = np.linalg.inv(B_I) @ (-skew_mat(w)@B_I@w.reshape(3,1) + u.reshape(3,1))

            return omega_dot.flatten()

        sigma = X[0,:] # attitude
        omega = X[1,:] # angular velocity 

        sigma_dot = MRP_derivative(sigma,omega) #time derivative of attitude
        omega_dot = w_dot(omega,u) #time derivative of angular velocity

        
        X_dot = np.array([sigma_dot,omega_dot])  #derivative of state vector
        return X_dot

    def sigma_norm(s):
        return np.sqrt(s[0]**2+s[1]**2+s[2]**2)

    B_omega_BN_t0 = np.deg2rad(np.array([1.00,1.75,-2.20])) #initial angular velocity in body frame
    sigma_BN_t0 = np.array([0.3,-0.4,0.5]) #initial attitude of body frame relative inertial frame

    X_0 = np.array([sigma_BN_t0,B_omega_BN_t0]) #first row attitude, second angular velocity

    t_max = t
    delta_t = 1 #time step
    t_n = 0 #to keep track of how fram the integration has gone

    r_LMO = 400 + 3396.19 #km, radius of LMO
    r_GMO = 20424.2 #km, radius of GMO
    w_LMO = 0.000884797 #rad/sec, orbital angular velocity of LMO
    w_GMO = 0.0000709003 #rad/sec, orbital angular velocity of GMO

    while t_n < t_max:
        euler_LMO = np.deg2rad( np.array([20, 30, 60]) ) + np.array([0, 0, w_LMO])*t_n #current attitude of LMO orbit 
        euler_GMO = np.deg2rad( np.array([0, 0, 250]) )+ np.array([0, 0, w_GMO])*t_n #current attitude of GMO orbit

        vec_r_LMO, n = position_and_velocity(r_LMO,euler_LMO) #position of LMO
        vec_r_GMO, n = position_and_velocity(r_GMO,euler_GMO) #position of GMO 

        #angle between the LMO and GMO vectors
        theta = np.arccos( np.dot(vec_r_LMO,vec_r_GMO) / ( np.linalg.norm(vec_r_LMO)*np.linalg.norm(vec_r_GMO) ) )

        #control gain selection
        P = 1/6
        K = 1/180

        if vec_r_LMO[1] > 0: #Power mode
            RN = dcm_RsN #set solar-point attitude
            N_w_RN = N_omega_RsN #solar-pointing reference frame velocity
            #tracking error:
            sigma_BR, omega_BR = Attitude_error(t_n,X_0[0,:],X_0[1,:],RN,N_w_RN)
            #control
            u = -K*sigma_BR - P*omega_BR
        elif vec_r_LMO[1] < 0 and np.rad2deg(theta) < 35: #Communication mode
            RN = dcm_RcN #set communication-point attitude
            N_w_RN = N_omega_RcN #communication-point reference frame velocity
            #tracking error:
            sigma_BR, omega_BR = Attitude_error(t_n,X_0[0,:],X_0[1,:],RN,N_w_RN)
            #control
            u = -K*sigma_BR - P*omega_BR
        elif vec_r_LMO[1] < 0 and np.rad2deg(theta) > 35: #Science mode
            RN = dcm_RnN #set nadir-point attitude
            N_w_RN = N_omega_Rn_N #nadir-pointing reference frame velocity
            #tracking error:
            sigma_BR, omega_BR = Attitude_error(t_n,X_0[0,:],X_0[1,:],RN,N_w_RN)
            #control
            u = -K*sigma_BR - P*omega_BR

        k1 = delta_t*f(X_0,u)
        k2 = delta_t*f(X_0+(k1/2),u)
        k3 = delta_t*f(X_0+(k2/2),u)
        k4 = delta_t*f(X_0+k3,u)

        X_0 = X_0 + (1/6)*(k1+2*k2+2*k3+k4)

        if sigma_norm(X_0[0,:]) > 1: # change to shadow set 
            X_0[0,:] = -X_0[0,:]/(sigma_norm(X_0[0,:])**2) 
        
        t_n += delta_t
    
    euler_LMO = np.deg2rad( np.array([20, 30, 60]) ) + np.array([0, 0, w_LMO])*t_n #current attitude of LMO orbit 
    euler_GMO = np.deg2rad( np.array([0, 0, 250]) )+ np.array([0, 0, w_GMO])*t_n #current attitude of GMO orbit

    vec_r_LMO, n = position_and_velocity(r_LMO,euler_LMO) #position of LMO
    vec_r_GMO, n = position_and_velocity(r_GMO,euler_GMO) #position of GMO 

        #angle between the LMO and GMO vectors
    theta = np.arccos( np.dot(vec_r_LMO,vec_r_GMO) / ( np.linalg.norm(vec_r_LMO)*np.linalg.norm(vec_r_GMO) ) )

    if vec_r_LMO[1] > 0: #Power mode
        print('Power-Mode')
    elif vec_r_LMO[1] < 0 and np.rad2deg(theta) < 35: #Communication mode
        print('Communicatio-mode')
    elif vec_r_LMO[1] < 0 and np.rad2deg(theta) > 35: #Science mode
        print('Science-mode')
    
    return X_0[0,:], X_0[1,:] #sigma and omega
