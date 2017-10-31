#!/usr/bin/python
#Quick and dirty macrospin flip flop implementation since Mathematica is very slow
#Coupling three MTJs. Code origins from spice.py got extended to mA_mQ.py which handles two MTJ and got further expanded to three MTJs in this file
#All quantities are in SI units
#Thomas Windbacher, 3.5.2017

#Required libraries and functionalities
import argparse
import os
import io
import time
import datetime
import math  as ma
import numpy as np
from numpy import linalg as la
from scipy.integrate import ode
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab


######################################################
#Import parameters from the commandline
######################################################
parser = argparse.ArgumentParser()
parser.add_argument('-Temperature', nargs='?', const=1, default= 0., type=float, help = 'Temperature used for the thermal field in Kelvin. Default is 0.K')
parser.add_argument('-Current', nargs='?', const=1, default=0., type=float, help="Applied current in Ampere. Default is 0.A.")
parser.add_argument('-Exchange', nargs='?', const=1, default=2.e-11, type=float, help="Effective exchange constant in J/m. Default is 2e-11 J/m.")
parser.add_argument("-Contact_Width", nargs='?', const=1, default=3.e-08, type=float, help="Contact width in m. Default is 3.e-8m.")
parser.add_argument("-Layer_Depth", nargs='?', default=3.e-08, type=float, help="Layer depth in m. Default is 3.e-8m.")
parser.add_argument("-Connection_Length", nargs='?', const=1, default=6.e-08, type=float, help="Distance between contacts in m. Default is 6.e-8m.")
parser.add_argument("-Layer_Thickness", nargs='?', const=1, default=3.e-09, type=float, help="Thickness of the magnetic layer. Default is 3.e-9m.")
parser.add_argument("-Distance", nargs='?', const=1, default=4.5e-08, type=float, help="Distance between centers of neighbouring boxes. Default is 4.5e-8m.")
parser.add_argument("-t0", nargs='?', const=1, default=0., type=float, help="Simulation start time in seconds. Default is 0.")
parser.add_argument("-t1", nargs='?', const=1, default=0., type=float, help="Time when current pulse starts in seconds. Default is 0.")
parser.add_argument("-t2", nargs='?', const=1, default=2.e-09, type=float, help="Time when current pulse ends in seconds. Default is 2.e-9s")
parser.add_argument("-tend", nargs='?', const=1, default=2.e-09, type=float, help="Time when simulation stops in seconds. Default is 2.e-9s")
parser.add_argument("-Plot", nargs='?', const=1, default=True, type=int, help="Plot the results after the simulation is finished. Default is True")
parser.add_argument("-Verbose", nargs='?', const=1, default=False, type=int, help="Plots the current progress of the simulation and the time elapsed up to now. Default is False")
parser.add_argument("-FileName", nargs='?', const=1, default='test', help="Filename for data dump. Default is test.crv")
args = parser.parse_args()

#Debug - Quick check
#print (args)



#start measuring run time
start_time = time.clock()

######################################################
#Constants and parameters
######################################################
kb    =  1.38064852e-23;  # J/K
hbar  =  1.054571800e-34; # Js
qe    =  1.60217662e-19;  # As
gamma =  2.11e5; # m/(As)
mu0   =  4.*ma.pi*1.e-7;  #Vs/Am
#Temperatur
Temp = args.Temperature;# 300.; #Kelvin

#Fixed time step dt 
dt    =  1.4875e-14;   #s
#Magnetic material properties
alpha =  0.01; # 1
#Uniaxial anisotropy
K1    =  1.e5; #1.e5 J/m^3
K1vec =  np.array([0., 0., 1.]);# 1
#Exchange constant
#Aexch =  4.0e-10;#for CoPt 1.4e-10; # J/m
Aexch  =  args.Exchange
#Magnetization saturation
Ms    = 4.e5; # A/m 

####################################################
#Parameters for spin-transfer torque calculation
####################################################

#Orientation of polarization vector (must be normalized)
s        = np.array([0.,0.,1.]);
sA       = np.array([0.,0.,1.]);
sB       = np.array([0.,0.,1.]);
#Aplied current in Ampere
#I        = 0.; 
#IA       = -7.0e-3;#2.e-5; 
#IB       =  7.0e-3;#-2.e-5; 
IA       = -1.*args.Current
IB       = args.Current 
#Polarization
p        = 0.3;# for copper 0.9; # for oxides
pA       = 0.3;# for copper 0.9; # for oxides
pB       = 0.3;# for copper 0.9; # for oxides
#Field like torque relative strength
epsilon  = 0.1
epsilonA = 0.1
epsilonB = 0.1

#####################################################
#Geometry related calculations
#####################################################
#a  = 3.e-08
#b  = 6.e-08
#c  = 3.e-09
a  = args.Contact_Width
ad = args.Layer_Depth
b  = args.Connection_Length
c  = args.Layer_Thickness

VA = a*ad*c
VQ = b*ad*c
VB = VA
V  = VA + VQ + VB

#Distance between MTJ_A and MTJ_Q
#d  = 4.5e-08;# 30nmx30nmx3nm boxes
d  = args.Distance

#####################################################
#Boundary conditions 
#####################################################

#By employing spherical coordinates |m|=1 is ensured
#Starting position of magnetization
# Values are same as for OOMMF studies
phi    = 0.
theta  = 0.175
phiB   = 0.
thetaB = 0.175
m0     = np.array([ma.cos(phi)*ma.sin(theta),ma.sin(phi)*ma.sin(theta),ma.cos(theta)])
m0A    = np.array([ma.cos(phi)*ma.sin(theta),ma.sin(phi)*ma.sin(theta),ma.cos(theta)])
m0Q    = np.array([ma.cos(phi)*ma.sin(theta),ma.sin(phi)*ma.sin(theta),ma.cos(theta)])
m0B    = np.array([ma.cos(phiB)*ma.sin(thetaB),ma.sin(phiB)*ma.sin(thetaB),ma.cos(thetaB)])
mAQB0  = np.hstack((m0A,m0Q,m0B))

#######################################################
#Parameters for data dump
#######################################################
time_step = 1.e-12; # after each time_step current data will be written
filename  = args.FileName + '.crv'

#Simulation time triggers
######################################################################
#       +------+
#       |      |
#       |      |
#-+-----+      +---------------------+
# t0    t1     t2                    tend
######################################################################
t0    = args.t0;   # 0.;    # start simulation
t1    = args.t1;   # 0.;    # start current pulse
t2    = args.t2;   # 2.e-8; # end current pulse
tend  = args.tend; # 2.e-8; # end simulation

#Initialization of arrays with start values
sol   = np.array([mAQB0])
t     = np.array([t0])

#Prefactor for RHS of equation
prefactor = -1.*gamma/(1+alpha*alpha)


######################################################################
#Define functions for effective field calculations
######################################################################
def Dz (a, b, c) :
 "Demagnetization factor along z-axis"
 return 1./ma.pi*( 
        ( a*a*a + b*b*b - 2.*c*c*c ) / (3.*a*b*c) + 
        c*( ma.hypot(a,c) + ma.hypot(b,c))/(a*b)  + 
        ((a*a + b*b - 2.*c*c)*ma.sqrt(a*a + b*b + c*c))/(3.*a*b*c) -
        (ma.pow(a*a + b*b,3./2.) + ma.pow(a*a + c*c, 3./2.) + ma.pow(b*b + c*c,3./2.))/(3.*a*b*c) +
        ((b*b - c*c)*ma.log((ma.sqrt(a*a + b*b + c*c) - a)/(ma.sqrt(a*a + b*b + c*c) + a)))/(2.*b*c) + 
        ((a*a - c*c)*ma.log((ma.sqrt(a*a + b*b + c*c) - b)/(ma.sqrt(a*a + b*b + c*c) + b)))/(2.*a*c) + 
        2.*ma.atan((a*b)/(c*ma.sqrt(a*a + b*b + c*c))) + 
        b*ma.log( ( ma.sqrt(a*a + b*b) + a) / (ma.sqrt(a*a + b*b) - a) ) / (2.*c) + 
        a*ma.log( ( ma.sqrt(a*a + b*b) + b) / (ma.sqrt(a*a + b*b) - b) ) / (2.*c) +
        c*ma.log( ( ma.sqrt(a*a + c*c) - a) / (ma.sqrt(a*a + c*c) + a) ) / (2.*b) +
        c*ma.log( ( ma.sqrt(b*b + c*c) - b) / (ma.sqrt(b*b + c*c) + b) ) / (2.*a)
        );

def Dx (x, y, z) :
 "Demagnetization factor along x-axis"
 return Dz(y,z,x);

def Dy (x, y, z) :
 "Demagnetization factor along y-axis"
 return Dz(z,x,y);

def Huni (m, axis, Ms, K) :
 "Definition of the uniaxial anisotropy field"
 return  (2.*K)/(mu0*Ms)*(m*axis)*axis;

def Hdem (m, Ddem, Ms) :
 "Demagnetization field"
 return -Ms*m*Ddem;

def Hstray (m, r, Ms, V) :
 "Stray/Dipole field from neighbor boxes"
 return Ms*V*(3.*(m*r).sum()*r - (r*r).sum()*m)/ma.pow((r*r).sum(),5./2.);

def Hexch (m1, m2, r, A, Ms) :
 "Exchange field"
 return (-2.*A)/(mu0*Ms)*(m1 - m2)/ma.pow(r,2.);

#Definition of thermal field -> Finocchio,J.Appl.Phys.99 doi:10.1063/1.2177049
def Htherm(T, alpha, gamma, V, Ms, dt) :
 "Thermal field according to Finocchio JAPP 99, 10.1063/1.2177049" 
 return ma.sqrt((2.*kb*T*alpha)/(gamma*V*mu0*Ms*dt*(1.+alpha*alpha)))*np.array(np.random.normal(0.,1.,3));

#Gets far to complicated as soon as you have to take care of three connected MTJs.
#Therefore, it is now taken care of where the actual LLG equations are assambled. --> f()
#Defining the effective field term
#def Heff(m, Ms, N, K1vec, K1,alpha,gamma,V,T,dt,mn,d,A,r1,r2):
# "Effective field containing all physical contributions"
# return Huni(m, K1vec, Ms, K1) + Hdem(m, N, Ms) + Htherm(T,alpha,gamma,V,Ms,dt) + Hexch(m,mn,d,A,Ms) + Hstray(m,r1,Ms,V) + Hstray(m,r2,Ms,V) 


#Definition of the spin transfer torque terms

def gox (p, m, s) :
 "Torque angle dependence for oxides"
 return p/(2. * ( 1.+p*p*(m*s).sum() ) );

def gmet (p, m, s) :
 "Torque angle dependence for metals"
 return 1./(-4.+ma.pow(1.+p,3.)*(3.+ (m*s).sum())/(4.*ma.pow(p,3./2.)));

def tau (m, s, I, p, V, Ms, gamma, epsilon, g) :
 "Torque term"
 mxs   = np.cross(m,s);
 mxmxs = np.cross(m,mxs);
 return (hbar*I)/(Ms*V*qe*mu0)*g*( mxmxs - epsilon*mxs );

#Define right hand side of the ODE
def f(t,m,prefactor,t1,t2,Ms,NA,NQ,NB,K1vec,K1,alpha,gamma,VA,VQ,VB,Temp,dt,pA,pB,sA,sB,epsilonA,epsilonB,IA,IB,d,A):
 "RHS of LLG-ODE"
 #Split m into two 3x1 normalized vectors
 mA= m[0:3]/la.norm(m[0:3])
 mQ= m[3:6]/la.norm(m[3:6])
 mB= m[6:9]/la.norm(m[6:9])
 #Calculations for MTJ_A
 heffA      = Huni(mA, K1vec, Ms, K1) + Hdem(mA, NA, Ms) + Htherm(Temp,alpha,gamma,VA,Ms,dt) + Hexch(mA,mQ,d,A,Ms) + Hstray(mQ,np.array([-1.*d,0.,0.]),Ms,VA) + Hstray(mB,np.array([-2.*d,0.,0.]),Ms,VA)
 precesionA = np.cross(mA,heffA)
 dampingA   = alpha*np.cross(mA,precesionA)
 if (t>= t1 and t<=t2):
  sttA       = tau(mA,sA,IA,pA,VA,Ms,gamma,epsilonA,gmet(pA,mA,sA))
 else:
  sttA = np.array([0.,0.,0.])
 rhsA       = precesionA + dampingA + sttA

 #Calculations for MTJ_Q
 heffQ      = Huni(mQ, K1vec, Ms, K1) + Hdem(mQ, NQ, Ms) + Htherm(Temp,alpha,gamma,VQ,Ms,dt) + Hexch(mQ,mA,d,A,Ms) + Hexch(mQ,mB,d,A,Ms) + Hstray(mA,np.array([1.*d,0.,0.]),Ms,VQ) + Hstray(mB,np.array([-1.*d,0.,0.]),Ms,VQ)
 precesionQ = np.cross(mQ,heffQ)
 dampingQ   = alpha*np.cross(mQ,precesionQ)
 rhsQ       = precesionQ + dampingQ;# + sttQ
 
 #Calculations for MTJ_B
 heffB      = Huni(mB, K1vec, Ms, K1) + Hdem(mB, NB, Ms) + Htherm(Temp,alpha,gamma,VB,Ms,dt) + Hexch(mB,mQ,d,A,Ms) + Hstray(mQ,np.array([1.*d,0.,0.]),Ms,VB) + Hstray(mA,np.array([2.*d,0.,0.]),Ms,VB)
 precesionB = np.cross(mB,heffB)
 dampingB   = alpha*np.cross(mB,precesionB)
 if (t>= t1 and t<=t2):
  sttB       = tau(mB,sB,IB,pB,VB,Ms,gamma,epsilonB,gmet(pB,mB,sB))
 else:
  sttB = np.array([0.,0.,0.])
 rhsB       = precesionB + dampingB + sttB
 
 #merge mA, mQ and mB again for sending back
 rhs         = np.hstack((rhsA,rhsQ,rhsB))
 return prefactor*rhs

# Geometry dependent but constant.
# It is sufficient to calculate only once before the integration
# The formular for the prisma defines that if you enter a length it will range from -l <= axis <= +l  (same for x, y and z) --> Therefore everything has to be divided by 2.  
N = np.array([Dx(a/2.,b/2.,c/2.),Dy(a/2.,b/2.,c/2.),Dz(a/2.,b/2.,c/2.)])
NA = np.array([Dx(a/2.,ad/2.,c/2.),Dy(a/2.,ad/2.,c/2.),Dz(a/2.,ad/2.,c/2.)])
NQ = np.array([Dx(b/2.,ad/2.,c/2.),Dy(b/2.,ad/2.,c/2.),Dz(b/2.,ad/2.,c/2.)])
NB = np.array([Dx(a/2.,ad/2.,c/2.),Dy(a/2.,ad/2.,c/2.),Dz(a/2.,ad/2.,c/2.)])

###################################################################
#Functions for data manipulation and export
###################################################################
def create_header(comment=''):
 'Creates the header information for .crv files'
 header  = '##STT_MacroSpin model \n'
 header += '##Contact: Thomas Windbacher (t.windbacher(at)gmail.com)\n'
 if comment :
  header += '##'+comment+'\n'
 header += '##p 4\n'
 header += '#n t mx_A my_A mz_A mx_Q my_Q mz_Q mx_B my_B mz_B\n'
 header += '#u s 1 1 1 1 1 1 1 1 1\n'
 return header

def write_data(filename,header,step,t,y):
 'Function to dump the simulation data into a .crv file'
 dt = step;# define threshold when data should be dumped
 with open(filename,'w') as f:
  f.write(header)
  for i in range(0,t.size):
   if ( t[i] > dt):
    f.write( str(t[i])+'  ')
    for l in y[i,:]:
     f.write(str(l)+'  ')
    dt += step;# increment threshold by step to pick the next point to write 
    f.write('\n')
 f.close()
 return 0. 

def write_header(filename):
 'Function to write header into a .crv file'
 with open(filename,'w') as f:
  f.write(create_header())
 f.close() 
 return f 

def append_data(filename, t, y):
 'Function to append the simulation data to a .crv file'
 with open(filename,'a') as f:
  f.write( str(t) + '  ' )
  for l in y[0,:]:
   f.write(str(l)+'  ')
  f.write( '\n' )
 f.close() 
 return 0. 
##########################################################################################
##Main section
##########################################################################################

#Set up ode solver with Runge Kutta method
#atol : float or sequence absolute tolerance for solution
#rtol : float or sequence relative tolerance for solution
#nsteps : int Maximum number of (internally defined) steps allowed during one call to the solver.
#first_step : float
#max_step : float
#safety : float Safety factor on new step selection (default 0.9)
#ifactor : float
#dfactor : float Maximum factor to increase/decrease step size by in one step
#beta : float Beta parameter for stabilised step size control.
# verbosity : int Switch for printing messages (< 0 for no messages).
#######################################################################################################
##Parameters::
##prefactor,Ms,NA,NQ,NB,K1vec,K1,alpha,gamma,VA,VQ,VB,Temp,dt,pA,pB,sA,sB,epsilonA,epsilonB,IA,IB,d,A
######################################################################################################
arguments = (prefactor,t1,t2,Ms,NA,NQ,NB,K1vec,K1,alpha,gamma,VA,VQ,VB,Temp,dt,pA,pB,sA,sB,epsilonA,epsilonB,IA,IB,d,Aexch)
r = ode(f).set_integrator('dopri5',first_step=dt,max_step=dt,nsteps=1e6,atol=1.e-4)
r.set_initial_value(mAQB0, t0).set_f_params(*arguments)


if args.Verbose == True:
   print("\n-----------------------------------------------------", end="\n" )
   print(" Simulation time | Total progress | Total run time", end="\n" )
   print("-----------------------------------------------------", end="\n" )

#Set initial trigger for output
next_step = r.t + time_step;
#Write header
write_header(filename)

while r.successful() and r.t <= tend:
      r.integrate(r.t+dt)
      if  r.t > next_step:
        if args.Verbose == True:
           print("   %3.6e       %3.3f%%        %s " % (t[-1], t[-1]/tend*100.,datetime.timedelta(seconds=(time.clock() - start_time))), end="\r" )
        next_step += time_step; # update trigger for output
        #Unindent to write every time step into sol
        sol  = np.append(sol,np.array([r.y]),axis=0)
        t    = np.append(t,r.t)
        append_data( filename, r.t, np.array([r.y]))


if args.Verbose == True:
   print("   %3.6e       %3.3f%%        %s " % (t[-1], t[-1]/tend*100.,datetime.timedelta(seconds=(time.clock() - start_time))), end="\n" )

#Write simulation data
#write_data('test_AQB'+str(time.strftime("%Y%d%m%H%M%S", time.localtime()))+'.crv',create_header(),1.e-12,t,sol)
#write_data( filename,create_header(),time_step, t, sol)



##########################################################################################
##Visualization of simulation results
##########################################################################################
if args.Plot:
 f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
 ax1.set_title(r'Normalized magnetization $\vec{\rm{m}}_{\rm{A}}$, $\vec{\rm{m}}_{\rm{Q}}$, and $\vec{\rm{m}}_{\rm{B}}$ as a function of time')

 #ax1.ylabel('MTJ_{A}')
 ax1.grid(True)
 ax1.set_ylabel(r'$\rm{MTJ}_{\rm{A}}$ (1)')
 ax1.plot(t, sol[:,0],label='$m_{x}$')
 ax1.plot(t, sol[:,1],label='$m_{y}$')
 ax1.plot(t, sol[:,2],label='$m_{z}$')
 ax1.legend()

 ax2.grid(True)
 ax2.set_ylabel(r'$\rm{MTJ}_{\rm{Q}}$ (1)')
 ax2.plot(t, sol[:,3],label='$m_{x}$')
 ax2.plot(t, sol[:,4],label='$m_{y}$')
 ax2.plot(t, sol[:,5],label='$m_{z}$')
 ax2.legend()

 ax3.grid(True)
 ax3.set_ylabel(r'$\rm{MTJ}_{\rm{B}}$ (1)')
 ax3.plot(t, sol[:,6],label='$m_{x}$')
 ax3.plot(t, sol[:,7],label='$m_{y}$')
 ax3.plot(t, sol[:,8],label='$m_{z}$')
 ax3.legend()

 ax3.set_xlabel('time (s)')

 f.subplots_adjust(hspace=0)
 plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

 #put it on the display
 plt.show()


############################################################################################Scratchpad and debug section
#########################################################################################

