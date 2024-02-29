import numpy as np 
import mpi4py 
from mpi4py import MPI 
# this is the code of 1 process.
# MPI variables for the communications inter-processes 
comm = MPI.COMM_WORLD 
NbP = comm.Get_size() # in command mpirun -np NbP
Me  = comm.Get_rank() 
prev=(Me-1)%NbP
next=(Me+1)%NbP
# Local variables of the process for its computations 
n = 4096 
V = np.empty(n,np.float64) 
M = np.empty([n,n],np.float64) 
Vout = np.zeros(n,np.float64) 
Norm = np.zeros(1,np.float64) 
Res  = np.zeros(1,np.float64)
filepath=None
routines=['Sr_replace','Bsend','Ssend']
routine=routines[0]

# STEP 0: Init of local V and M arrays from data files
if filepath is None: 
    np.random.seed(Me)
    V=np.random.rand(V.size) #generate locally on each process. Not modified after.
    M=np.random.rand(M.size)
else:
    with open(filepath,'r') as file: #just a template
        L=file.readlines()
        V=L[0][Me]
        M=L[1][Me]

#Both direction works
#Step 1
if routine=='Sr_replace':
    for i in range(NbP):                                                         
        # Local computations 
        Vout = np.matmul(M,V) 
        Norm[0] = np.linalg.norm(Vout) 
        Res[0] += Norm[0] 
        # Data circulation (of V array)
        
        comm.Sendrecv_replace(V,dest=next,source=prev)  #V is used as both send and receive buffer. overwrited. #two ops in one. Inner workings Depends on the distribution
        
        print('PE',Me,'received',V)

    print("PE",Me,": end")

if routine=='Bsend':
    buff=MPI.Alloc_mem((V.size+MPI.BSEND_OVERHEAD)*1) #we are passing V of size n.
    #buff1=MPI.Alloc_mem((V.size+MPI.BSEND_OVERHEAD)*NbP): put attach and detach outside of the loop.
    #MPI.Attach_buffer(buff1)
    for i in range(NbP):                                                          
        # Local computations 
        Vout = np.matmul(M,V) 
        Norm[0] = np.linalg.norm(Vout) 
        Res[0] += Norm[0] 
        # Data circulation (of V array)
        MPI.Attach_buffer(buff) # To copy in buff. Configure the bsend to makes copy in buff. Doesn't depends on the behavior of another process.
        comm.Bsend(V,dest=next)#bsend: no mention of buff  Not blocking
        #Caution: small letter: for standard python object.
        comm.Recv(V,source=prev)#overwrite V with the previous one. Blocking (blocking)
        #^  To receive a np array, do not return a np array, and everything as function params
        MPI.Detach_buffer() # is a blocking until the last byte stored has been sent.
        print('PE',Me,'received',V)
    #MPI.Detach_buffer()
    MPI.Free_mem(buff)
    print("PE",Me,": end")


#Inside of the loop : allowcated *1
    #Adv: No memory overflow.
    #Disadv: slower sensitive to the number of bsend used with the same buffer.
#Outside of the loop : allowcated *NbP
    #Adv: (Probably) Faster
    #Disadv: Risk of overflowing, needs to wait for all the data to be sent. So depends of program.
#Ideal solution:
    #Put a buffer for *10, and at each 10 iterations, attach/    at each 9 (something) detach.

    

if routine=='Ssend': #Synchronous
    temprecv=np.empty_like(V)
    for i in range(NbP):                                                          
        # Local computations 
        Vout = np.matmul(M,V) 
        Norm[0] = np.linalg.norm(Vout) 
        Res[0] += Norm[0] 
        # Data circulation (of V array)
        if Me%2==0: #important to alternate or else every1 is sending=> Deadlock
            comm.Ssend(V,dest=next) #ssend is blocking
            comm.Recv(V,source=prev)
        else:
            comm.Recv(temprecv,source=prev)#else V will get overwrite
            comm.Ssend(V,dest=next)
            V,temprecv=temprecv,V #exchange the pointers. No lose in memory/no time spent creating copies.
        print('PE',Me,'received',V)
    print("PE",Me,": end")


comm.gather()


'''

1 process per node
if tc is the calculation time (same for all routine).
ts+q*tw: we assume all can be done in //. Memory bus is avaible to r w in //

Sr_replace
Assumption on hardware= 
H1: comms are 'load balanced'. This is ideal comms scheme, and not saturating. 
H2: send/recv in parallel: Duplex card. Limit in memory/cache/architecture
 NbP*tc+NbP*(ts+q*tw)  bc NbP steps =>
 if q is  big: ts+q*tw ~ q*tw
 if q is small: ts+q*tw ~ ts
==> Important to group data to send. The time take to allocate the data can be < ts

 Bsend, same hypothesis on Sr_replace.
 NbP*(tc)+Nbp*(ts+q*tw) to where to is buffer setup and ops time

 Ssend
 E(NbP/2)*(tc+2*(ts+q*tw))+(E(NbP/2)+1)*(tc+2*(ts+q*tw)  E(x)=interger part of x
          NbP*(tc)+2NbP*(ts+q*tw) each process if doing Ssend: which is blocking. Is longer but impossible to saturate network


2 process in one node with NbP/2 nodes
Sr_replace
NbP*(tc)+NbP*(ts+q*tw)

Bsend
NbP*(tc)+NbP*(ts+q*tw), same as above

Ssend: need to take account the number of machines.
NbP*(tc)+NbP*(ts+q*tw)

'''  

''''
Question 4
10 matrix, 1process/node  Numpy can be // on  4 cores/node

4.1.1: 10 nodes are needed.
4.1.2: mpirun -np 10 (number of process) -map-by python file.py
Install one process /node and give access to all the cores.

4.2 : one process per physical cores
4.2.1: #to do for next time.

Deployement with MPI, need to explain what do we want. (VERY IMPORTANT)

'''