
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
import random
import time 

class modular:
	def __init__(self,constant , vec):
		self.constant=constant
		self.vec=vec
	def get_m(self,subset):
		if subset.size==0:
			return self.constant
		# print '***',subset.astype(int),'****'
		return self.constant+self.vec[subset.astype(int)].sum()
	def get_m_singleton(self,ground_set):
		tmp=np.zeros(ground_set.shape[0])
		for i in range(ground_set.shape[0]):
			tmp[i]=self.get_m(np.array([int(ground_set[i])]))
		return tmp


class SubMod:
	def __init__(self):
		pass

		
class triage_human_machine:
	def __init__(self,data_dict):
		self.X=data_dict['X']
		self.Y=data_dict['Y']
		self.test=data_dict['test']
		self.dist_mat=data_dict['dist_mat']
		self.c=np.square(data_dict['human_pred']-data_dict['Y'])
		self.c_te=np.square(self.test['human_pred']-self.test['Y'])
		self.dim=self.X.shape[1]
		self.n=self.X.shape[0]
		# print 'n',self.n
		self.V=np.arange(self.n)
		self.epsilon=float(1)
		self.BIG_VALUE=100000


	def get_c(self,subset):
		return np.array([int(i) for i in self.V if i not in subset])	
	
	def get_minus(self,subset,elm):
		return np.array([i for i in subset if i != elm] )

	def get_added(self,subset,elm):
		return np.concatenate((subset,np.array([int(elm)])),axis=0)
	

	def elm_mat(self,elm,s):
		if s=='f':
			v= np.hstack(( np.array([self.Y[elm]]) , self.X[elm] ))	
			return v.reshape(self.dim+1,1).dot(v.reshape(1,self.dim+1))
		if s == 'g':
			return self.X[elm].reshape(self.dim,1).dot(self.X[elm].reshape(1,self.dim))

	def addend(self,l,s,subset=None):
		if s == 'f':
			arr=np.eye(self.dim+1)
			for i in range(self.dim+1):
				if i == 0 :
					if l==0 :
						arr[0,0]=0
					else:
						arr[0,0]=self.c[subset].sum()
				else:
					arr[i,i]=self.lamb*(self.n-l)
		if s == 'g':
			arr=self.lamb*(self.n-l)*np.eye(self.dim)
		return arr

	def modular_upper_bound(self,subset):
		l_subset=subset.shape[0]
		Y_X=np.concatenate((self.Y.reshape(1,self.n),self.X.T),axis=0)
		subset_c=self.get_c(subset)
		Y_X_sub = Y_X[:,subset_c]
		A=Y_X_sub.dot(Y_X_sub.T)
		B=self.addend(l_subset, 'f',subset )
		f_subset=np.log( LA.det(A+B))
		f_inc=np.zeros(self.n)
		for elm in subset:
			buffer_new = A + self.elm_mat(elm,'f') + self.addend( l_subset-1, 'f',self.get_minus(subset,elm) )
			f_inc[elm]=f_subset - np.log( LA.det(buffer_new) )
			
			
		A=Y_X.dot(Y_X.T) 
		B=self.addend(0, 'f',np.array([]) )
		f_null=np.log(LA.det(A+B))

		for elm in subset_c:
			buffer_new=A - self.elm_mat(elm,'f') + self.addend(1,'f',np.array([elm]))
			f_inc[elm]=np.log(LA.det(buffer_new)) - f_null
 
 
		if subset.size==0:
			m_f = modular(f_subset, f_inc)
		else:
			m_f = modular(f_subset - f_inc[subset].sum(),f_inc)
		
		# plt.plot(m_f)
		# plt.show()
		return m_f

	def find_max_elm(self,ground_set,m):
		g=np.zeros(ground_set.shape[0])
		A=self.X.T.dot(self.X)
		B = self.addend(1, 'g' )
		for elm,idx in zip(ground_set,range(ground_set.shape[0])):
			g[idx]=np.log(LA.det(  B + A - self.elm_mat(elm, 'g')))

		# plt.plot(g-m)
		# plt.title('g-m')
		# plt.show()

		ind=np.argmax(g-m.get_m_singleton(ground_set)) 
		print 'selected',ground_set[ind]
		# time.sleep(1000)
		return ground_set[ind]

	
	def g_S(self,subset):
		
		l_subset=subset.shape[0]
		subset_c=self.get_c(subset)
		X_sub = self.X[subset_c].T
		A=X_sub.dot(X_sub.T)
		B = self.addend(l_subset,'g')
		g_subset = np.log( LA.det(A+B)) 		
	
		B_elm = self.addend(l_subset-1,'g')
		g_S=[]
		for elm in subset:			
			g_S.append(np.log(LA.det( A + self.elm_mat(elm,'g') + B_elm )))
		return g_subset,np.array([g_S]).flatten()	
		
		
	def check_delete(self,subset,m,approx): 
	
		if subset.size==0:
			print 'subset empty'
			return False,subset
			
		g_subset,g_S_arr = self.g_S(subset)
		g_m_subset = g_subset -m.get_m(subset)

		g_S_m=np.zeros(subset.shape[0])
		for i in range(subset.shape[0]):
			
			g_S_m[i]=g_S_arr[i]- m.get_m(self.get_minus(subset,subset[i]))  #  -(m[subset].sum() - m[subset] ) 

		print '----subset,value'
		print subset, g_m_subset
		
		print '---g-m(S/i) for i \in S---'
		print g_S_m

		# time.sleep(10000)


	
		if np.max(g_S_m) >= g_m_subset*approx:
			item_to_del=subset[np.argmax(g_S_m)]
			subset_sel = self.get_minus(subset,item_to_del)
			print '----Following item is deleted---',item_to_del
			print 'now subset---> ', subset_sel
			
			return True, subset_sel
		print 'now subset---> ', subset
		return False, subset


	def check_exchange_greedy(self,subset,m,ground_set,approx,K):
		
		
		l_subset=subset.shape[0]
		subset_c=self.get_c(subset)
		X_sub = self.X[subset_c].T
		A=X_sub.dot(X_sub.T)
		B=self.addend( l_subset,'g')
		if subset.size == 0:
			g_m_subset=np.log( LA.det(A+B))
		else:
			g_m_subset=np.log( LA.det(A+B))  - m.get_m(subset) # m[subset].sum()
		
		if subset.shape[0]<K:
			subset_with_null = np.hstack((subset,np.array([-1]))) # ch
		else:
			subset_with_null=subset

			
		print 'g_m_subset---------------->',g_m_subset

		# declare 
		subset_c_gr=np.array([ i for i in ground_set if i not in subset])
		g_m_exchange = np.zeros((subset_with_null.shape[0],subset_c_gr.shape[0]))
		for e,row_ind in zip(subset_with_null,range(subset_with_null.shape[0])) : 
			for d,col_ind in zip(subset_c_gr,range(subset_c_gr.shape[0])):
				if e == -1:		
					g_part=np.log(LA.det(A-self.elm_mat(d,'g')+self.addend(l_subset+1,'g')  ))
					m_part= m.get_m(self.get_added(subset,int(d)))
				else:
					g_part=np.log(LA.det(A+self.elm_mat(e,'g')-self.elm_mat(d,'g')+B))
					m_part = m.get_m( self.get_added( self.get_minus(subset, e) , int(d) ) )
				g_m_exchange[row_ind][col_ind]=g_part - m_part
		# display
		print '---------g_m_exchange---------------'
		print g_m_exchange
		print '-------------------------------------'

		# plt.plot(g_m_exchange.flatten())
		# plt.title('g_m_exchange')
		# plt.show()

		if np.max(g_m_exchange) > g_m_subset*approx:
			r,c = np.unravel_index(np.argmax(g_m_exchange,axis=None),g_m_exchange.shape)
			print 'index of max element ',r,c
			e=subset_with_null[r]
			d=subset_c_gr[c]
			print e,' is exchanged with ',d
			if e == -1:
				subset_with_null[r]=d
				return True,subset_with_null
			else:
				ind_e = np.where( subset == e)[0]
				print 'subset',subset
				subset[ind_e]=d
				print '-----------------------'
				print subset
				print '-----------------------'
				return True, subset
		return False,subset
	

	def approx_local_search(self,m,K,ground_set):
		# max_A (g-m)(A) given |A|<=k  	implementing local search by J.Lee 2009 STOC
		approx=1+self.epsilon/float(self.n**4)

		print '--- max elm of g-m --------'
		curr_subset=np.array([self.find_max_elm(ground_set,m)]) 
		while True:
			print ' ---   Delete ----- '
			flag_delete,curr_subset = self.check_delete(curr_subset,m,approx) 
			if flag_delete==False:
				print ' --- Exchange ---- '
				flag_exchange,curr_subset = self.check_exchange_greedy(curr_subset,m,ground_set,approx,K) 
				# time.sleep(100000)
				if flag_exchange==False:
					break
				else:
					print 'exchange done'
			else:
				print 'deletion done'
			print '-------------------------------------------------'
			print '-------------------------------------------------'
			print curr_subset
			print '-------------------------------------------------'
			print '-------------------------------------------------'
		return curr_subset

	def eval_g_m(self,subset,m):
		subset_c=self.get_c(subset)
		X_sub = self.X[subset_c].T
		A=X_sub.dot(X_sub.T)
		B=self.addend(subset.shape[0],'g')
		g_subset = np.log( LA.det(A+B))
		return g_subset - m.get_m(subset)
		
	def constr_submod_max(self,m,K):
		
		ground_set=self.V
		print '----- local search 1 '
		subset_1=self.approx_local_search(m,K,ground_set)
		ground_set=self.get_c(subset_1)
		print '----- local search 2 '
		subset_2=self.approx_local_search(m,K,ground_set)
		if self.eval_g_m(subset_1,m)> self.eval_g_m(subset_2,m):  
			return subset_1
		else:
			return subset_2

	def sel_subset_diff_submod(self):	
		# solve difference of submodular functions
		subset_old=np.array([])

		itr=0
		while True :
			print '-------------------------------Iter ', itr, '  ---------------------------------------'
			print 'modular upper bound '
			m_f=self.modular_upper_bound(subset_old) 
			print 'constr submodular max '
			subset=self.constr_submod_max(m_f,self.K) 
			print '--------OLD-------------------'
			print subset_old
			print '---------New-------------------'
			print subset
			if set(subset) == set(subset_old) :
				return subset
			else:
				subset_old=subset
			itr += 1
		
	def set_param(self,lamb,K):
		self.lamb=lamb 
		self.K=K 
	
	def get_optimal_pred(self,subset):
		subset_c=self.get_c(subset)
		X_sub=self.X[subset_c].T
		Y_sub=self.Y[subset_c]
		subset_l=self.n-subset.shape[0]
		return LA.inv( self.lamb*subset_l*np.eye(self.dim) + X_sub.dot(X_sub.T) ).dot(X_sub.dot(Y_sub))
		
	def plot_subset(self,w,subset,K):
		x=self.X[subset]
		y=self.Y[subset]
		plt.scatter(x,y,c='red',label='machine')
		
		c_subset = np.array([i for i in self.V if i not in subset])
		x=self.X[c_subset]
		y=self.Y[c_subset]
		plt.scatter(x,y,c='blue',label='human')
		
		x=self.X
		y=np.hstack(( self.X, np.ones((self.n,1)) )).dot(w.reshape(2,1))
		plt.scatter(x,y,c='black',label='prediction')		
		plt.legend()
		plt.grid(True)
		plt.ylim([-2,2])
		plt.title('Fraction of sample to human'+str(1-K) )
		plt.show()		

	def algorithmic_triage(self,param):
		# print 'check',K#int(K*self.n)
		self.set_param(param['lamb'],int(param['K']*self.n))
		subset_for_machine  = self.sel_subset_diff_submod() 
		w_m = self.get_optimal_pred(subset_for_machine)
		# self.plot_subset(w_m,subset_for_machine,param['K']) 
		# res_dict=self.get_avg_accuracy(w_m, subset_for_machine,num_nbr) #
		return {} # res_dict ***************************

			
	# def get_avg_accuracy(self,w,subset,nbr):
		
	# 	predict=(self.Y[subset]-self.X[subset].dot(w)).flatten()
	# 	error = ( predict.dot(predict) + self.c.sum()-self.c[subset].sum())/self.n
	# 	subset_te=[]
	# 	for dist in self.dist_mat:
	# 		indices = np.argsort(dist)[:nbr]
	# 		dist_elm= dist[indices]
	# 		indicator = np.array([1 if i in subset else -1 for i in indices])
	# 		if dist_elm.dot(indicator) > 0 :
	# 			subset_te.append(1)
	# 		else:
	# 			subset_te.append(0)
	# 	subset_te=np.array(subset_te,dtype=bool)

	# 	predict_te=(self.test['Y'][ subset_te ] -self.test['X'][ subset_te ].dot(w)).flatten()
	# 	error_te = (predict_te.dot(predict_te) + self.c_te.sum()-self.c_te[subset_te].sum())/self.test['Y'].shape[0]


	# 	res={'avg_train_err':error,'avg_test_err':error_te}
	# 	return res 


	# def check_exchange_incremental(self,subset,m,ground_set,approx,K):
		
		
	# 	l_subset=subset.shape[0]
	# 	subset_c=self.get_c(subset)
	# 	X_sub = self.X[subset_c].T
	# 	A=X_sub.dot(X_sub.T)
	# 	B=self.addend( l_subset,'g')
	# 	if subset.size == 0:
	# 		g_m_subset=np.log( LA.det(A+B))
	# 	else:
	# 		g_m_subset=np.log( LA.det(A+B))  - m[subset].sum()
		
	# 	if subset.shape[0]<K:
	# 		subset_with_null = np.hstack((subset,np.array([-1])))
			
	# 	print 'g_m_subset---------------->',g_m_subset
	# 	for e in subset_with_null : 
	# 		for d in [ i for i in ground_set if i not in subset]:
	# 			if e == -1:		
	# 				g_part=np.log(LA.det(A-self.elm_mat(d,'g')+self.addend(l_subset+1,'g')  ))
	# 				if subset.size==0:
	# 					m_part = m[d]
	# 				else:
	# 					m_part=m[subset].sum()+m[d]
	# 			else:
	# 				g_part=np.log(LA.det(A+self.elm_mat(e,'g')-self.elm_mat(d,'g')+B))
	# 				if subset.size==0:
	# 					m_part = m[subset].sum()-m[e]+m[d]
	# 			g_m_new = g_part+m_part
	# 			print 'g_m_new----------------->',g_m_new
	# 			if g_m_new >= g_m_subset: # *approx: *******************
	# 				print 'e------------->',e,', d--------------> ',d
					
	# 				if e == -1:
						
	# 					return  np.array(( subset , np.array([d]) ))
	# 				else:
						
	# 					ind_e = np.where( subset == e)[0]
	# 					print 'subset',subset
	# 					subset[ind_e]=d
	# 					print '-----------------------'
	# 					print subset
	# 					print '-----------------------'
	# 					return True, subset
					
	# 	return False,subset