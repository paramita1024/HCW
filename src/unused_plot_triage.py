import sys
import os
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data

class plot_triage:
	def __init__(self,list_of_K,list_of_std, list_of_lamb,list_of_option,curve,option,sigma):
		self.list_of_K=list_of_K
		self.list_of_std=list_of_std
		self.list_of_lamb=list_of_lamb
		self.curve=curve
		self.option=option
		self.sigma=sigma
		self.list_of_option=list_of_option
		
	def extract_result(self,res_file,res_plot_file):
		num_K=len(self.list_of_K)
		res_saved=load_data(res_file)
		res={}
		for std in self.list_of_std:
			res[str(std)]={}
			for lamb in self.list_of_lamb:
				res[str(std)][str(lamb)]=np.zeros((2,num_K))
				for K,idx in zip(self.list_of_K,range(num_K)):
					res[str(std)][str(lamb)][0][idx]=res_saved[str(std)][str(K)][str(lamb)][self.option]['train_res']
					res[str(std)][str(lamb)][1][idx]=res_saved[str(std)][str(K)][str(lamb)][self.option]['test_res']['avg_test_err']
		save(res,res_plot_file)

	def plot_result(self,res_plot_file,image_file_prefix):
		res=load_data(res_plot_file)
		for std in self.list_of_std:
			for lamb in self.list_of_lamb:
				image_file=image_file_prefix+'std'+str(std)+'lamb'+str(lamb)+'.jpg'
				self.plot_performance_vs_K(res[str(std)][str(lamb)],image_file,{'std':std,'lamb':lamb})

	def plot_performance_vs_K(self,res,image_file,param):
		plt.xticks(range(len(self.list_of_K)),self.list_of_K)
		plt.plot(res[0],label='Train Error')
		plt.plot(res[1],label='Test Error')
		plt.title('std of human'+str(param['std'])+',lamb'+str(param['lamb']))
		plt.xlabel('Fraction of Input to Machine')
		plt.ylabel('Average Squared Error')
		plt.grid(True)
		plt.legend()
		plt.savefig(image_file)
		plt.show()

	def plot_subset_selection(self,prefix,res_file):

		res=load_data(res_file)
		for i0,std in zip(range(len(self.list_of_std)),self.list_of_std):
			for i1,K in zip(range(len(self.list_of_K)),self.list_of_K):
				for i2,lamb in zip(range(len(self.list_of_lamb)),self.list_of_lamb):
					obj=res[str(std)][str(K)][str(lamb)][self.option]
					self.plot_obj(obj,prefix,[i0,i1,i2],std,K,lamb)
					K_total = obj['human']['x'].shape[0] 
					self.write_obj_txt( obj, K_total)
					# return 	
		
	def write_obj_txt( self, plt_obj , K_total):
		# print plt_obj.keys()
		for key in ['machine','human','prediction']:
			print key , plt_obj[key]['x'].shape
			n=plt_obj[key]['x'].shape[0]
			tmp = np.vstack(( plt_obj[key]['x'], plt_obj[key]['y'] ))
			file_name = 'Sigmoid_' + str(K_total) + '_' + key + '.txt'
			with open( file_name,'w') as f:
				for i in range(n):
					v = tmp[:,i].flatten()
					if key == 'human':
						f.write(str(v[0]) + ' red '+ str(v[1]) + '\n' )
					if key == 'machine':
						f.write(str(v[0]) + ' blue ' + str(v[1]) + '\n')
					if key == 'prediction':
						f.write(str(v[0]) + ' black ' + str(v[1]) + '\n' )
			
	def plot_obj(self,plt_obj,prefix,ind,std,K,lamb):

		file_plot= prefix+'std_'+str(ind[0])+'_K_'+str(ind[1])+'_lamb_'+str(ind[2])+'.pdf'

		x=plt_obj['human']['x']
		y=plt_obj['human']['y']
		plt.scatter(x,y,c='red',label='human')
		
		x=plt_obj['machine']['x']
		y=plt_obj['machine']['y']
		plt.scatter(x,y,c='blue',label='machine')
		
		x=plt_obj['prediction']['x']
		y=plt_obj['prediction']['y']
		plt.scatter(x,y,c='black',label='prediction')		
		
		plt.legend()
		plt.grid(True)
		plt.title('Std= '+str(std)+', Fraction(K) = '+str(K)+', Lambda='+str(lamb) )
		# plt.tight_layout()
		plt.xlabel('<-----------x------------->')
		plt.ylabel('<-----------y------------->')
		# plt.show()
		# plt.savefig(file_plot,dpi=600, bbox_inches='tight')
		plt.close()

	def print_kile_all(self):

		num_std = len(self.list_of_std)
		num_K=len(self.list_of_K)
		num_lamb=len(self.list_of_lamb)
		print '\\section{Sigmoid}' #$\sigma_d='+str(self.sigma)+'$}'
		print '\\subsection{Vary with $\lambda$}'
		for i in range(num_std):
			std = self.list_of_std[i]
			for j in range(num_K):
				K=self.list_of_K[j]
				print '\\subsubsection{$\sigma=',std,',K=',K,'$}'
				print '\\begin{figure}[H]'	
				for k in range(num_lamb):
					filename= self.curve.replace('.','_')+'_std_'+str(i)+'_K_'+str(j)+'_lamb_'+str(k)
					caption='$\lambda='+str(self.list_of_lamb[k])+'$'
					self.print_figure_singleton(filename,caption)
				print '\\caption{Variation with $\lambda$, DS(GREEDY) }'
				print '\\end{figure}'

	def print_kile_fix_lambda(self):
		num_std = len(self.list_of_std)
		num_K=len(self.list_of_K)
		lamb=0.0001
		k=0
		for i in range(num_std):
			print '\\begin{figure}[ht]'
			for j in range(num_K):
				filename= self.curve.replace('.','_')+'_std_'+str(i)+'_K_'+str(j)+'_lamb_'+str(k)
				caption='K='+str(self.list_of_K[j])
				self.print_figure_singleton(filename,caption)
			print '\\caption{Greedy, Variation with K, (std,lamb)=('+str(self.list_of_std[i])+','+str(lamb)+')}'
			print '\\end{figure}'

	def print_kile_fix_lambda_K(self):

		num_std = len(self.list_of_std)
		K=0.6
		lamb=0.01
		j=2
		k=2
		print '\\begin{figure}'
		for i in range(num_std):
			filename= self.curve.replace('.','_')+'_std_'+str(i)+'_K_'+str(j)+'_lamb_'+str(k)
			caption='std='+str(self.list_of_std[i])
			self.print_figure_singleton(filename,caption)
		print '\\caption{Greedy, Variation with std of human prediction, (K,lamb)=('+str(K)+','+str(lamb)+')}'
		print '\\end{figure}'

	def print_kile_fix_std_lambda(self):

		num_K = len(self.list_of_K)
		std=0.01
		lamb=0.01
		i=0
		k=0
		
		print '\\begin{figure}[H]'
		for j in range(num_K):
			filename= self.curve.replace('.','_')+'_std_'+str(i)+'_K_'+str(j)+'_lamb_'+str(k)
			caption='K='+str(self.list_of_K[j])
			self.print_figure_singleton(filename,caption)
		print '\\caption{Variation with K('+self.option+')}'
		print '\\end{figure}'

	def print_kile_error_VS_K(self,file_name):
		print '\\begin{figure}[H]'
		print '\t \\centering\\includegraphics[width=3cm]{Figure/'+file_name.split('/')[-1].replace('.','_')+'.pdf}'
		caption='Final Average Error Vs K'
		print '\t \\caption{'+caption+'}'
		print '\\end{figure}'

	def print_figure_singleton(self,filename,caption):
		print '\\begin{subfigure}{4cm}'
		print '\t \\centering\\includegraphics[width=3cm]{Figure/'+filename+'.pdf}'
		print '\t \\caption{'+caption+'}'
		print '\\end{subfigure}'

	def get_avg_error_vary_K(self,res_file,data_file,plot_file,std,lamb):
		res=load_data(res_file)
		err_K=[]
		err_K_te=[]
		for K in self.list_of_K:
			err_K.append(res[str(std)][str(K)][str(lamb)][self.option]['train_res'])
			err_K_te.append(res[str(std)][str(K)][str(lamb)][self.option]['test_res']['avg_test_err'])
		if os.path.exists(plot_file+'.pkl'):
			res_plot=load_data(plot_file)
		else:
			res_plot={}
		res_plot[self.option]={'train':err_K,'test':err_K_te}

		print res_plot
		save(res_plot,plot_file)
		
	def get_indices(self,x,x_full):
		x=x.flatten()
		x_full=x_full.flatten()
		return np.array([np.where(x_full == x_elm)[0][0] for x_elm in x ])

	def get_avg_error(self,plt_obj,data_obj,std):

		human_ind = self.get_indices(plt_obj['human']['x'],plt_obj['prediction']['x'])
		human_error = (LA.norm(data_obj.human_pred_train[str(std)][human_ind]-plt_obj['human']['y']))**2
		machine_ind = self.get_indices(plt_obj['machine']['x'],plt_obj['prediction']['x'])
		tmp1=plt_obj['machine']['y']-plt_obj['prediction']['y'][machine_ind]
		tmp2=LA.norm(tmp1)
		machine_error=tmp2**2
		# machine_error = (LA.norm(plt_obj['machine']['y']-plt_obj['prediction'][machine_ind]))**	2
		n=plt_obj['prediction']['y'].flatten().shape[0]
		avg_error=(human_error+machine_error)/n
		return avg_error

	def plot_err_vs_K(self,plot_file,image_file):
		plot_data=load_data(plot_file)
		print type(plot_data['greedy'])
		print plot_data['greedy']
		plt.plot(plot_data['greedy']['train'],label='GREEDY Train',linewidth=8,linestyle='--',marker='o', markersize=10,color='red')
		plt.plot(plot_data['greedy']['test'],label='GREEDY Test',linewidth=8,linestyle='--',marker='o', markersize=10,color='green')
		
		plt.plot(plot_data['diff_submod']['train'],label='DS Train',linewidth=8,linestyle='-',marker='o', markersize=10,color='blue')
		plt.plot(plot_data['diff_submod']['test'],label='DS Test',linewidth=8,linestyle='-',marker='o', markersize=10,color='black')
		# plt.setp(line, linewidth=4,linestyle=ls,marker='o', markersize=10)
		plt.grid()
		plt.legend()
		plt.xlabel('K')
		plt.ylabel('Average Squared Error')
		plt.title('Average Squared Error')
		plt.xticks(range(len(self.list_of_K)),self.list_of_K)
		plt.savefig(image_file+'.pdf',dpi=600, bbox_inches='tight')
		plt.show()

	def get_nearest_human(self,dist,tr_human_ind):
		
		n_tr=dist.shape[0]
		human_dist=float('inf')
		machine_dist=float('inf')
		for d,tr_ind in zip(dist,range(n_tr)):
			if tr_ind in tr_human_ind:
				if d < human_dist:
					human_dist=d
			else:
				if d < machine_dist:
					machine_dist=d 
		return (human_dist -machine_dist)

	def get_test_error(self,res_obj,x,y,y_h,K,dist_mat,plot_file_path):

		w=res_obj['prediction']['w']
		tr_human=res_obj['human']['x']
		tr_all=res_obj['prediction']['x']
		tr_human_ind = self.get_indices(tr_human,tr_all)
		c=(y-y_h)**2
		n=dist_mat.shape[0]
		no_human=int(K*n)
		# print 'no human',no_human
				
		diff_arr=[ self.get_nearest_human(dist,tr_human_ind) for dist in dist_mat]
		# print len(diff_arr)

		indices=np.argsort(np.array(diff_arr))
		human_ind = indices[:no_human]
		machine_ind=indices[no_human:]
		# print human_ind
		# print machine_ind
		# print x.shape
		# print y.shape
		# print x.dot(w).shape
		pred = (y - x.dot(w))**2

		test_err = (pred[machine_ind].sum() + c[human_ind].sum())/ float(n)
		# print '---------------'
	
		# print '---------------'

		plt_obj = {}
		plt_obj['human']={'x':x[human_ind],'y':y[human_ind]}
		plt_obj['machine']={'x':x[machine_ind],'y':y[machine_ind]}
		test_res={}
		test_res['avg_test_err']=test_err
		test_res['plt_obj']=plt_obj

		#self.plot_test_allocation(res_obj,plt_obj,plot_file_path)
		return test_res

	def plot_test_allocation(self,train_obj,test_obj,plot_file_path):

		x=train_obj['human']['x']
		y=train_obj['human']['y']
		plt.scatter(x,y,c='blue',label='train human')

		x=train_obj['machine']['x']
		y=train_obj['machine']['y']
		plt.scatter(x,y,c='green',label='train machine')

		x=test_obj['machine']['x'][:,0].flatten()
		y=test_obj['machine']['y']
		plt.scatter(x,y,c='yellow',label='test machine')

		x=test_obj['human']['x'][:,0].flatten()
		y=test_obj['human']['y']
		plt.scatter(x,y,c='red',label='test human')

		plt.legend()
		plt.grid()
		plt.xlabel('<-----------x------------->')
		plt.ylabel('<-----------y------------->')
		plt.savefig(plot_file_path,dpi=600, bbox_inches='tight')
		plt.close()

		# plt.show()

	def get_train_error(self,plt_obj,y_h,y):

		c=(y_h-y)**2
		y_m=plt_obj['machine']['y']
		x_m=plt_obj['machine']['x']
		w=plt_obj['prediction']['w']
		n=plt_obj['prediction']['x'].shape[0]
		human_ind = self.get_indices(plt_obj['human']['x'],plt_obj['prediction']['x'])
		predict=(y_m-w[1]-w[0]*x_m).flatten()
		error = ( predict.dot(predict) + c[human_ind].sum())/n
		return error

	def compute_test_result(self,res_file,data_file,plot_file):
		data=load_data(data_file)
		res=load_data(res_file)
		x=data.Xtest
		y=data.Ytest
		dist_mat=data.dist_mat
		for std,i0 in zip(self.list_of_std,range(self.list_of_std.shape[0])):
			y_h=data.human_pred_test[str(std)]
			y_h_tr=data.human_pred_train[str(std)]
			for K,i1 in zip(self.list_of_K,range(len(self.list_of_K))):
				#print res[str(std)][str(K)].keys()
				for lamb,i2 in zip(self.list_of_lamb,range(len(self.list_of_lamb))):
					res_obj=res[str(std)][str(K)][str(lamb)][self.option]
					plot_file_path=plot_file+'std_'+str(i0)+'_K_'+str(i1)+'_lamb_'+str(i2)+'.pdf'
					test_res = self.get_test_error(res_obj,x,y,y_h,K,dist_mat,plot_file_path)
					train_res = self.get_train_error(res_obj,y_h_tr,data.Ytrain)
					res[str(std)][str(K)][str(lamb)][self.option]['test_res']=test_res
					res[str(std)][str(K)][str(lamb)][self.option]['train_res']=train_res
		save(res,res_file)

	def split_res_over_K(self,data_file,res_file,unified_K):
		res=load_data(res_file)
		for std in self.list_of_std:
			if str(std) not in res:
				res[str(std)]={}
			for K in self.list_of_K:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in self.list_of_lamb:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					for option in self.list_of_option:
						if option not in res[str(std)][str(K)][str(lamb)]:
							res[str(std)][str(K)][str(lamb)][option]={}
						if K != unified_K:
							res_dict = res[str(std)][str(unified_K)][str(lamb)][option]
							if res_dict:
								res[str(std)][str(K)][str(lamb)][option] = self.get_res_for_subset(data_file,res_dict,lamb,K)
		save(res,res_file)

	def get_optimal_pred(self,data,subset,lamb):
		
		n,dim= data.X.shape
		subset_c=  np.array([int(i) for i in range(n) if i not in subset])	
		X_sub=data.X[subset_c].T
		Y_sub=data.Y[subset_c]
		subset_c_l=n-subset.shape[0]
		return LA.inv( lamb*subset_c_l*np.eye(dim) + X_sub.dot(X_sub.T) ).dot(X_sub.dot(Y_sub))

	def get_res_for_subset(self,data_file,res_dict,lamb,K):
		data=load_data(data_file)
		curr_n = int( data.X.shape[0] * K )
		subset_tr = res_dict['subset'][:curr_n]
		# print '----------------'
		# print subset_tr.shape
		# print '----------------'
		w= self.get_optimal_pred(data,subset_tr,lamb)
		return {'w':w,'subset':subset_tr}
	
def main():

	# Sigmoid
	# list_of_std=np.array([.01,.05,.1,.5])
	# list_of_lamb=[0.01,0.1,0.5,1.0]
	# list_of_lamb=[0.0001,0.001,0.01,0.1]
	# Sigmoid 
	list_of_std =[.01,.05,.1,.5] # [.01,.05,.1,.5, 1] # [[0.001,0.01,0.1][int(sys.argv[1])]]
	list_of_K=[0.1,0.2,0.3,.4,0.5,.6,.7,.8,.9] # [0.2,0.6,0.8]#[0.01,0.1,0.2,0.3,.4,0.5,.6,.7,.8,.9,.99]
	list_of_lamb= [0.01,0.05] # [.1,.5] # [0.0001,0.001,0.01,0.1,.4,.6,.8]#[[int(sys.argv[2])]] # [0.0001,0.001,0.01,0.1] #

	# Gauss
	# list_of_std=np.array([0.01])#01,0.005])#([0.001,.005,0.01,.05,.1])
	# list_of_lamb=[0.01]#[0.0001 ,0.001,0.01,0.1]
	# # list_of_K=[.2,.4,.6,.8]
	# list_of_K=[.2,.4,.6,.8] # [0.01,0.1,0.2,0.3,.4,0.5,.6,.7,.8,.9,.99]
	sigma=2
	# data_curve='sigmoid' # +str(sigma)
	# # data_curve='Gauss_'+str(sigma) 
	curve='sigmoid_n500d10' # _d_s'
	# curve='Gauss_'+str(sigma) # +'_d_s'#_gr' #
	# curve='Gauss_'
	option='greedy' #'diff_submod_greedy'# 'greedy'#'diff_submod_greedy'#'greedy'#
	list_of_option = ['greedy']
	path = '../Synthetic_data/'
	data_file = path + 'data_sigmoid_n500d10'
	res_file = path + 'res_sigmoid_n500d10'
	
	obj=plot_triage(list_of_K,list_of_std, list_of_lamb,list_of_option,curve,option,sigma)
	# plot_path='../Synthetic_data/subset_sel_plots_with_test/'+curve
	# if not os.path.exists(plot_path):
	# 	os.mkdir(plot_path)
	# plot_file='../Synthetic_data/subset_sel_plots_with_test/'+curve+'/'+curve+'_'
	
	unified_K = 0.9
	# obj.split_res_over_K(data_file,res_file,unified_K)
	option='greedy'
	
	# obj.compute_test_result(res_file,data_file,plot_file)
	
	# prefix='../Synthetic_data/error_plots/'+data_curve
	# std =list_of_std[0]
	# lamb=list_of_lamb[0]
	# suffix='_s'+str(std).replace('.','_')+'l'+str(lamb).replace('.','_')
	# plot_file=prefix+suffix
	# image_file='../Synthetic_data/error_plots/'+data_curve.replace('.','_')+suffix

	# obj.get_avg_error_vary_K(res_file,data_file,plot_file,std,lamb)
	# obj.plot_err_vs_K(plot_file,image_file)
	# obj.print_kile_error_VS_K(image_file)

	plot_path='../Synthetic_data/subset_sel_plots/'+curve
	if not os.path.exists(plot_path):
		os.mkdir(plot_path)
	prefix='../Synthetic_data/subset_sel_plots/'+curve+'/'+curve.replace('.','_')+'_'
	obj.plot_subset_selection(prefix,res_file)

	# obj.print_kile_all()
	# obj.print_kile_fix_lambda_K()
	# obj.print_kile_fix_std_lambda()
	# obj.print_kile_fix_lambda()	


	# res_plot_file='../Synthetic_data/res_'+curve+'_for_plot'
	# image_file_prefix='../Synthetic_data/plot_'+curve+'_'
	# obj.extract_result( res_file, res_plot_file)
	# obj.plot_result(res_plot_file,image_file_prefix)


	

if __name__=="__main__":
	main()


	# def update_result(self,res_file):
	# 	res=load_data(res_file)
	# 	for std in self.list_of_std:
	# 		for K in self.list_of_K:
	# 			for lamb in self.list_of_lamb:
	# 				obj=res[str(std)][str(K)][str(lamb)][self.option]
	# 				updated_obj = obj['plt_obj']#self.update_obj(obj) 
	# 				res[str(std)][str(K)][str(lamb)][self.option]=updated_obj
	# 	save(res,res_file)

	# def update_obj(self,obj):
	# 	indicator,centroids=self.get_indicator_centroid(obj)
	# 	final_obj={}
	# 	final_obj['plt_obj']=obj
	# 	final_obj['indicator']=indicator
	# 	final_obj['centroid']=centroids
	# 	return final_obj

	# def get_indicator_centroid(self,plt_obj):

	# 	indicator={}
	# 	human_ind = self.get_indices(plt_obj['human']['x'],plt_obj['prediction']['x'])
	# 	machine_ind = self.get_indices(plt_obj['machine']['x'],plt_obj['prediction']['x'])
	# 	indicator={'human':human_ind,'machine':machine_ind}
	# 	centroid={}
	# 	centroid['human']=np.mean(plt_obj['human']['x'].flatten())
	# 	centroid['machine']=np.mean(plt_obj['machine']['x'].flatten())

	# 	# x=plt_obj['human']['x']
	# 	# y=plt_obj['human']['y']
	# 	# plt.scatter(x,y,c='blue')

	# 	# x=centroid['human']
	# 	# y=0
	# 	# plt.scatter(x,y,c='red')

	# 	# x=plt_obj['machine']['x']
	# 	# y=plt_obj['machine']['y']
	# 	# plt.scatter(x,y,c='green')

	# 	# x=centroid['machine']
	# 	# y=0
	# 	# plt.scatter(x,y,c='yellow')


	# 	# plt.show()
	# 	return indicator,centroid


	# def get_test_error_centroid(self,res_obj,x,y,y_h,K):
	# 	# test_res_obj={}

	# 	# test_res_ranking=0
	# 	# test_res_nc=0
	# 	# test_res_obj['ranking']=0
	# 	# test_res_obj['nearest_centroid']=0
	# 	n=x.shape[0]
	# 	w=res_obj['plt_obj']['prediction']['w']
	# 	human_ind,machine_ind=self.get_nc(res_obj['centroids'],x)
	# 	human_error=(y-y_h)**2
	# 	machine_error=(y-w*x)**2
	# 	test_err = float(human_error[human_ind].sum()+machine_error[machine_ind].sum())/n

	# 	test_res={}
	# 	test_res['avg_test_err']=test_err
	# 	test_res['plt_obj']={'human':x[human_ind],'machine':x[machine_ind]}
	# 	return test_res


	# def get_nc(self,c,x):
	# 	n=x.shape[0]
	# 	l=[1  if x[i]-c['human']<= x[i]-c['machine'] else 0 for i in range(n)]
	# 	human_ind = np.array(l,dtype=bool)
	# 	machine_ind=np.array(np.ones(n)-np.array(l),dtype=bool)
	# 	return human_ind,machine_ind
