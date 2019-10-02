import time
import sys
import os
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data

class plot_triage_real:
	def __init__(self,list_of_K,list_of_std, list_of_lamb,list_of_option,list_of_test_option, flag_synthetic=None):
		self.list_of_K=list_of_K
		self.list_of_std=list_of_std
		self.list_of_lamb=list_of_lamb
		self.list_of_option=list_of_option
		self.list_of_test_option = list_of_test_option
		self.flag_synthetic = flag_synthetic

	def plot_fig3( self, res_file, data_file , std, K, lamb):
		def split_c( c , c_threshold):
			indices_low = np.array([ i for i in range(c.shape[0]) if c[i] < c_threshold] )
			indices_high = np.array([ i for i in range(c.shape[0]) if i not in indices_low ])
			# print indices_low.shape[0]+indices_high.shape[0]
			return indices_low, indices_high

		def f( err_l, err_h, human_ind ): 
			err_low_human = len([ i for i in human_ind if i in err_l ])
			err_high_human = len([ i for i in human_ind if i in err_h ])

			num_low = err_l.shape[0]
			num_high = err_h.shape[0]
			ratio_human = [ float( err_low_human)/num_low, float( err_high_human)/num_high ]
			ratio_machine = [ 1-ratio_human[0], 1- ratio_human[1]]

			return ratio_human, ratio_machine

		data = load_data(data_file)

		# plt.plot( data['c'][std])
		# plt.show()
		# print np.mean( data['c'][std] )

		c_threshold= float(sys.argv[6])#input('enter threshold')

		# return
		res_obj = load_data(res_file)[std][K][lamb]['greedy']		
		tr_h = res_obj['subset'] 
		# plot for tr
		err_low , err_high = split_c(data['c'][std], c_threshold)
		ratio_h, ratio_m = f( err_low, err_high , tr_h )
		print ratio_h
		print ratio_m
		# ratio = np.vstack((np.array(ratio_h), np.array(ratio_m) )).T
		# plot_arr = np.hstack((np.array([[0],[1]]), ratio))
		# self.write_to_txt(plot_arr, 'Fig3_mesidor_re.txt')
		# return 
		self.bar_plot( ratio_h, ratio_m, '' )

	def bar_plot( self, human_err, machine_err, file_w):

		labels = ['Low Error', 'High Error']
	
		x = np.arange(len(labels))  # the label locations
		width = 0.35  # the width of the bars

		fig, ax = plt.subplots()
		rects1 = ax.bar(x - width/2, human_err, width, label='Human')
		rects2 = ax.bar(x + width/2, machine_err, width, label='Machine')

		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_ylabel('Fraction of sample')
		ax.set_title('Distribution of samples among human, machine')
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend()


		# def autolabel(rects):
		# 	"""Attach a text label above each bar in *rects*, displaying its height."""
		# 	for rect in rects:
		# 		height = rect.get_height()
		# 		ax.annotate('{}'.format(height),
		# 					xy=(rect.get_x() + rect.get_width() / 2, height),
		# 					xytext=(0, 3),  # 3 points vertical offset
		# 					textcoords="offset points",
		# 					ha='center', va='bottom')


		# autolabel(rects1)
		# autolabel(rects2)

		fig.tight_layout()
		# plt.savefig('Fig3_mesidor_re')
		plt.savefig('Fig3_stare5.pdf',dpi=600, bbox_inches='tight')
		plt.show()

	def get_avg_error_vary_K(self,res_file,image_path):
		res=load_data(res_file)
		for std in self.list_of_std:
			print '\\begin{figure}[H]'	
		
			for lamb in self.list_of_lamb:
				for test_method in self.list_of_test_option:
					suffix='std_'+str(std)+'_lamb_'+str(lamb)+'_'+test_method
					image_file=image_path +suffix.replace('.','_')
					caption='$\\rho = '+str(std)+',    \\lambda='+str(lamb)+'$'
					self.print_figure_singleton(image_file.split('/')[-1],caption)
					
					plot_obj={}
					plot_arr = np.zeros( ( len( self.list_of_option) , len( self.list_of_K)))
					ind=0
					for option in self.list_of_option:
						option_flag=0
						err_K_tr=[]
						err_K_te=[]
						for K in self.list_of_K:
							if option in res[str(std)][str(K)][str(lamb)]:
								# print 'option', option
								option_flag=1
								err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
								# print 'option.key', res[str(std)][str(K)][str(lamb)][option].keys()
								# print '*'*50
								# print 'subset size',res[str(std)][str(K)][str(lamb)][option]['subset'].shape
								# print 'test err',res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]['error']
								err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]['error'])
						if option_flag==1 :
							plot_obj[option]={'train':err_K_tr,'test':err_K_te}
							plot_arr[ind]=np.array( err_K_tr )
							ind+=1
						
					self.plot_err_vs_K(image_file,plot_obj)
					K_col = np.array([ int(k*500*0.8) for k in self.list_of_K ])		
					plot_arr = np.hstack(( K_col.reshape(K_col.shape[0],1), plot_arr.T))
					self.write_to_txt(plot_arr, image_file+'.txt')
			print '\\caption{'+res_file.split('/')[-1].split('_')[0]+'}'
			print '\\end{figure}'

	def get_avg_error_exp3(self,res_file,image_path):
		def smooth( vec ):
			vec = np.array(vec)
			tmp = np.array( [ (.25*vec[ind]+.5*vec[ind-1]+.25*vec[ind+1]) for ind  in range(1,vec.shape[0]-1)  ])
			vec[0] = vec[0]*.75+vec[1]*.25
			vec[-1] = vec[-1]*.75+vec[-2]*.25
			vec[1:-1] = tmp
			return vec
		option = 'greedy'
		test_method='nearest'
		res=load_data(res_file)
		print '\\begin{figure}[H]'	
		
		for lamb in self.list_of_lamb:
			suffix='_lamb_'+str(lamb) + '_' + sys.argv[4]
			image_file=image_path #  +suffix.replace('.','_')		
			caption='$\\lambda='+str(lamb)+'$'
			self.print_figure_singleton(image_file.split('/')[-1],caption)
			plot_obj={}
			plot_arr = np.zeros( ( len( self.list_of_std) , len( self.list_of_K)))		
			ind=0
			for std in self.list_of_std:
				err_K_tr=[]
				err_K_te=[]
				for K in self.list_of_K:
					err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
					# print 'option.key', res[str(std)][str(K)][str(lamb)][option].keys()
					# print '*'*50
					# print 'subset size',res[str(std)][str(K)][str(lamb)][option]['subset'].shape
					# print 'test err',res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]['error']
					err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]['error'])
				plot_obj[str(std)]={'train':smooth(err_K_tr),'test':smooth(err_K_te )}
				plot_arr[ind]=np.array( err_K_te )
				ind+=1
				
			self.plot_err_vs_K(image_file,plot_obj)
			K_col = np.array([ int(k*self.n) for k in self.list_of_K ])		
			plot_arr = np.hstack(( K_col.reshape(K_col.shape[0],1), plot_arr.T))
			self.write_to_txt(plot_arr, image_file+'.txt')
		print '\\caption{'+res_file.split('/')[-1].split('_')[0]+'}'
		print '\\end{figure}'

	def print_figure_singleton(self,filename,caption):
		print '\\begin{subfigure}{4cm}'
		print '\t \\centering\\includegraphics[width=3cm]{Figure/'+filename+'.pdf}'
		print '\t \\caption{'+caption+'}'
		print '\\end{subfigure}'

	def plot_err_vary_std_K(self, res_file, res_file_txt, n):
		res=load_data(res_file)
		plot_arr= np.zeros( (len(self.list_of_std), len(self.list_of_K)))
		K_axis = np.array([ int(k*n*0.8) for k in self.list_of_K ])
		for std,std_ind in zip(self.list_of_std, range( len(self.list_of_std) )):
			for lamb in self.list_of_lamb:
				for test_method in self.list_of_test_option:
					# suffix='lamb_'+str(lamb)+'_'+test_method
					# image_file=image_path+suffix.replace('.','_')
					plot_obj={}
					for option in self.list_of_option:
						# err_K_tr=[]
						err_K_te=[]
						for K in self.list_of_K:
							# err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
							err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]['error'])
			plot_arr[std_ind]=np.array( err_K_te )
			plt.plot( err_K_te, label = str(std),linewidth=8,linestyle='--',marker='o', markersize=10)
		plt.grid()
		plt.xticks(range(len(self.list_of_K)),K_axis)
		plt.legend()
		plt.xlabel('K')
		plt.ylabel('Average Squared Error')
		plt.title('Average Squared Error  Vs Deviation of human error')
		# plt.savefig(res_file_txt+'.jpg')
		plt.savefig(res_file_txt+'.pdf')
		plt.show()
		# K = np.array([ int(k*n*0.8) for k in self.list_of_K ])
		
		plot_arr = np.hstack(( K_axis.reshape(K_axis.shape[0],1), plot_arr.T))
		self.write_to_txt(plot_arr, res_file_txt)

	def write_to_txt(self, res_arr, res_file_txt):
		with open( res_file_txt, 'w') as f:
			for row in res_arr:
				f.write( '\t'.join(map( str, row)) + '\n' )

	def plot_err_vs_K(self,image_file,plot_obj):
		key = sys.argv[4]
		# key = 'train'
		for option in plot_obj.keys():
			plt.plot( plot_obj[option][key], label=key+' '+option, linewidth=8,linestyle='--',marker='o', markersize=10)
		# plt.plot(plot_obj['greedy']['test'],label='GR',linewidth=8,linestyle='--',marker='o', markersize=10,color='red')
		# plt.plot(plot_obj['diff_submod']['test'],label='DS',linewidth=8,linestyle='-',marker='o', markersize=10,color='blue')
		# plt.plot(plot_obj['distort_greedy']['test'],label='DG',linewidth=8,linestyle='--',marker='o', markersize=10,color='green')
		# plt.plot(plot_obj['stochastic_distort_greedy']['test'],label='SDG',linewidth=8,linestyle='-',marker='o', markersize=10,color='yellow')
		# plt.plot(plot_obj['kl_triage']['test'],label='KL',linewidth=8,linestyle='-',marker='o', markersize=10,color='black')
		plt.grid()
		plt.legend()
		plt.xlabel('K')
		plt.ylabel('Average Squared Error')
		plt.title('Average Squared Error')
		plt.xticks(range(len(self.list_of_K)),self.list_of_K)
		plt.savefig(image_file+'.pdf',dpi=600, bbox_inches='tight')
		plt.savefig(image_file+'.jpg',dpi=600, bbox_inches='tight')
		plt.savefig('../../writing/Figure/'+image_file.split('/')[-1]+'.pdf',dpi=600, bbox_inches='tight')
		plt.show()
		save(plot_obj,image_file)
		plt.close()

	def get_nearest_human(self,dist,tr_human_ind):
		
		# start= time.time()
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
		# print 'Time required -----> ', time.time() - start , ' seconds'
		return (human_dist -machine_dist)

	def get_test_error(self,res_obj,dist_mat,x,y,y_h=None,c=None,K=None):
		
		w=res_obj['w']
		subset=res_obj['subset']
		n,tr_n=dist_mat.shape
		no_human=int((subset.shape[0]*n)/float(tr_n))

		y_m=x.dot(w)
		err_m=(y-y_m)**2
		if y_h==None:
			err_h=c  
		else: 
			err_h=(y-y_h)**2

		# start = time.time()
		diff_arr=[ self.get_nearest_human(dist,subset) for dist in dist_mat]
		# print 'Time required -----> ', time.time() - start , ' seconds'

		indices=np.argsort(np.array(diff_arr))
		subset_te_r = indices[:no_human]
		subset_machine_r=indices[no_human:]

		if subset_te_r.size==0:
			error_r =  err_m.sum()/float(n)
		else:
			error_r = ( err_h[subset_te_r].sum() + err_m.sum() - err_m[subset_te_r].sum() ) /float(n)


		subset_te_n = np.array([int(i)  for i in range(len(diff_arr)) if diff_arr[i] < 0 ])
		# print 'subset size test', subset_te_n.shape
		subset_machine_n = np.array([int(i)  for i in range(len(diff_arr)) if i not in subset_te_n ])
		# print 'sample to human--> ' , str(subset_te_n.shape[0]), ', sample to machine--> ', str( subset_machine_n.shape[0])

		if subset_te_n.size==0:
			error_n =  err_m.sum()/float(n)
		else:
			error_n = ( err_h[subset_te_n].sum() + err_m.sum() - err_m[subset_te_n].sum() ) /float(n)

		# return {'error':error, 'human_ind':subset_te, 'machine_ind':subset_machine}
		error_n={'error':error_n, 'human_ind':subset_te_n, 'machine_ind':subset_machine_n}
		error_r={'error':error_r, 'human_ind':subset_te_r, 'machine_ind':subset_machine_r}
		return error_n, error_r

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
					
	def get_train_error(self,plt_obj,x,y,y_h=None,c=None):
		subset = plt_obj['subset']
		# print np.min(subset)
		# print np.max(subset)

		w=plt_obj['w']
		n=y.shape[0]
		if y_h==None:
			err_h=c
		else:
			err_h=(y_h-y)**2

		# print x.shape

		y_m= x.dot(w)
		err_m=(y_m-y)**2
		# print '-----------'
		# print err_h.shape
		# print err_m.shape
		# print '-----------'
		error = ( err_h[subset].sum()+err_m.sum() - err_m[subset].sum() ) /float(n)
		return {'error':error}

	def compute_result(self,res_file,data_file,option, image_file_prefix =None):
		data=load_data(data_file)
		res=load_data(res_file)
		for std,i0 in zip(self.list_of_std,range( len(self.list_of_std) )):
			for K,i1 in zip(self.list_of_K,range(len(self.list_of_K))):
				for lamb,i2 in zip(self.list_of_lamb,range(len(self.list_of_lamb))):
					if option in res[str(std)][str(K)][str(lamb)]:
						res_obj=res[str(std)][str(K)][str(lamb)][option]
						# print 'std,K,lamb,subset_size ---- > ',std,K,lamb,  res_obj['subset'].shape
						suffix='_'+option + '_std_'+str(std)+'_K_'+str(K)+'_lamb_'+str(lamb)
						image_file =  image_file_prefix + suffix #'../Synthetic_data/demo/'
						# self.plot_subset_allocation( data['X'], data['Y'], res_obj['w'], res_obj['subset'], image_file )
						# print 'std', str(std), '  K', str(K), '  lamb  ', str(lamb)
						train_res = self.get_train_error(res_obj,data['X'],data['Y'],y_h=None,c=data['c'][str(std)])
						test_res_n,test_res_r = self.get_test_error(res_obj,data['dist_mat'],data['test']['X'],data['test']['Y'],y_h=None,c=data['test']['c'][str(std)],K=K)
						res[str(std)][str(K)][str(lamb)][option]['test_res']={'ranking':test_res_r,'nearest':test_res_n}
						res[str(std)][str(K)][str(lamb)][option]['train_res']=train_res
					# else:
						
					# 	print option, ' is not evaluated for (std,K,lamb) = ', std , K , lamb
		save(res,res_file)

	def plot_subset_allocation( self, X, Y, w, subset, image_file):

		x=X[:,0].flatten()[subset]
		y=Y[subset]
		plt.scatter(x,y,c='blue',label='human')

		subset_c = np.array([ i for i in range( Y.shape[0]) if i not in  subset])
		x=X[:,0].flatten()[subset_c]
		y=Y[subset_c]
		plt.scatter(x,y,c='green',label='machine')

		
		x=X[:,0].flatten()
		y=X.dot(w)
		plt.scatter(x,y,c='yellow',label='prediction')

		plt.legend()
		plt.grid()
		plt.xlabel('<-----------x------------->')
		plt.ylabel('<-----------y------------->')
		plt.savefig(image_file+'.pdf',dpi=600, bbox_inches='tight')
		# plt.savefig(image_file+'.jpg',dpi=600, bbox_inches='tight')
		# plt.show()
		plt.close()
		# plt.show()
			
	def merge_results(self,input_res_files,merged_res_file):

		res={}
		for std in self.list_of_std:
			if str(std) not in res:
				res[str(std)]={}
			for K in self.list_of_K:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in self.list_of_lamb:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					r=load_data(input_res_files[str(lamb)])
					# print r['0.0'].keys()
					# print res['0.0'].keys()
					res[str(std)][str(K)][str(lamb)] = r[str(std)][str(K)][str(lamb)]
		save(res,merged_res_file)

	def split_res_over_K(self,data_file,res_file,unified_K,option):
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
					
					if option not in res[str(std)][str(K)][str(lamb)]:
						res[str(std)][str(K)][str(lamb)][option]={}
					if K != unified_K:
						res_dict = res[str(std)][str(unified_K)][str(lamb)][option]
						if res_dict:
							res[str(std)][str(K)][str(lamb)][option] = self.get_res_for_subset(data_file,res_dict,lamb,K)
		save(res,res_file)

	def get_optimal_pred(self,data,subset,lamb):
		
		n,dim= data['X'].shape
		subset_c=  np.array([int(i) for i in range(n) if i not in subset])	
		X_sub=data['X'][subset_c].T
		Y_sub=data['Y'][subset_c]
		subset_c_l=n-subset.shape[0]
		return LA.inv( lamb*subset_c_l*np.eye(dim) + X_sub.dot(X_sub.T) ).dot(X_sub.dot(Y_sub))

	def get_res_for_subset(self,data_file,res_dict,lamb,K):
		data=load_data(data_file)
		curr_n = int( data['X'].shape[0] * K )
		subset_tr = res_dict['subset'][:curr_n]
		w= self.get_optimal_pred(data,subset_tr,lamb)
		return {'w':w,'subset':subset_tr}

	def set_n( self, n ):
		self.n = n
def main():
	#---------Real Data-------------------------------------------
	setting = ['vary_std_noise','random_noise', 'norm_rand_noise', 'mapped_y_discrete', 'mapped_y_vary_discrete', 'mapped_y_vary_discrete_old', 'mapped_y_vary_discrete_3'][5]
	list_of_std =[0.2, 0.4, 0.6, 0.8]
	# list_of_std.reverse()
	list_of_lamb =[ float(sys.argv[3]) ]# [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1,0.5]#, list_of_lamb=[0.001]#[0.001, 0.005, 0.01, 0.05, 0.1, 0.5 ]#float(sys.argv[5])] # [ 0.001, 0.005, 0.01, 0.05, 0.1, 0.5 ]
	list_of_K = [  0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ] 
	list_of_option =['greedy','diff_submod','distort_greedy','kl_triage' ]#'stochastic_distort_greedy'
	list_of_test_option = ['nearest']
	file_name_list = ['stare5','stare11','mesidor_re','messidor_rg','eyepac','chexpert7','chexpert8', 'messidor_full_re','messidor_full_rg']
	file_name_list = [ file_name_list[int(sys.argv[1])]]
	list_of_option = [ list_of_option[int( sys.argv[2]) ] ]
	path = '../Real_Data_Results/'
	obj=plot_triage_real(list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option)
	for file_name in file_name_list:
		data_file = path + 'data/' + file_name + '_pca50_' + setting
		res_file= path + file_name + '_res_pca50_' + setting
		# print res_file
		# print('-'*50+'\n'+file_name+'\n\n'+'-'*50)
		obj.set_n( load_data( data_file )['X'].shape[0] )
		for option in list_of_option:
			# if True:
			# 	obj.plot_fig3( res_file, data_file, str( list_of_std[0] ), str( list_of_K[0] ), str( list_of_lamb[0]) )
			# 	return
			# print('*'*10+'\n'+option+'\n'+'*'*10)
			if option not in [ 'diff_submod']: # ,'distort_greedy'
				unified_K = 0.99
				# print load_data(res_file)['0.05']['0.99']['0.001'].keys()
				obj.split_res_over_K(data_file,res_file,unified_K,option)
			obj.compute_result(res_file,data_file,option, 'dummy')
		image_path = path + 'Fig3/Fig3_' + file_name #+ setting + '/'+file_name
		obj.get_avg_error_exp3(res_file,image_path)	

if __name__=="__main__":
	main()
