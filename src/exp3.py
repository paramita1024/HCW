import time
import os
import sys 
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data 
from triage_human_machine import triage_human_machine


class exp3:
	def __init__(self,data_file,list_of_K, list_of_lamb, list_of_noise_ratio):
		self.data=load_data(data_file)
		self.list_of_K = list_of_K
		self.list_of_lamb = list_of_lamb
		self.list_of_noise_ratio = list_of_noise_ratio

	def eval_loop(self, res_file ):	

		def find_frac( set_a, set_b):
			b_in_a = [ i for i in set_b if i in set_a ] 
			return float( len( b_in_a))/ set_b.shape[0]

		res=load_data(res_file,'ifexists')		
		for K in self.list_of_K :
			if str(K) not in res:
				res[str(K)]={}
			for lamb in self.list_of_lamb :
				num_noise = len( self.list_of_noise_ratio )
				human_low = np.zeros( num_noise ) 
				for noise_ratio, noise_ind  in zip( self.list_of_noise_ratio , range(num_noise ) ) :
					local_data = self.data[ str( noise_ratio ) ]
					triage_obj=triage_human_machine( local_data, True )    
					subset =  triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim='greedy')['subset']
					human_low[ noise_ind ] = find_frac( local_data['low' ] , subset )

				human_high = np.array( [ 1 - human_low[i] for i in range( num_noise ) ] )
				res[ str(K)][ str(lamb)] = human_low
				save( res, res_file )

	def plot_fig( self, res_file, image_file_pre):
		res=load_data(res_file,'ifexists')	
		print '\\begin{figure}[H]'		
		for K in self.list_of_K:
			for lamb in self.list_of_lamb:
				caption='$\\K = '+str(K)+',    \\lambda='+str(lamb)+'$'
				num_noise = len( self.list_of_noise_ratio) 
				suffix = '_'+str(K) + '_' + str(lamb)
				image_file = image_file_pre #+ suffix.replace( '.', '_')
				human_low = res[ str(K)][ str(lamb)]
				human_high = np.array( [ 1 - human_low[i] for i in range( num_noise ) ] )
				self.bar_plot( human_low, human_high , self.list_of_noise_ratio, image_file  )
				X_axis = np.array(self.list_of_noise_ratio).reshape( num_noise, 1 )
				plot_arr = np.hstack(( X_axis, np.vstack(( human_low, human_high )).T ))
				self.write_to_txt( plot_arr, image_file )
				self.print_figure_singleton( image_file, caption )
		print '\\caption{'+res_file.split('/')[-1].split('_')[0]+'}'
		print '\\end{figure}'

	def print_figure_singleton(self,filename,caption):
		print '\\begin{subfigure}{4cm}'
		print '\t \\centering\\includegraphics[width=3cm]{Figure/'+filename+'.pdf}'
		print '\t \\caption{'+caption+'}'
		print '\\end{subfigure}'
				
	def bar_plot( self, human_err, machine_err, x_axis, image_file ):

		# print 'human err', human_err.shape
		# print 'machine', machine_err.shape
		labels = x_axis

		x = np.arange(len(labels))  # the label locations
		width = 0.35  # the width of the bars

		fig, ax = plt.subplots()
		rects1 = ax.bar(x - width/2, human_err, width, label='Fraction of Human in Low Noise')
		rects2 = ax.bar(x + width/2, machine_err, width, label='Fraction of Human in High Noise')

		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_xlabel('Fraction of low noise points')
		ax.set_ylabel('Fraction of sample')
		ax.set_title('Distribution of samples among human, machine')
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend()
		fig.tight_layout()
		plt.savefig(image_file+'.pdf',dpi=600, bbox_inches='tight')
		plt.savefig('../../writing/Figure/'+image_file.split('/')[-1]+'.pdf',dpi=600, bbox_inches='tight')
		plt.show()
		plt.close()

	def write_to_txt(self, res_arr, res_file_txt):
		with open( res_file_txt + '.txt' , 'w') as f:
			for row in res_arr:
				f.write( '\t'.join(map( str, row)) + '\n' )

def main():
	#---------Real Data-------------------------------------------
    setting = ['vary_std_noise','random_noise', 'norm_rand_noise', 'mapped_y_discrete', 'mapped_y_vary_discrete'][int(sys.argv[1])]
    list_of_noise_ratio=[.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_of_lamb=[0.1, 0.5 ]
    list_of_K = [0.6]	
    file_name_list = ['stare5','stare11','mesidor_re','messidor_rg','eyepac','chexpert7','chexpert8']
    path = '../Real_Data_Results/'
    file_name_list = [ file_name_list[int(sys.argv[2])]]
    for file_name in file_name_list:
		print('-'*50+'\n'+file_name+'\n\n'+'-'*50)
		data_file = path + 'data/'+file_name+'_pca50_'+setting
		res_file= path + file_name + '_res_pca50_'+setting
		obj=exp3(data_file, list_of_K, list_of_lamb, list_of_noise_ratio)
		# obj.eval_loop(res_file)	 
		image_file_pre = path + 'Fig3/Fig3_'+file_name 
		obj.plot_fig( res_file, image_file_pre)
	#---------------Synthetic Data------------------------
	# s = ['sigmoid_fig_2_n500d5','Gauss_fig_2_n500d5'][int(sys.argv[1])]
	# if 'sigmoid' in s : # Sigmoid 
	# 	list_of_std = [0.01,0.02,0.03,0.04,0.05]
	# 	list_of_lamb= [0.001]#,0.005,0.01,0.05] #[0.01,0.05] [.1,.5] # [0.0001,0.001,0.01,0.1,.4,.6,.8]#[[int(sys.argv[2])]] # [0.0001,0.001,0.01,0.1] #
	# else:
	# 	list_of_std = [0.01,0.02,0.03,0.04,0.05] # [.01,.05,.1,.5, 1] # [[0.001,0.01,0.1][int(sys.argv[1])]]
	# 	list_of_lamb= [0.005]#,.0005] #0.0001, [.1,.5] # [0.0001,0.001,0.01,0.1,.4,.6,.8]#[[int(sys.argv[2])]] # [0.0001,0.001,0.01,0.1] #
	# list_of_option =['greedy']#,'diff_submod','distort_greedy','stochastic_distort_greedy','kl_triage' ]#[int(sys.argv[2])]
	
	# data_file='../Synthetic_data/data_dict_' +s
	# res_file='../Synthetic_data/res_' +s
	# # ---------------------------------------------------------
	# obj=eval_triage(data_file,real_wt_std=True)
	# for option in list_of_option:
	# 	if option in ['diff_submod', 'stochastic_distort_greedy' ]:
	# 		list_of_K = [ 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ] 
	# 	else:
	# 		list_of_K = [0.99]
	# 	param={'std':list_of_std,'K':list_of_K,'lamb':list_of_lamb}
	# 	obj.eval_loop(param,res_file,option) 
	#---------------Real Data-----------------------------
	# path='../Real_Data/Hatespeech/Davidson/'
	# data_file= path + 'input_tr'
	# res_file= path + 'res'+'_lamb_'+str(list_of_lamb[0])
	# obj=eval_triage(data_file,real_flag=True)
	# obj.eval_loop(param,res_file,option)
	#------------------------------------
	# path_list = []
	# path_list.extend(['../Real_Data/STARE/5/','../Real_Data/STARE/11/'])
	# path_list.append('../Real_Data/Messidor/MESSIDOR/Risk_edema/')
	# path_list.append('../Real_Data/Messidor/MESSIDOR/Retino_grade/')
	# path_list.append('../Real_Data/EyePAC/')
	# # # path = '../Real_Data/Messidor/Messidor_txt/'
	# path_list.extend([ '../Real_Data/CheXpert/data/' + str(i) +'/' for i in [7,8] ] ) # ,13,14,15,16
	# for path,file_name in zip(path_list, file_name_list ):
	# 	data_file_old = path + 'data_split_pca'
	# 	data_file_new = '../Real_Data_Results/data/' + file_name + '_pca100'
	# 	relocate_data( data_file_old, data_file_new)
	#----------------------------------------------------
	# for path in path_list:
	# 	data_file_old = path + 'data_split_pca'
	# 	data_file_new = path + 'data_split_pca10'
	# 	print_new_pca10_data( data_file_old, data_file_new)
	#----------------------------------------------------

		# data_file_old = path + 'data/'+file_name+'_pca100'
		# data_file_new = path + 'data/'+file_name+'_pca50'
		# print_new_pca_data( data_file_old, data_file_new,50)
	#----------------------------------------------------
	# path_list = []
	# path_list.extend(['../Real_Data/STARE/5/','../Real_Data/STARE/11/'])
	# path_list.append('../Real_Data/Messidor/MESSIDOR/Risk_edema/')
	# path_list.append('../Real_Data/Messidor/MESSIDOR/Retino_grade/')
	# path_list.append('../Real_Data/EyePAC/')
	# # # path = '../Real_Data/Messidor/Messidor_txt/'
	# path_list.extend([ '../Real_Data/CheXpert/data/' + str(i) +'/' for i in [7,8] ] ) # ,13,14,15,16
	# for path,file_name in zip(path_list, file_name_list ):
	# 	data_file_old = path + 'data_split_pca10'
	# 	data_file_new = '../Real_Data_Results/data/' + file_name
	# 	relocate_data( data_file_old, data_file_new)
	#--------------------------------------------------
	# data_file = path + 'data/'+file_name
	# print('*'*50+'\n'+file_name+':'+str(load_data(data_file)['X'].shape[0]) +'\n')
	#-----------------------------------------------
	# path='../Real_Data/Movielens/ml-20m/'
	# data_file= path + 'data_tr_splitted'
	# res_file= path + 'res'+'_lamb_'+str(list_of_lamb[0])
	# obj=eval_triage(data_file,real_flag=True)
	# obj.eval_loop(param,res_file,option)
	#---------------------------------------------------
	# path='../Real_Data/BRAND_DATA/'
	# data_file= path + 'data_ht4_vec_split_1'	
	# res_file= path + 'res_ht4_1'# +'_lamb_'+str(list_of_lamb[0])
	# obj=eval_triage(data_file,real_flag=True)
	# obj.eval_loop(param,res_file,option)

if __name__=="__main__":
	main()

	# def eval_kl(self,param,res_file,option):
	# 	res=load_data(res_file,'ifexists')		
	# 	for std in param['std']:
	# 		if self.real:
	# 			data_dict=self.data
	# 		if self.real_wt_std:
	# 			test ={'X':self.data['test']['X'],'Y':self.data['test']['Y'],'c':self.data['test']['c'][str(std)]}
	# 			data_dict = {'X':self.data['X'],'Y':self.data['Y'],'c': self.data['c'][str(std)], 'test':test}

	# 		if str(std) not in res:
	# 			res[str(std)]={}
	# 		for K in param['K']:
	# 			if str(K) not in res[str(std)]:
	# 				res[str(std)][str(K)]={}
	# 			for lamb in param['lamb']:
	# 				if str(lamb) not in res[str(std)][str(K)]:
	# 					res[str(std)][str(K)][str(lamb)]={}

	# 				print 'std-->', std, 'K--> ',K,' Lamb--> ',lamb
	# 				data_dict['lamb'] = lamb

	# 				kl_obj=kl_triage(data_dict)
	# 				res[str(std)][str(K)][str(lamb)][option] = {'train_res':0,'test_res':{'nearest':{'error':kl_obj.testing(K)} } }
	# 	save(res,res_file)