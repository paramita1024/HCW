import time
import os
import sys 
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data 
from triage_human_machine import triage_human_machine


def print_new_pca_data( data_file_old, data_file_new,n):
	data=load_data( data_file_old )
	data['X']=data['X'][:,:n]
	data['test']['X']=data['test']['X'][:,:n]
	save( data, data_file_new)

def relocate_data( data_file_old, data_file_new ):
	data=load_data( data_file_old)
	save( data, data_file_new )

class eval_triage:
	def __init__(self,data_file,real_flag=None, real_wt_std=None):
		self.data=load_data(data_file)
		self.real=real_flag		
		self.real_wt_std=real_wt_std


	def eval_loop(self,param,res_file,option):	
		res=load_data(res_file,'ifexists')		
		for std in param['std']:
			if self.real:
				data_dict=self.data
				triage_obj=triage_human_machine(data_dict,self.real)
			else:
				if self.real_wt_std:
					data_dict = {'X':self.data['X'],'Y':self.data['Y'],'c': self.data['c'][str(std)]}
					triage_obj=triage_human_machine(data_dict,self.real_wt_std)
				else:
					test={'X':self.data.Xtest,'Y':self.data.Ytest,'human_pred':self.data.human_pred_test[str(std)]}
					data_dict = {'test':test,'dist_mat':self.data.dist_mat,  'X':self.data.Xtrain,'Y':self.data.Ytrain,'human_pred':self.data.human_pred_train[str(std)]}
					triage_obj=triage_human_machine(data_dict,False)
			if str(std) not in res:
				res[str(std)]={}
			for K in param['K']:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in param['lamb']:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					# res[str(std)][str(K)][str(lamb)]['greedy'] = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim='greedy')
					print 'std-->', std, 'K--> ',K,' Lamb--> ',lamb
					res_dict = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim=option)
					res[str(std)][str(K)][str(lamb)][option] = res_dict
					save(res,res_file)


def main():
	#---------Real Data-------------------------------------------
	setting = ['vary_std_noise','random_noise', 'norm_rand_noise', 'mapped_y_discrete', 'mapped_y_vary_discrete', 'mapped_y_vary_discrete_old', 'mapped_y_vary_discrete_3'][5]#int(sys.argv[1])]
	# list_of_std=[ float(sys.argv[4]) ]
	# list_of_lamb=[ float(sys.argv[5]) ]
	list_of_std =[0.2, 0.4, 0.6, 0.8]#range(6) #  [.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	# list_of_std.reverse()
	list_of_lamb=[float(sys.argv[3])] #, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1,0.5]#, 
	list_of_option =['greedy','diff_submod','distort_greedy','kl_triage' ]
	file_name_list = ['stare5','stare11','mesidor_re','messidor_rg','eyepac','chexpert7','chexpert8', 'messidor_full_re','messidor_full_rg']
	path = '../Real_Data_Results/'
	file_name_list = [ file_name_list[int(sys.argv[1])]]
	list_of_option = [ list_of_option[int( sys.argv[2]) ] ]
	for file_name in file_name_list:
		print('-'*50+'\n'+file_name+'\n\n'+'-'*50)
		data_file = path + 'data/'+file_name+'_pca50_'+setting
		
		res_file= path + file_name + '_res_pca50_'+setting
		for option in list_of_option:
			if option not in ['diff_submod']:
				print( '*'*50 + '\n' + option + '\n' + '*'*50 )
				if option in ['diff_submod']: 
					list_of_K = [ 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ] 
				else:
					list_of_K = [0.99]	
				param={'std':list_of_std,'K':list_of_K,'lamb':list_of_lamb}
				obj=eval_triage(data_file,real_wt_std=True)
				obj.eval_loop(param,res_file,option)	
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