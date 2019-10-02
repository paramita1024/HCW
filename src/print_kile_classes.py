import sys
import os
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data

class plot_triage:
	def __init__(self,list_of_K,list_of_std, list_of_lamb,curve=None,option=None,sigma=None):
		self.list_of_K=list_of_K
		self.list_of_std=list_of_std
		self.list_of_lamb=list_of_lamb
		self.curve=curve
		self.option=option
		self.sigma=sigma
		
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
		# print '\\subsection{'+'}'
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

	def print_kile_fix_lambda_real(self,file_name):
		num_K=len(self.list_of_K)
		num_lamb=len(self.list_of_lamb)
		print '\\subsection{'+file_name.replace('_',' ')+'}'
		for test_method in ['ranking', 'nearest']:
			print '\\begin{figure}[H]'
			for std in self.list_of_std:
				for lamb in self.list_of_lamb:
					filename= file_name +'_std_'+str(std).replace('.','_') +'_lamb_'+str(lamb).replace('.','_')+'_'+test_method
					# caption='K='+str(self.list_of_K[j])
					caption = '$ \\lambda='+str(lamb)+'$' # \\rho='+str(std)+'
					self.print_figure_singleton(filename,caption)
			print '\\caption{Greedy, Variation with K, testing by '+test_method+' }'
			print '\\end{figure}'



def main():
	# file_name = 'Hatespeech_Davidson'
	# file_name= 'Messidor_txt'
	# file_name='EyePAC'
	# file_name = 'Retino_grade'
	# file_name='Risk_edema'
	# file_name='STARE_5'
	file_name='STARE_11'
	list_of_std=np.array([0.0])
	# list_of_std=np.array([0.001,0.01,0.1])
	list_of_lamb= [0.0001,0.001,0.01,0.1,0.4,0.6,0.8] # ,0.1] #	 [0.001,0.01] #
	list_of_K=[0.01,0.1,0.2,0.3,.4,0.5,.6,.7,.8,.9,.99]
	obj=plot_triage(list_of_K,list_of_std, list_of_lamb)
	obj.print_kile_fix_lambda_real(file_name)
	#--------------------------------------------------------------------------------
	# synthetic data
	# Sigmoid
	# list_of_std=np.array([.01,.05,.1,.5])
	# list_of_lamb=[0.01,0.1,0.5,1.0]
	# list_of_lamb=[0.0001,0.001,0.01,0.1]
	
	# Gauss
	# list_of_std=np.array([0.01])#01,0.005])#([0.001,.005,0.01,.05,.1])
	# list_of_lamb=[0.01]#[0.0001 ,0.001,0.01,0.1]
	# list_of_K=[.2,.4,.6,.8]
	# list_of_K=[0.01,0.1,0.2,0.3,.4,0.5,.6,.7,.8,.9,.99]
	# sigma=0.1
	# data_curve='sigmoid'
	# data_curve='Gauss_'+str(sigma) 
	# curve='sigmoid_d_s'
	# curve='Gauss_'+str(sigma) # +'_d_s'#_gr' #
	# option='greedy'#'diff_submod_greedy'# 'greedy'#'diff_submod_greedy'#'greedy'#
	# data_file='../Synthetic_data/data_'+data_curve
	# res_file='../Synthetic_data/res_'+curve
	
	# synthetic data
	# obj=plot_triage(list_of_K,list_of_std, list_of_lamb,curve,option,sigma)
	# plot_path='../Synthetic_data/subset_sel_plots_with_test/'+curve
	# if not os.path.exists(plot_path):
	# 	os.mkdir(plot_path)
	# plot_file='../Synthetic_data/subset_sel_plots_with_test/'+curve+'/'+curve+'_'
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

	# plot_path='../Synthetic_data/subset_sel_plots/'+curve
	# if not os.path.exists(plot_path):
	# 	os.mkdir(plot_path)
	# prefix='../Synthetic_data/subset_sel_plots/'+curve+'/'+curve.replace('.','_')+'_'
	# obj.plot_subset_selection(prefix,res_file)

	# obj.print_kile_all()
	# obj.print_kile_fix_lambda_K()
	# obj.print_kile_fix_std_lambda()
	# obj.print_kile_fix_lambda()	

if __name__=="__main__":
	main()

