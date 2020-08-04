#Herein we shall create a new file, similar to esn.py 
#where we transform the notebook into an object oriented approach worthy of Reinier's library.

def pp(variable, label):
	"""
	custom print function
	"""
	print(label +": " + str(variable))

class EchoStateExperiment:
	"""
	#TODO description
	bounds:
	size: a string in ["small", "medium", "publish"] that refer to different dataset sizes.
	file_path: a string that describes the directory where the data is located. (load from)
	out_path: where to save the data
	target_frequency: in Hz which frequency we want to target as the center of a block experiment or the only frequency in the case of a simple prediction.

	"""
	def __init__(self, 
				 size, 
				 file_path = "spectogram_data/", 
				 target_frequency = 2000,
				 out_path = None):
		# Parameters
		assert size in ["small", "medium", "publish"], "Please choose a size from ['small', 'medium', 'publish']"
		#self.dataset = datase
		self.size = size
		self.file_path = file_path + self.size + "/"
		self.target_frequency = target_frequency
		self.load_data()
		self.horiz_display()
		self.bounds = {"observer_bounds" : None, "response_bounds" : None} 
		self.out_path = out_path

	

	def hz2idx(self, 
		   	   obs_hz = None, 
		   	   target_hz = None, 
		   	   silent = True):
		"""
		This function acts as a helper function to simple_block and get_observers
		and is thus designed. It takes a desired hz amount and translates that to indices of the data.
		
		To do one frequency use Freq2idx.
		"""
		### START helper functions
		def my_sort(lst):
			return(list(np.sort(lst)))

		def endpoints2list(lb, ub): # [lb, ub] stands for [lowerbound, upperbound]
			return list(range(int(lb), int(ub + 1)))
		### END helper functions

		midpoint = self.target_frequency 
		height   = self.freq_axis_len 

		#items needed for the file name:
		self.obs_kHz, self.target_kHz = obs_hz / 1000.0, target_hz / 1000

		# spread vs total hz
		obs_spread, target_spread = obs_hz / 2, target_hz / 2
		
		# get the obs, response range endpoints
		respLb, respUb = [Freq2idx(midpoint - target_spread), 

						  Freq2idx(midpoint + target_spread)]
		obs_high_Ub, obs_high_lb =  respUb + Freq2idx(obs_spread) + 1, respUb + 1
		obs_low_lb, obs_low_Ub = respLb - Freq2idx(obs_spread) - 1, respLb - 1
	  
		# Listify:
		resp_idx_Lst = endpoints2list(respLb, respUb)
		obs_idx_Lst1, obs_idx_Lst2 =  endpoints2list(obs_low_lb, obs_low_Ub), endpoints2list(obs_high_lb, obs_high_Ub)

		# collect frequencies:
		resp_Freq_Lst = [experiment.f[i] for i in resp_idx_Lst]
		obs_Freq_Lst1, obs_Freq_Lst2 = [experiment.f[i] for i in obs_idx_Lst1], [experiment.f[i] for i in obs_idx_Lst2]
		
		#INVERSION
		resp_idx_Lst = [height - i for i in resp_idx_Lst]
		obs_idx_Lst1, obs_idx_Lst2 = [height - i for i in obs_idx_Lst1 ], [height - i for i in obs_idx_Lst2 ]
		
		#SORT
		obs_idx_Lst1, obs_idx_Lst2, resp_idx_Lst = my_sort(obs_idx_Lst1), my_sort(obs_idx_Lst2), my_sort(resp_idx_Lst)
		
		
		if silent != True:
			print("resp_indexes : " + str(resp_idx_Lst))
			print("observer frequencies upper domain: " + str(resp_Freq_Lst) + 
				  " , range: "+ str(abs(resp_Freq_Lst[0] - resp_Freq_Lst[-1])) +" Hz\n")

			print("observer indexes lower domain: " + str(obs_idx_Lst1))
			print("observer frequencies lower domain: " + str(obs_Freq_Lst1) + 
				  " , range: "+ str(abs(obs_Freq_Lst1[0] - obs_Freq_Lst1[-1])) +" Hz\n")

			print("observer indexes upper domain: " + str(obs_idx_Lst2))
			print("observer frequencies upper domain: " + str(obs_Freq_Lst2) + 
				  " , range: "+ str(abs(obs_Freq_Lst2[0] - obs_Freq_Lst2[-1])) +" Hz\n")

		assert obs_idx_Lst2 + resp_idx_Lst + obs_idx_Lst1 == list(range(obs_idx_Lst2[ 0 ], obs_idx_Lst1[ -1] + 1))

		dict2Return = {"obs_idx": obs_idx_Lst2 + obs_idx_Lst1, 
					   "resp_idx": resp_idx_Lst,
					   "obs_freq" : obs_Freq_Lst1 + obs_Freq_Lst2,
					   "resp_freq" : resp_Freq_Lst}

		self.resp_obs_idx_dict = dict2Return



	def load_data(self, verbose = True):
		assert type(verbose) == bool, "verbose must be a boolean"

		spect_files  = { "publish" : "_new", "small" : "_512" , "original" : "", "medium" : "_1024"}

		files2import = [self.file_path  + i + spect_files[self.size] for i in ("T", "f", "Intensity") ]
		
		data_lst = []
		for i in files2import:
			data_lst.append(loadmat(i))

		self.T, self.f, self.A = data_lst #TODO rename T, f and A (A should be 'spectogram' or 'dataset')

		#preprocessing
		self.T, self.A = self.T['T'], self.A['M']
		self.T, self.A = np.transpose(self.T), (self.A - np.mean(self.A))/np.std(self.A)


		self.A_orig = self.A.copy()
		self.A_orig = np.rot90(self.A_orig, k = 1, axes = (0, 1))
		

		self.f = self.f['f'].reshape(-1,).tolist()
		self.max_freq = int(np.max(self.f))
		self.Freq2idx(self.target_frequency, init = True)

		self.freq_axis_len = self.A.shape[0]
		self.time_axis_len = self.A.shape[1]
		str2print = ""
		if verbose == True:
			for file_name_ in files2import:
				str2print += "successfully loaded: " + file_name_ + ".mat, "
			print("maximum frequency: " + str(self.max_freq))
			print("dataset shape: " + str(self.A.shape))
		self.freq_idx = [int(i) for i in self.f]
			#plt.imshow(A_orig)

		self.key_freq_idxs = {}
		for i in (2000, 4000, 8000):
			height = self.A.shape[0]
			self.Freq2idx(i)
			self.key_freq_idxs[i] = height - self.targetIdx

	def olab_display(self, axis, return_index = False):
		"""
		#TODO reconsider renaming vert_display
		Plot a version of the data where time is along the x axis, designed to show RPI lab
		"""
		AOrig = self.A_orig
		oA = np.rot90(AOrig, k = 3, axes = (0, 1))
		#oA stands for other lab A
		oA = pd.DataFrame(oA).copy()
		
		oA.index = self.freq_idx
		yticks = list( range( 0, self.max_freq, 1000))
		y_ticks = [ int(i) for i in yticks]
		my_heat = sns.heatmap(oA, center=0, cmap=sns.color_palette("CMRmap"), yticklabels = self.A.shape[0]//10, ax = axis)
		#, cmap = sns.color_palette("RdBu_r", 7))
		axis.set_ylabel('Frequency (Hz)')#,rotation=0)
		axis.set_xlabel('time')
		my_heat.invert_yaxis()
		plt.yticks(rotation=0)
		if return_index == True:
			return(freq_idx)

	def Freq2idx(self, val, init = False):
		"""
		Translates a desired target frequency into a desired index
		"""
		freq_spec = min(range(len(self.f)), key=lambda i: abs(self.f[i] - val))
		assert type(init) == bool, "init must be a bool"
		if init == False:
			return(freq_spec)
		else:
			self.targetIdx = freq_spec

	def simple_block(self, 
					 target_freq = 2000, 
					 split = 0.5, 
					 target_timeseries = None, 
					 n_obs = None, 
					 silent = False, 
					 aspect = 1):
		"""
		This is a helper function for get_observers which is a way to simplify the block method
		for the purposes of our research. It only accepts 3 parameters and doesn't allow for multiple blocks.
		"""

		ctr = self.key_freq_idxs[target_freq]
		if target_timeseries != None:  
			#TODO EVEN AND ODD TIMESERIES
			target_spread = target_timeseries // 2
			#resp_bounds is the response bounds ie the target area bounds.
			resp_bounds = [ctr - target_spread, ctr + target_spread] 
		else: #TODO does this part of the if-else statement actually do anything?
			response_bounds = None
			resp_bounds = [ctr, ctr]
		assert n_obs != None, "if you want to have no observers then #TODO"
		
		obs_bounds  = [[resp_bounds[0] - n_obs, resp_bounds[0]],
					   [resp_bounds[1], resp_bounds[1] + n_obs ]]

		if silent != True:
			print("response bounds: " + str(resp_bounds))
			print("observers bounds: " + str(obs_bounds))
		self.bounds = {"response_bounds" : resp_bounds, "observer_bounds" : obs_bounds}
		self.get_observers(method = "block",
					missing = ctr,
					split = split,
					observer_range = self.bounds["observer_bounds"],  #format: [[425, 525], [527,627]],
					response_range = self.bounds["response_bounds"], #format: [[525, 527]],
					aspect = aspect)


	
	#TODO: horizontal display
	def horiz_display(self, plot = True):
		assert type(plot) == bool, "plot must be a bool"
		A_pd = pd.DataFrame(self.A_orig)
		A_pd.columns = self.freq_idx
		if plot == True:
			fig, ax = plt.subplots(1,1, figsize = (6,4))
			my_heat= sns.heatmap(A_pd,  center=0, cmap=sns.color_palette("CMRmap"), ax = ax)
			ax.set_xlabel('Frequency (Hz)')
			ax.set_ylabel('time')
		self.A = A_pd.values

	#TODO plot the data



	def build_pd(self, np_, n_series):
		series_len = np_.shape[0]
		for i in range(n_series): 
			id_np =  np.zeros((series_len, 1)).reshape(-1, 1) + i
			series_spec = np_[:, i].reshape(-1, 1)
			t = np.array( list( range( series_len))).reshape(-1, 1)
			pd_spec = np.concatenate( [ t, series_spec, id_np], axis = 1)
			pd_spec = pd.DataFrame(pd_spec)
			pd_spec.columns = ["t", "x", "id"]
			if i == 0:
				df = pd_spec 
			else:
				df = pd.concat([df, pd_spec], axis = 0)
		return(df)


	def plot_timeseries(self, prediction_, train, test, titl = "ESN ", series2plot = 0, method = None, label_loc = (0., 0.)):
		'''
		This function makes three plots:
			the prediction, the residual, the loss.
		It was built for single predictions, but needs to be upgraded to deal with multiple output.
		We need to show: average residual, average loss.
		'''
		full_dat = np.concatenate([train, test], axis = 0); full_dat_avg = np.mean(full_dat, axis = 1)
		n_series, series_len = test.shape[1], test.shape[0]
		assert method in ["all", "single", "avg"], "Please choose a method: avg, all, or single"
		#assert method != "all", "Not yet implimented #TODO"
		
		if method == "single":
			label_loc = (0.02, 0.65)
		
		#key indexes
		trainlen, testlen, pred_shape = train.shape[0], test.shape[0], prediction_.shape[0]
		
		if method == "single":
			if n_series > 1:
				print("There are " + str(n_series) + " time series, you selected time series " 
					+ str(series2plot + 1))
			
			# avoid choosing all of the columns. subset by the selected time series.
			train, test, prediction = train[:, series2plot], test[:, series2plot], prediction_[:, series2plot]
			
			
			# set up dataframe
			xTrTarg_pd = pd.DataFrame(test)
			t = pd.DataFrame(list(range(len(xTrTarg_pd))))
			
			# append time
			Target_pd = pd.concat([xTrTarg_pd, t], axis = 1)
			Target_pd.columns = ["x", "t"]
			
			 #calculate the residual
			resid = test.reshape(-1,)[:pred_shape] - prediction.reshape(-1,) #pred_shape[0]
			
			rmse_spec =  str(round(myMSE(prediction, test), 5))
			full_dat = np.concatenate([train, test], axis = 0)
			
		elif method == "avg":
			rmse_spec =  str(round(myMSE(prediction_, test), 5))
			prediction = prediction_.copy().copy()
			
			def collapse(array):
				return(np.mean(array, axis = 1))

			vals = []
			#y - yhat
			resid_np = test - prediction_
			
			for i in [train, test, prediction_, resid_np]:
				vals.append(collapse(i))
				
			train, test, prediction_avg, resid = vals
			#return(prediction)
		else: ##############################################################################################
			#TODO make a loop and finish this, hopefully pretty colors.
			
			rmse_spec =  str(round(myMSE(prediction_, test), 5))
			
			pd_names = ["Lines", "prediction", "resid"]
			pd_datasets = [ full_dat, prediction_, test - prediction_]
			rez = {}
			
			for i in range(3):
				# TODO functionalize this to streamline the other plots.
				name_spec = pd_names[i]
				dataset_spec = pd_datasets[i]
				rez[name_spec] = build_pd(dataset_spec, n_series)
				
			Lines_pd, resid_pd, prediction_pd = rez["Lines"], np.abs(rez["resid"]), rez["prediction"]
			#display(Lines_pd) #np.zeros((4,1))
		
		####### labels
		if method in ["single"]:	
			plot_titles = [ titl + "__: Prediction vs Ground Truth, rmse_: " + rmse_spec,
						   titl + "__: Prediction Residual",
						   titl + "__: Prediction Loss"]
			plot_labels = [
				["Ground Truth","prediction"]
			]
		elif method == "avg":
			plot_titles = [titl + "__: Avg Prediction vs Avg Ground Truth, total rmse_: " + rmse_spec,
						   titl + "__: Avg Prediction Residual",
						   titl + "__: Avg Prediction Loss"]
			plot_labels = [
				[ "", "Avg Ground Truth", "avg. prediction"]
			]
		elif method == "all":
			plot_titles = [titl + "__: Visualization of Time series to Predict, rmse_: " + rmse_spec,
						   titl + "__: Prediction Residuals", titl + "__: Prediction Loss"
						  ]
		
		### [plotting]	
		
		
		
		#display(Target_pd)
		fig, ax = plt.subplots(3, 1, figsize=(16,10))
		
		i = 0 # plot marker
		j = 0 # subplot line marker
		
		######################################################################## i. (avg.) prediction plot
		if method in ["single", "avg"]:
			
			if method == "single": col, alph = "cyan", 0.5,
			else: col, alph = "grey", 0.3
			
			### ground truth
			ax[i].plot(range(full_dat.shape[0]), full_dat,'k', label=plot_labels[i][j],
					  color = col, linewidth = 1, alpha = alph); j+=1
			
			if method == "avg":
				ax[i].plot(range(full_dat.shape[0]), full_dat_avg,'k', label=plot_labels[i][j],
					  color = "cyan", linewidth = 1, alpha = 0.8); j+=1
				# ground truth style
				ax[i].plot(range(full_dat.shape[0]), full_dat_avg,'k', color = "blue", linewidth = 0.5, alpha = 0.4)
			else:
				# ground truth style
				ax[i].plot(range(full_dat.shape[0]), full_dat,'k', color = "blue", linewidth = 0.5, alpha = 0.4)
			
			
			### prediction
			#pred style, pred
			if method == "single":
				ax[i].plot(range(trainlen,trainlen+testlen), prediction,'k',
						 color = "white",  linewidth = 1.75, alpha = .4)
				ax[i].plot(range(trainlen,trainlen+testlen), prediction,'k',
						 color = "red",  linewidth = 1.75, alpha = .3)
				ax[i].plot(range(trainlen,trainlen+testlen),prediction,'k',
						 label=plot_labels[i][j], color = "magenta",  linewidth = 0.5, alpha = 1); j+=1
			else: #potentially apply this to the all plot as well. Maybe only have two methods.
				ax[i].plot(range(trainlen,trainlen+testlen), prediction,'k',
						 color = "pink",  linewidth = 1.75, alpha = .35)
				ax[i].plot(range(trainlen,trainlen+testlen), prediction_avg,'k',
						 color = "red",  linewidth = 1.75, alpha = .4, label = "prediction avg")		   #first plot labels		   ax[i].set_title(plot_titles[i])		   ax[i].legend(loc=label_loc)		   i+=1; j = 0	   else:		   sns.lineplot( x = "t", y = "x", hue = "id", ax = ax[i], 						data = Lines_pd, alpha = 0.5,						palette = sns.color_palette("hls", n_series))		   ax[i].set_title(plot_titles[i])		   i+=1	   	   if method in ["single", "avg"]:		   ######################################################################## ii. Residual plot		   ax[i].plot(range(0,trainlen),np.zeros(trainlen),'k',					label="", color = "black", alpha = 0.5)		   ax[i].plot(range(trainlen, trainlen + testlen), resid.reshape(-1,),'k',					color = "orange", alpha = 0.5)		   # second plot labels		   #ax[1].legend(loc=(0.61, 1.1))		   ax[i].set_title(plot_titles[i])		   i+=1	   else:		   resid_pd_mn = resid_pd.pivot(index = "t", 										columns = "id", 										values = "x"); resid_pd_mn = resid_pd_mn.mean(axis = 1)	   		   sns.lineplot( x = "t", y = "x", hue = "id", ax = ax[i], data = resid_pd, alpha = 0.35, label = None)		   for j in range(n_series):			   ax[i].lines[j].set_linestyle((0, (3, 1, 1, 1, 1, 1)))#"dashdot")		   		   sns.lineplot(ax = ax[i], data = resid_pd_mn, alpha = 0.9, color = "r",						 label = "mean residual")		   		   ax[i].set_title(plot_titles[i])		   i+=1	   ####################################################################### iii. Loss plot	   if method in ["single", "avg"]:		   		   ax[i].plot(range(0,trainlen),np.zeros(trainlen),'k',					label="", color = "black", alpha = 0.5)		   ax[i].plot(range(trainlen,trainlen+testlen),resid.reshape(-1,)**2,'k',					color = "r", alpha = 0.5)		   # second plot labels		   #ax[2].legend(loc=(0.61, 1.1))		   ax[i].set_title(plot_titles[i])		   	   elif method == "all":		   # create the loss dataframe		   loss_pd = resid_pd.copy(); 		   vals =  loss_pd['x'].copy().copy(); loss_pd['x'] = vals **2		   		   loss_pd_mn = loss_pd.pivot(index = "t", 										columns = "id", 										values = "x"); loss_pd_mn = loss_pd_mn.mean(axis = 1)	   		   sns.lineplot( x = "t", y = "x", hue = "id", ax = ax[i], data = loss_pd, alpha = 0.35, label = None)		   for j in range(n_series):			   ax[i].lines[j].set_linestyle((0, (3, 1, 1, 1, 1, 1)))#"dashdot")		   		   sns.lineplot(ax = ax[i], data =loss_pd_mn, alpha = 0.9, color = "magenta",						 label = "mean loss")		   		   ax[i].set_title(plot_titles[i])		   i+=1	   plt.subplots_adjust(hspace=0.5)
		plt.show()

	def diff(self, first, second):
		second = set(second)
		return [item for item in first if item not in second]

	def my_range2lst(self, response_range):
		"""
		This function takes on two forms: lst and lst_of_lsts
		in the lst form, it simply takes a list [a,b] where a<b ie a numerical range, and converts that into a list
		of all of the values contained by the range.
		The reason we have a function at all is because of the lst_of_lsts option, where it returns multiple ranges.
		"""
		if type(response_range[0]) != list:
			response_range_lst = [response_range]
		else: 
			response_range_lst = response_range
		
		lst_idx = []
		for i, range_ in enumerate(response_range_lst):
			range_start = range_[0]
			range_stop  = range_[1]
			lst_idx += np.sort( np.array( list( range( range_start, range_stop)))).tolist()
		lst_idx = np.sort(np.array(lst_idx)).tolist()
		return(lst_idx)

	def myMSE(prediction,target):
		return np.sqrt(np.mean((prediction.flatten() - target.flatten() )**2))

	def Shape(lst):
		npObj, label = lst; print(label + " shape: " +  str(npObj.shape))

	# validation version
	def get_observers(self, 
					  missing = None,  # missing = self.key_freq_idxs[2000], 
					  aspect = 6,
					  method  = "random", 
					  num_observers = 20,
					  observer_range = None,
					  plot_split = True,
					  response_range = None,
					  split = 0.2
					  ): 
		"""
		arguments:
			aspect: affect the size of the returned plot.
			dataset: obvious
			method: 
				(+) random 
				(+) equal #similar to barcode, equal spacing, with k missing block. Low priority.
				(+) block
				(+) barcode #TODO block but with gaps between observers.
					# I think this will show that you don't really need every line of the data to get similar accuracy
			
			missing: either 
				(+) any integer:  (standing for column of the spectogram) or 
				(+) "all" : which stands for all of the remaining target series.
			num_observers: the number of observers that you want if you choose the "random" method.
			observer_range: if you select the "block" opion
		"""
		#preprocessing:
		dataset, freq_idx  = self.A,  self.f
		n_rows, n_cols = dataset.shape[0], dataset.shape[1]
		train_len = int(n_rows * split)
		test_len =  n_rows - train_len
		col_idx = list(range(n_cols))

		self.split = split
		
		#remove the response column which we are trying to use for inpainting
		if method == "random":
			col_idx.remove(missing)
			obs_idx = np.random.choice(col_idx, num_observers, replace = False)
			response  = dataset[ : , missing].reshape(-1,1)
			response_idx = [missing]
			
		elif method == "eq":
			print("equal spacing")
			print("NOT YET IMPLIMENTED")
			
		elif method == "all":
			obs_idx = np.random.choice( col_idx, num_observers, replace = False)
			response_idx  = diff( col_idx, obs_idx.tolist())
			response  = dataset[ : , response_idx]
		
		### BLOCK: this is oldschool and super-annoying: you have to specify indices.
		elif method == "block":
			"""
			This method either blocks observers and/or the response area.
			"""
			print("you selected the block method")
			if response_range == None:
				response_idx  = [missing]
				response	  = dataset[ : , missing].reshape( -1, 1)
			else:
				
				response_idx =  self.my_range2lst(response_range)
				response = dataset[ : , response_idx].reshape( -1, len( response_idx))
				
			for resp_idx_spec in response_idx:
				col_idx.remove( resp_idx_spec)
			
			if observer_range == None:
				col_idx.remove( missing)
				obs_idx = np.sort( np.random.choice( col_idx, 
													num_observers, 
													replace = False))
			else:
				obs_idx = self.my_range2lst(observer_range)
				
			# check for problems with the block method:
			union_obs_resp_set = set(obs_idx) & set(response_idx)
			err_msg = "Error: overlap in obs_idx and response_idx \n"
			err_msg += "overlap: " + str(list(union_obs_resp_set))
			assert list(union_obs_resp_set) == [], err_msg

		elif method == "freq":
			"""
			The newest method, the only one we care to have survive because it is not based on indices but rather desired Hz.
			This method is just like simple_block but upgraded to take in only frequencies by using the helper function hz2freq which must
			be called first.
			"""
			obs_idx  = self.resp_obs_idx_dict["obs_idx"]
			response_idx = self.resp_obs_idx_dict["resp_idx"]
			assert type(obs_idx) != type(None), "oops, your observer index cannot be None, first run hz2idx helper function"
			assert type(response_idx) != type(None), "oops, your response index cannot be None"
			response = dataset[ : , response_idx].reshape( -1, len( response_idx))



		assert method == "freq", "at this time only use the freq method."
				
		
		# PARTITION THE DATA
		observers = dataset[ : , obs_idx]

		observers_tr, observers_te = observers[ :train_len, : ], observers[ train_len  , : ]

		response_tr, response_te = response[ :train_len, : ], response[ train_len , : ]

		
		### Visualize the train test split and the observers
		if plot_split == True:
			red, yellow, blue, black = [255, 0, 0], [255, 255, 0], [0, 255, 255], [0, 0, 0]
			orange, green, white = [255, 165, 0], [ 0, 128, 0], [255, 255, 255]

			#preprocess:
			split_img = np.full(( n_rows, n_cols, 3), black)

			# assign observer lines
			for i in obs_idx:
				split_img[ : , i] = np.full(( 1, n_rows, 3), yellow)

			# assign target area
			for i in response_idx:
				split_img[ :train_len, i] = np.full(( 1, train_len, 3), blue)
				split_img[ train_len:, i] = np.full(( 1, test_len,  3), red)

			legend_elements = [Patch(facecolor='cyan', edgecolor='blue', label='Train'),
						   	   Patch(facecolor='red', edgecolor='red', label='Test'),
						       Patch(facecolor='yellow', edgecolor='orange', label='Observers')]
			
			
			# Create the figure
			fig, ax = plt.subplots( 1, 2, figsize = ( 12, 6))
			ax = ax.flatten()
			
			
			solid_color_np =  np.transpose(split_img.T, axes = (1,2,0))
			
			#solid_color_pd.index = freq_idx
			
			# The legend:
			#https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
			
			
			##################################### START plots
			
			#++++++++++++++++++++++++++++++++++++ plot 1: sns heatmap on the right
			self.olab_display(ax[1])
			ax[1].set_title("Spectogram Data")
			
			# retrieve labels to share with plot 0
			# We need to retrieve the labels now.
			plt.sca(ax[1])
			locs, labels = plt.yticks()
			freq_labels = np.array([int(label.get_text()) for label in labels])
			
			#++++++++++++++++++++++++++++++++++++ plot 0: diagram showing training, test splits and observers. LHS
			ax[0].set_title("Dataset Split Visualization")
			ax[0].imshow(solid_color_np, aspect = aspect)
			
			### fixing labels on plot 0, involved!
			# label axes, legend
			ax[0].set_ylabel('Frequency (Hz)'); ax[0].set_xlabel('time')
			ax[0].legend(handles=legend_elements, loc='lowerright')
			
			#now calculate the new positions
			max_idx = solid_color_np.shape[0]
			
			#new positions
			new_p = (freq_labels/self.max_freq) * max_idx 
			adjustment = max_idx - np.max(new_p); new_p += adjustment -1; new_p  = np.flip(new_p)
			plt.sca(ax[0]); plt.yticks(ticks = new_p, labels = freq_labels)
			###
			
			plt.show()
			
			##################################### END plots
			
			# print dimensions ect.
			print_lst =  [(observers_tr, "X target"), (observers_te, "X test")]
			print_lst += [(response_tr, "response train"), (response_te, "response test")]
				
			for i in print_lst:
				Shape(i)
			
			print("observer_range: " + str(observer_range))
			if response_idx == None:
				print("target index: " + str(missing))
			else:
				print("response range: " + str(response_range))

			
		self.dat = {"obs_tr"  : observers_tr, 
				"obs_te"  : observers_te,
				"resp_tr" : response_tr,
				"resp_te" : response_te,
				"obs_idx" : obs_idx,
				"resp_idx" : response_idx}
		self.Train, self.Test = self.dat["obs_tr"], self.dat["obs_te"]
		self.xTr, self.xTe = self.dat["resp_tr"], self.dat["resp_te"]
		print("total observers: " + str(len(self.dat["obs_idx"])))
		print("total targets: " + str(len(self.dat["resp_idx"])))

		
		self.outfile = "experiment_results/" + str(int(self.target_frequency / 1000)) + "k/" + self.size + "/split_" + str(split)  +"/" + "target_" + str(self.target_kHz) +"kh"
	

	""" #TODO Fix these data saving functions to work with the new file system. Some thoughts:
	# this should be easy. get_new_filename can be eliminated, self.outfile serves the same purpose.
	# The hard part will be to look at how to save the data. Perhaps save the entire experiment object? Thoughts.

	def count_files(path, current):
	    count = 0
	    for path in pathlib.Path(path).iterdir():
	        if path.is_file():
	            count += 1
	    if current:
	        count = count - 1
	    return("_" + str(count))


	def get_new_filename(exp, 
	                     obs = len(dat["obs_idx"]), 
	                     target_freq = "2k",
	                     ctr = key_freq_idxs[2000],
	                     spread = target_freq_["spread"],
	                     current = False
	                    ):
	    '''
	    ideally this function will serve two purposes: it will return a new filename and return 
	    a dict of data so that we can recreate the experiment. 
	    This should include 
	        1) The obs and resp indices, the "best_arguments" (the optimized hyper-parameters),
	        and the prediction.
	    '''
	    if exp:
	        prefix = 'exp_w'
	    else: 
	        prefix = 'non_exp_w'
	    obs, ctr, spread  = str(obs), str(ctr), str(spread)
	    
	    new_dir = "results/" + size + "/" + target_freq + "/"
	    count = count_files(new_dir, current = current)
	    new_filename =  prefix  + count + ".txt"
	    return(new_dir +  new_filename )

	def getData2Save(): #best_arguments, prediction = obs_prediction
	    '''
	    Save the data
	    current issue: how do we initialize this function properly?
	    '''
	    err_msg = "YOU NEED TO CALL THIS FUNCTION LATER "
	    json2be = {}
	    
	    
	    
	    # 1) saving the structure of the data and split
	    json2be["basic_info"] = {"size" : size, 
	                             "freq" : freq,
	                        "target_freq_" : target_freq_, 
	                        "n_obs" : len(dat["obs_idx"]),
	                        "n_target" : len(dat["resp_idx"]),
	                        "split_cutoff" : dat["resp_tr"].shape[0]}
	    #jsonify dat
	    new_dat = dat.copy().copy()
	    for key, item in new_dat.items():
	        if type(item) != list:
	            new_dat[key] = item.tolist()
	    
	    json2be["dat"] = new_dat
	    
	    
	    # 2) saving the optimized hyper-parameters
	    try:
	        best_arguments
	    except NameError:
	         err_msg + "RC not yet trained"
	    else:
	        json2be["best_arguments"] = best_arguments
	    
	    # 3) saving the prediction, mse
	    try:
	        obs_prediction
	    except NameError:
	         err_msg + "obs_prediction not yet created"
	    else:
	        json2be["prediction"] = obs_prediction.tolist()
	        mse = my_MSE(obs_prediction, dat["resp_te"], verbose = False)
	        json2be["results"] = {
	            "MSE" :  mse,
	            "RMSE" : np.sqrt(mse)
	        }
	    return(json2be)

	def save_json(exp):
	    save_spec_ = getData2Save()
	    new_file = get_new_filename(exp = exp)
	    with open(new_file, "w") as outfile:
	        data = json.dump(save_spec_, outfile)
	        
	def my_MSE(prediction, truth, verbose = True, label = ""):
	    mse_matrix = (prediction - truth)**2
	    mse = np.sum(mse_matrix)/(mse_matrix.shape[0]*mse_matrix.shape[1])
	    if verbose == True:
	        print(label + " MSE: " + str(mse))
	    return(mse)
	"""

	"""
	def currTime():
		now = datetime.now()

		current_time = now.strftime("%H:%M:%S")
		print("Current Time =", current_time)
	currTime()
	"""
#TODO THIS NEEDS EDITING, is mostly useless.
complex_dict = {
"small" : { 

#target_frequencies
"2k" : 101,
"4k" : 206,
"8k" : 307,

#target spread sizes
"no_spread" : None,
"small_spread" : 4,
"medium_spread" : 12,
"large_spread" : 24,

#observer values
"small_obs" : 10,
"medium_obs" : 25,
"large_obs" : 50
},
"medium" : { 

#target_frequencies
"2k" : 101,
"4k" : 206,
"8k" : 307,

#target spread sizes
"no_spread" : None,
"small_spread" : 4,
"medium_spread" : 12,
"large_spread" : 24,

#observer values
"small_obs" : 10,
"medium_obs" : 25,
"large_obs" : 50
},

"publish": {
"2k" : 546,
"4k" : 1089,
"8k" : 2177,
"0.5_sec" : 1371,
"0.7_sec" : 1924
}
}

