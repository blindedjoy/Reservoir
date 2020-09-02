#Herein we shall create a new file, similar to esn.py 
#where we transform the notebook into an object oriented approach worthy of Reinier's library.
from reservoir import *
from PyFiles.imports import *
from scipy.interpolate import Rbf
import pickle




def Merge(dict1, dict2): 
	res = {**dict1, **dict2} 
	return res 


def nrmse(pred_, truth, columnwise = False):
	"""
	inputs should be numpy arrays
	variables:
	pred_ : the prediction numpy matrix
	truth : ground truth numpy matrix
	columnwise: bool if set to true takes row-wise numpy array (assumes reader thinks of time as running left to right
		while the code actually runs vertically.)

	"""
	if columnwise:
		rmse_ = np.sum((truth - pred_) ** 2, axis = 1).reshape(-1, )
		denom_ = np.sum(truth ** 2) * (1/len(rmse_))#np.sum(truth ** 2, axis = 1).reshape(-1, )
	else:
		rmse_ = np.sum((truth - pred_) ** 2)
		denom_ = np.sum(truth ** 2)
	
	nrmse_ = np.sqrt(rmse_ / denom_)
	return(nrmse_)



def idx2Freq(val):
	idx = min(range(len(f)), key=lambda i: abs(f[i]-val))
	return(idx)

def ifdel(dictt, key):
	if key in list(dictt.keys()):
		del dictt[key]
	return(dictt)



def pp(variable, label):
	"""
	custom print function
	"""
	print(label +": " + str(variable))
def Shape(lst):
	npObj, label = lst; print(label + " shape: " +  str(npObj.shape))

def is_numeric(x):
	booll = (type(x) == float) or (type(x) == int)
	return(booll)

#seems useless, but we will see.
def class_copy(class_spec):
	CopyOfClass = type('esn_cv_copy', class_spec.__bases__, dict(class_spec.__dict__))
	return class_spec


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
				 file_path = "spectrogram_data/", 
				 target_frequency = None,
				 out_path = None,
				 obs_hz = None, 
				 target_hz = None, 
				 train_time_idx = None,
				 test_time_idx = None,
				 verbose = True,
				 smooth_bool = False,
				 interpolation_method = "griddata-linear",
				 prediction_type = "block",
				 librosa = False,
				 spectrogram_path = None,
				 flat = False,
				 obs_freqs  = None,
				 target_freqs = None
				 ):
		# Parameters
		self.size = size
		self.flat = flat

		self.bounds = {"observer_bounds" : None, "response_bounds" : None} 
		self.esn_cv_spec = class_copy(EchoStateNetworkCV)
		self.esn_spec	= class_copy(EchoStateNetwork)
		self.file_path = file_path + self.size + "/"
		self.interpolation_method = interpolation_method
		self.json2be = {}
		self.librosa = librosa

		self.out_path = out_path
		self.prediction_type = prediction_type
		self.smooth_bool = smooth_bool
		self.spectrogram_path = spectrogram_path
		
		self.verbose = verbose
		

		if obs_freqs:
			self.target_frequency =  float(np.mean(target_freqs))
		else:
			self.target_frequency = target_frequency
		

		#these indices should be exact lists, not ranges.
		if train_time_idx:
			self.train_time_idx = train_time_idx
		if test_time_idx:
			self.test_time_idx  = test_time_idx




		assert self.target_frequency, "you must enter a target frequency"
		assert is_numeric(self.target_frequency), "you must enter a numeric target frequency"
		assert size in ["small", "medium", "publish", "librosa"], "Please choose a size from ['small', 'medium', 'publish']"
		assert type(verbose) == bool, "verbose must be a boolean"
		
		#order dependent attributes:
		self.load_data()
		if self.prediction_type == "block":
			"BONK"
			if obs_freqs:
				self.obs_idx  = [self.Freq2idx(freq) for freq in obs_freqs]
				self.resp_idx = [self.Freq2idx(freq) for freq in target_freqs]
			if obs_hz and target_hz:
				assert is_numeric(obs_hz), "you must enter a numeric observer frequency range"
				assert is_numeric(target_hz), "you must enter a numeric target frequency range"
			if not target_freqs:
				self.hz2idx(obs_hz = obs_hz, target_hz = target_hz)
		
		self.horiz_display()

	def save_pickle(self, path, transform):
		self.getData2Save()
		save_path = path # + ".pickle"
		with open(save_path, 'wb') as handle:
			pickle.dump(transform, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def hz2idx(self, 
		   	   obs_hz = None, 
		   	   target_hz = None, 
		   	   silent = True):
		"""
		This function acts as a helper function to simple_block and get_observers
		and is thus designed. It takes a desired hz amount and translates that to indices of the data.
		
		To do one frequency use Freq2idx.
		"""
		#print("RUNNING HZ2IDX")

		midpoint = self.target_frequency 
		height   = self.freq_axis_len 

		#items needed for the file name:
		self.obs_kHz, self.target_kHz = obs_hz / 1000, target_hz / 1000

		# spread vs total hz
		obs_spread, target_spread = obs_hz / 2, target_hz / 2
		### START helper functions
		def my_sort(lst):
			return(list(np.sort(lst)))
		def drop_overlap(resp_lst, obs_lst): #obs_lst = drop_overlap(resp_lst, obs_lst)

			intersection_set = set.intersection(set(resp_lst), set(obs_lst))
			#find intersection of list1 and list2

			intersection_list = list(intersection_set)

			#print(intersection_list)
			for i in intersection_list:
				obs_lst.remove(i)
			#print(obs_lst)
			return(obs_lst)

		def endpoints2list(lb, ub, obs = False, obs_spread = obs_spread, height = height): # [lb, ub] stands for [lowerbound, upperbound]

			#if self.librosa:
			f = np.array(self.f)

			def librosa_range(lb_, ub_):
				"""
				takes lower and upper bounds and returns the indices list. Potentially could make the else statement
				below obselete.
				"""
				bounds = my_sort([lb_, ub_])
				lb, ub = bounds
				lb_bool_vec, ub_bool_vec = (f >= lb), (f <= ub)
				and_vector = ub_bool_vec * lb_bool_vec

				freqs = f[and_vector]			#frequencies between bounds
				freq_idxs = np.where(and_vector)[0].tolist() #indices between bounds
				return(freq_idxs)

			
			resp_range = librosa_range(lb, ub)
			respLb, respUb = f[resp_range][0], f[resp_range][-1]
			#print("respLb, respUb: " + str(respLb) + ", " +  str(respUb))
			obs_hLb, obs_hUb = respUb, respUb + obs_spread 
			#print("obs_hLb, obs_hUb : " + str(obs_hLb)  + ", " + str(obs_hUb))

			obs_lLb, obs_lUb = respLb - obs_spread, respLb 
			#print("obs_lLb, obs_lUb : " + str(obs_lLb)  + ", " + str(obs_lUb))
			
			obs_L, obs_H = librosa_range(obs_lLb, obs_lUb), librosa_range(obs_hLb, obs_hUb )
			#print("Obs_H" + str(obs_H))
			#print("resp_range" + str(resp_range))
			#drop the lowest index of obs_H and the highest index of obs_L to avoid overlap with resp_idx
			obs_H = drop_overlap(resp_lst = resp_range, obs_lst = obs_H)
			obs_L = drop_overlap(resp_lst = resp_range, obs_lst = obs_L)
				#print("Obs_H after haircut: " + str(obs_H))
				

			"""	
			else:
				def make_range(lb_, ub_):
					return list(range(int(lb), int(ub + 1)))
				
				# response ranges
				lb, ub = self.Freq2idx(lb), self.Freq2idx(ub)
				resp_idx_range = make_range(lb, ub)
				print(resp_idx_)
				resp_freq_range = [self.f[idx] for idx in resp_idx_range]
				

				# observer ranges
				obs_hUb, obs_hLb = respUb + self.Freq2idx(obs_spread) + 1, respUb + 1 ##uUB: high range upper bound
				obs_lLb, obs_lUb = respLb - self.Freq2idx(obs_spread) - 1, respLb - 1
				obs_L, obs_H = make_range(obs_lLb, obs_lUb), make_range(obs_hLb, obs_hUb )
			"""
				
			ranges = (resp_range, obs_L, obs_H ) 
			#enforce integer type
			final_ranges = []
			for range_ in ranges:
				range_ = [int(idx) for idx in range_]
				range_ = [height - i for i in range_]
				range_ = my_sort(range_)
				final_ranges.append(range_)
			#Inversion:

			return(ranges)
		### END helper functions

		
		
		
		# get the obs, response range endpoints
		resp_freq_Lb, resp_freq_Ub = [midpoint - target_spread, midpoint + target_spread]
		

		#This could be incorrect but I'm trying to protect the original block method.
		#respLb, respUb = [self.Freq2idx(midpoint - target_spread), self.Freq2idx(midpoint + target_spread)]

		#print("frequencies: " + str([ round(x,1) for x in self.f]))
		#print("bounds2convert: (" +str(midpoint - target_spread ) + ", " + str(midpoint + target_spread ) + str(")"))

		#current_location

		# Listify:
		resp_idx_Lst, obs_idx_Lst1, obs_idx_Lst2 = endpoints2list(resp_freq_Lb, resp_freq_Ub)

		def get_frequencies(idx_lst):
			freq_lst = [self.f[idx] for idx in idx_lst]
			return freq_lst

		for i, idx_lst in enumerate([resp_idx_Lst, obs_idx_Lst1, obs_idx_Lst2]):
			if not i:
				freq_lst = []
			freq_lst += [get_frequencies(idx_lst)]

		resp_freq_Lst, obs_freq_Lst1, obs_freq_Lst2 = freq_lst

		
		if not silent:
			print("resp_indexes : " + str(resp_idx_Lst))
			print("observer frequencies upper domain: " + str(resp_freq_Lst) + 
				  " , range: "+ str(abs(resp_Freq_Lst[0] - resp_freq_Lst[-1])) +" Hz\n")

			print("observer indexes lower domain: " + str(obs_idx_Lst1))
			print("observer frequencies lower domain: " + str(obs_freq_Lst1) + 
				  " , range: "+ str(abs(obs_Freq_Lst1[0] - obs_freq_Lst1[-1])) +" Hz\n")

			print("observer indexes upper domain: " + str(obs_idx_Lst2))
			print("observer frequencies upper domain: " + str(obs_freq_Lst2) + 
				  " , range: "+ str(abs(obs_Freq_Lst2[0] - obs_freq_Lst2[-1])) +" Hz\n")
		
		#if not self.librosa:
		#	assert obs_idx_Lst2 + resp_idx_Lst + obs_idx_Lst1 == list(range(obs_idx_Lst2[ 0 ], obs_idx_Lst1[ -1] + 1))

		dict2Return = {"obs_idx": obs_idx_Lst2 + obs_idx_Lst1, 
					   "resp_idx": resp_idx_Lst,
					   "obs_freq" : obs_freq_Lst1 + obs_freq_Lst2,
					   "resp_freq" : resp_freq_Lst}

		self.resp_obs_idx_dict = dict2Return
		self.obs_idx  = [int(i) for i in dict2Return["obs_idx"]]
		self.resp_idx = [int(i) for i in dict2Return["resp_idx"]]


	def smooth(self):
		from scipy.ndimage import gaussian_filter
		#from scipy.ndimage import gaussian_filter
		self.A = gaussian_filter( self.A, sigma = 1)

	def load_data(self, 
				  smooth = True, 
				  log = False, 
				  method = ("librosa", "db")):
		
		if self.librosa:
			spectrogram_path = "./pickle_files/spectrogram_files/" + self.spectrogram_path + ".pickle"
			with open(spectrogram_path, 'rb') as handle:

				pickle_obj = pickle.load(handle)

			self.f = pickle_obj["transform"]["f"].reshape(-1,).tolist()

			self.spectrogram_type = method[1]
			assert self.spectrogram_type in ["power", "db"]

			if self.spectrogram_type == "power":
				self.A_unnormalized = pickle_obj["transform"]["Xpow"]
			else:
				self.A_unnormalized = pickle_obj["transform"]["Xdb"]
		else:
			spect_files  = { "publish" : "_new", "small" : "_512" , "original" : "", "medium" : "_1024"}

			files2import = [self.file_path  + i + spect_files[self.size] for i in ("T", "f", "Intensity") ]
			
			data_lst = []
			for i in files2import:
				data_lst.append(loadmat(i))

			self.T, self.f, self.A = data_lst #TODO rename T, f and A (A should be 'spectrogram' or 'dataset')

			self.A_unnormalized = self.A['M'].copy()

			#preprocessing
			self.T = self.T['T']
			self.T = np.transpose(self.T)
			self.f = self.f['f'].reshape(-1,).tolist()

		self.A =  self.A_unnormalized.copy()
		#print("A_shape":self.A.shape)
		self.A =  (self.A - np.mean(self.A)) / np.std(self.A)

		self.A_orig = self.A.copy()

		if self.smooth_bool:
			self.smooth()

		self.A_orig = self.A.copy()
		self.A_orig = np.rot90(self.A_orig, k = 1, axes = (0, 1))
		

		
		if log:
			self.f = np.log(self.f)
		self.max_freq = int(np.max(self.f))
		self.Freq2idx(self.target_frequency, init = True)

		self.freq_axis_len = self.A.shape[0]
		self.time_axis_len = self.A.shape[1]
		str2print = ""
		if self.verbose:
			for file_name_ in files2import:
				str2print += "successfully loaded: " + file_name_ + ".mat, "
			print("maximum frequency: " + str(self.max_freq))
			print("dataset shape: " + str(self.A.shape))

		if log:
			self.freq_idx = [float(i) for i in self.f]
		else:
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
		if return_index:
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
		if target_timeseries:  
			#TODO EVEN AND ODD TIMESERIES
			target_spread = target_timeseries // 2
			#resp_bounds is the response bounds ie the target area bounds.
			resp_bounds = [ctr - target_spread, ctr + target_spread] 
		else: #TODO does this part of the if-else statement actually do anything?
			response_bounds = None
			resp_bounds = [ctr, ctr]
		assert n_obs, "if you want to have no observers then #TODO"
		
		obs_bounds  = [[resp_bounds[0] - n_obs, resp_bounds[0]],
					   [resp_bounds[1], resp_bounds[1] + n_obs ]]

		if not silent:
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
	def horiz_display(self, plot = False):
		assert type(plot) == bool, "plot must be a bool"
		A_pd = pd.DataFrame(self.A_orig)
		A_pd.columns = self.freq_idx
		if plot:
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


	def plot_timeseries(self, 
						titl = "ESN ", 
						series2plot = 0, 
						method = None, 
						label_loc = (0., 0.)): #prediction_, train, test, 
		'''
		This function makes three plots:
			the prediction, the residual, the loss.
		It was built for single predictions, but needs to be upgraded to deal with multiple output.
		We need to show: average residual, average loss.
		'''
		prediction_ = self.prediction
		train = self.Train
		test  = self.Test


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
			rmse_spec =  str(round(nrmse(prediction_, test), 5))
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
			
			rmse_spec =  str(round(nrmse(prediction_, test), 5))
			
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

	

	# validation version
	def get_observers(self, 
					  missing = None,  # missing = self.key_freq_idxs[2000], 
					  aspect = 6,
					  method  = "random", 
					  num_observers = 20,
					  observer_range = None,
					  plot_split = False,
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
				(+) any integer:  (standing for column of the spectrogram) or 
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
		self.method = method
		self.aspect = aspect
		
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
			
			assert union_obs_resp_set, err_msg # check if the list is empty

		elif method == "freq":
			"""
			The newest method, the only one we care to have survive because it is not based on indices but rather desired Hz.
			This method is just like simple_block but upgraded to take in only frequencies by using the helper function hz2freq which must
			be called first.
			"""
			obs_idx  = self.resp_obs_idx_dict["obs_idx"]
			response_idx = self.resp_obs_idx_dict["resp_idx"]
			#assert type(obs_idx) != type(None), "oops, your observer index cannot be None, first run hz2idx helper function"
			#assert type(response_idx) != type(None), "oops, your response index cannot be None"
			assert obs_idx, "oops, your observer index cannot be None, first run hz2idx helper function"
			assert response_idx, "oops, your response index cannot be None"

			response = dataset[ : , response_idx].reshape( -1, len( response_idx))

		elif method == "exact":
			"""
			The newest method, the only one we care to have survive because it is not based on indices but rather desired Hz.
			This method is just like simple_block but upgraded to take in only frequencies by using the helper function hz2freq which must
			be called first.
			"""
			
			if self.prediction_type == "block":
				obs_idx  = self.obs_idx
				response_idx = self.resp_idx
				assert obs_idx, "oops, your observer index cannot be None, first run hz2idx helper function"
				assert response_idx, "oops, your response index cannot be None"
				response = dataset[ : , response_idx].reshape( -1, len( response_idx))

			elif self.prediction_type == "column":
				obs_idx, response_idx  = [], range(self.A.shape[1])
				train_time_idx  = self.train_time_idx
				test_time_idx   = self.test_time_idx
				self.target_kHz = "all"
				self.obs_kHz	= 0.0


		assert method in  ["freq", "exact"], "at this time only use the 'freq' method for cluster, \
												  'exact' for analysis"
				
		if self.prediction_type != "column":
			# PARTITION THE DATA
			#print(obs_idx)
			observers = dataset[ : , obs_idx]

			observers_tr, observers_te = observers[ :train_len, : ], observers[ train_len:  , : ]

			response_tr, response_te = response[ :train_len, : ], response[ train_len: , : ]
		else:
			#no observers so assign empty arrays
			obs_arrs = [np.array([[]]) for i in [1, 2, 3]]
			observers, observers_tr, observers_te = obs_arrs

			response_tr  = dataset[ self.train_time_idx, : ]
			response_te = dataset[ self.test_time_idx , : ]

			train_len = len(self.train_time_idx)
			test_len = len(self.test_time_idx)

		### Visualize the train test split and the observers
		if plot_split:
			red, yellow, blue, black = [255, 0, 0], [255, 255, 0], [0, 255, 255], [0, 0, 0]
			orange, green, white = [255, 165, 0], [ 0, 128, 0], [255, 255, 255]

			#preprocess:
			split_img = np.full(( n_rows, n_cols, 3), black)

			# assign observer lines
			for i in obs_idx:
				split_img[ : , i] = np.full(( 1, n_rows, 3), yellow)

			if self.prediction_type != "column":
				# assign target area
				for i in response_idx:
					split_img[ :train_len, i] = np.full(( 1, train_len, 3), blue)
					split_img[ train_len:, i] = np.full(( 1, test_len,  3), red)
			else:
				for i in response_idx:
					split_img[ train_time_idx, i ] = np.full(( 1, train_len, 3), blue)
					split_img[ test_time_idx,  i ] = np.full(( 1, test_len,  3), red)

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
			ax[1].set_title("spectrogram Data")
			
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
			ax[0].legend(handles=legend_elements, loc='lower right')
			
			#now calculate the new positions
			max_idx = solid_color_np.shape[0]
			
			#new positions
			new_p = (freq_labels/self.max_freq) * max_idx 
			adjustment = max_idx - np.max(new_p); new_p += adjustment -1; new_p  = np.flip(new_p)
			plt.sca(ax[0]); plt.yticks(ticks = new_p, labels = freq_labels)
			###
			
			plt.show()
			
			##################################### END plots
			
		self.dat = {"obs_tr"  : observers_tr, 
					"obs_te"  : observers_te,
					"resp_tr" : response_tr,
					"resp_te" : response_te,
					"obs_idx" : obs_idx,
					"resp_idx" : response_idx}
		self.Train, self.Test = self.dat["obs_tr"], self.dat["obs_te"]
		self.xTr, self.xTe = self.dat["resp_tr"], self.dat["resp_te"]
		
		if self.prediction_type == "column":
			self.Train, self.Test = np.ones(self.xTr.shape), np.ones(self.xTe.shape)


		self.runInterpolation()

		# print statements:
		if self.verbose:
			print_lst =  [(observers_tr, "X target"), (observers_te, "X test")]
			print_lst += [(response_tr, "response train"), (response_te, "response test")]
				
			for i in print_lst:
				Shape(i)
			print("total observers: " + str(len(self.dat["obs_idx"])))
			print("total targets: " + str(len(self.dat["resp_idx"])))
		
			if method != "freq":
				print("observer_range: " + str(observer_range))
				if not response_idx:
					print("target index: " + str(missing))
				else:
					print("response range: " + str(response_range))
		

		#assert self.xTr.shape[1] == self.xTe.shape[1], "something is broken, xTr and xTe should have the same column dimension"
		
		
		#self.outfile = "experiment_results/" + str(int(self.target_frequency / 1000)) + "k/" + self.size
		#self.outfile += "/split_" + str(split)  +"/" + "targetKhz:_" + str(self.target_kHz) + "__obskHz:_"

		#self.outfile += str(self.obs_kHz)



		#"targetKhz:_{}__obskHz:_{}"


	def getData2Save(self): 
		'''
		Save the data
		current issue: how do we initialize this function properly?
		'''

		def jsonMerge(new_dict):
			self.json2be = Merge(self.json2be, new_dict)

		if self.json2be == {}:
			print("initialiazing json2be")
			#self.runInterpolation() 
			ip_pred = {"interpolation" : self.ip_res["prediction"].tolist()}
			ip_nrmse = {"interpolation" : self.ip_res["nrmse"]}
			jsonMerge({"prediction" : ip_pred})
			jsonMerge({"nrmse" : ip_nrmse})
			jsonMerge({"best arguments" : {}})
			self.json2be["obs_idx"] = self.obs_idx
			self.json2be["resp_idx"] = self.resp_idx

			#print("json2be after initi: ")
			#print(self.json2be)


		err_msg = "YOU NEED TO CALL THIS FUNCTION LATER "

		
		# 1) Here stored are the inputs to 
		self.json2be["experiment_inputs"] = {
			 "size" : self.size, 
			 "target_frequency" : int(self.target_frequency),
			 "obs_hz" :	float(self.obs_kHz)	* 1000,
			 "target_hz" : float(self.target_kHz) * 1000,
			 "verbose" : self.verbose,
			 }
		self.json2be["get_observer_inputs"] = {
				"method" : self.method,
				"split" : self.split,
				"aspect" : float(self.aspect)
			}
								 #"target_freq_" : target_freq_, 
								 #"num_observer_timeseries" : len(dat["obs_idx"]),
								 #"num_target_timeseries" : len(dat["resp_idx"]),
								 #"split_cutoff" : dat["resp_tr"].shape[0]}
		
		# TODO: REWRITE THE BELOW, dat no longer makes sense as a way to save data.

		#1) jsonify dat
		new_dat = self.dat.copy().copy()
		for key, item in new_dat.items():
			if type(item) != list:
				new_dat[key] = item.tolist()
				if type(new_dat[key]) == int:
					new_dat[key] = [int(item) for item in new_dat[key]]

				if type(new_dat[key]) == float:
					new_dat[key] = [float(item) for item in new_dat[key]]
		
		#json2be["dat"] = new_dat
		
		# 2) saving the optimized hyper-parameters, nrmse

		#print("json2be before pred assignment: ")
		#print(self.json2be)
		try:
			self.best_arguments
		except NameError:
			err_msg + "#TODO Err message"

		args2export = self.best_arguments

		#data assertions, cleanup
		if self.model == "exp":
			assert self.esn_cv.exp_weights# == True
		elif self.model == "hybrid":
			assert self.esn_cv.exp_weights# == True
		elif self.model == "uniform":
			assert not self.esn_cv.exp_weights# == False
			args2export = ifdel(args2export, "llambda")
			args2export = ifdel(args2export, "noise")


		



		pred = self.prediction.tolist()
		self.json2be["prediction"]= Merge(self.json2be["prediction"], {self.model: pred}) #Merge(self.json2be["prediction"], )
		self.json2be["nrmse"][self.model] = nrmse(pred, self.xTe, columnwise = False)
		self.json2be["best arguments"] = Merge(self.json2be["best arguments"], {self.model: args2export}) 

		
			

	
	def RC_CV(self, cv_args, model, hybrid_llambda_bounds = (-5, 1)): #TODO: change exp to 
		"""
		example bounds:
		bounds = {
			'llambda' : (-12, 1), 
			'connectivity': 0.5888436553555889, #(-3, 0)
			'n_nodes': (100, 1500),
			'spectral_radius': (0.05, 0.99),
			'regularization': (-12, 1),

			all are log scale except  spectral radius and n_nodes
		}
		example cv args:

		cv_args = {
			bounds : bounds,
			initial_samples : 100,
			subsequence_length : 250, #150 for 500
			eps : 1e-5,
			cv_samples : 8, 
			max_iterations : 1000, 
			scoring_method : 'tanh',
			exp_weights : False,
		}
		#esn_cv_spec equivalent: EchoStateNetworkCV
		"""
		self.model = model
		assert self.model in ["uniform", "exponential", "hybrid"], self.model + " model not yet implimented"

		if self.model in ["hybrid", "uniform"]:
			exp = False
			exp_w_ = {'exp_weights' : False}

		elif self.model == "exponential":
			self.exp = True
			exp_w_ = {'exp_weights' : True}

		### hacky intervention:
		if self.prediction_type != "column":
		
			predetermined_args = {
				'obs_index' : self.obs_idx,
				'target_index' : self.resp_idx
			}
			
			input_dict = { **cv_args, 
						   **predetermined_args,
						   **exp_w_}
		else:
			input_dict = { **cv_args, 
						   **exp_w_}

		# subclass assignment: EchoStateNetworkCV
		self.esn_cv = self.esn_cv_spec(**input_dict)


		if self.model in ["exponential", "uniform"]:
				print(self.model + "rc cv set, ready to train")
		else:
			print("training hybrid part one: finding unif parameters")
		
		self.best_arguments =  self.esn_cv.optimize(x = self.Train, y = self.xTr) 
		

		if self.model == "hybrid":
			print("old input dict: ")
			print(input_dict)
			old_bounds, old_bounds_keys  = cv_args["bounds"], list(cv_args["bounds"].keys())
			
			new_bounds = {}
			not_log_adjusted = ["n_nodes", "spectral_radius"]
			for i in old_bounds_keys:
				if i not in not_log_adjusted:
					print("adjusting " + i)
					new_bounds[i] = float(np.log(self.best_arguments[i])/np.log(10))
				else:
					new_bounds[i] = self.best_arguments[i]

			new_bounds['llambda'] = hybrid_llambda_bounds
			

			cv_args["bounds"] = new_bounds

			#print("HYBRID, New Bounds: " + str(cv_args["bounds"]))
			#print(cv_args)


			self.exp = True
			self.esn_cv.exp_weights = True
			exp_w_ = {'exp_weights' : True}



			input_dict = { **cv_args, 
					   **predetermined_args,
					   **exp_w_}
			print("new input dict: ")
			print(input_dict)
			
			
			#self.esn_cv = self.esn_cv_spec(**input_dict)
			self.best_arguments =  self.esn_cv.optimize(x = self.Train, y = self.xTr) 

		self.esn = self.esn_spec(**self.best_arguments,
								 obs_idx  = predetermined_args['obs_index'],
								 resp_idx = predetermined_args['target_index'])

		self.esn.train(x = self.Train, y = self.xTr)

		def my_predict(test, n_steps = None):
			if not n_steps:
				n_steps = test.shape[0]
			return self.esn.predict(n_steps, x = test[:n_steps,:])

		self.prediction = my_predict(self.Test)
		"""
		esn_obs = EchoStateNetwork(**exp_best_args, exponential = True, 
						   resp_idx = dat["resp_idx"], obs_idx = dat['obs_idx'], plot = True)
		#esn_obs.llambda = 0.01
		esn_obs.train(x = Train, y = xTr)
		"""

		self.save_json()
		print("\n \n exp rc cv data saved @ : " + self.outfile +".txt")


	def already_trained(self, best_args, exponential):

		self.best_arguments = best_args

		self.esn = self.esn_spec(**self.best_arguments,
								 obs_idx  = self.obs_idx,
								 resp_idx = self.resp_idx,
								 exponential = exponential)

		self.esn.train(x = self.Train, y = self.xTr)

		def my_predict(test, n_steps = None):
			if not n_steps:
				n_steps = test.shape[0]
			return self.esn.predict(n_steps, x = test[ :n_steps, :])

		self.prediction = my_predict(self.Test)

	"""
	def Unif_RC_CV(self, cv_args):
		'''
		for example bounds see the above function. 
		#TODO consider combining these functions.
		'''

		predetermined_args = {
			'exp_weights' : False,
			'obs_index' : self.resp_obs_idx_dict['obs_idx'],
			'target_index' : self.resp_obs_idx_dict["resp_idx"]
		}

		input_dict = { **cv_args, **predetermined_args}

		# subclass assignment: EchoStateNetworkCV
		self.unif_esn_cv = self.esn_cv_spec(**input_dict)
		print("uniform rc cv set, ready to train")

		self.unif_best_arguments = self.unif_esn_cv.optimize(x = self.Train, y = self.xTr) 

		self.save_json(exp = Falsepi)
		print("uniform rc cv data saved")
	"""

	def save_json(self):
		
		self.getData2Save()
		if not self.librosa:
			new_file = self.outfile

			"""
			if self.exp == True:
				model_ = "_exp"
			else:
				model_ = "_unif"
			"""
			new_file += ".txt"

			with open(new_file, "w") as outfile:
				data = json.dump(self.json2be, outfile)
		else:
			#spectrogram
			librosa_outfile = "./pickle_files/results/" + self.spectrogram_path +"/" 
			# spectrogram type
			librosa_outfile += self.spectrogram_type + "/"

			# flat or NOT
			#TODO
			flat_str = "untouched" if not self.flat else "flat"
			librosa_outfile += str(flat_str)  +"/"

			#split
			librosa_outfile += "split_"  + str(self.split) + "/"

			librosa_outfile += "tf_" + str(self.target_frequency)
			librosa_outfile += "__obsHz_"  + str(self.obs_kHz)
			librosa_outfile += "__targHz_" + str(self.target_kHz) + ".pickle"
			print("outfile: " + str(librosa_outfile))
			self.save_pickle(path = librosa_outfile, transform = self.json2be)


	def rbf_add_point(self, point_tuple, test_set = False):

		x, y = point_tuple
		if test_set:
			self.xs_unknown  += [x]
			self.ys_unknown  += [y]
		else:
			self.xs_known += [x]
			self.ys_known += [y]
			self.values   += [self.A[x,y]]

	def runInterpolation(self, columnwise = False, show_prediction = False):
		#2D interpolation
		#observer coordinates
		
		"""
		for i, column_idx in enumerate(dat["resp_idx"]):
				print(column_idx)
				values += list(A[:,column_idx].reshape(-1,))
				point_lst += list(zip(range(A.shape[0]), [column_idx]*A.shape[0]))
		print(len(point_lst))
		print(len(values))
		"""
		
		if self.prediction_type == "block":
		
			#Training points

			resp_idx = self.resp_idx
			obs_idx  = self.obs_idx

			total_zone_idx = resp_idx + obs_idx

		#missing_ = 60
		assert self.interpolation_method in ["griddata-linear", "rbf", "griddata-cubic"]

		if self.interpolation_method	 in ["griddata-linear", "griddata-cubic"]:
			#print(self.interpolation_method)
			points_to_predict = []
			
			values = []
			#visible
			point_lst = []
			
			if self.prediction_type == "block":
			
				#Train zone
				for x in range(self.xTr.shape[0]):
					# resonse points : train
					for y in total_zone_idx:
						point_lst += [(x,y)]#list(zip(range(Train.shape[0]) , [missing_]*Train.shape[0]))
						values	  += [self.A[x,y]]
						
				#Test zone
				for x in range(self.xTr.shape[0], self.A.shape[0]):
					# test set
					for y in resp_idx:
						points_to_predict += [(x,y)]
						
					# test set observers
					for y in obs_idx:
						point_lst += [(x,y)]
						values	+= [self.A[x,y]]

			elif self.prediction_type == "column":

				total_zone_idx = range(self.A.shape[1])

				#Train zone
				for x in range(self.xTr.shape[0]):
					# resonse points : train
					for y in total_zone_idx:
						point_lst += [(x,y)]#list(zip(range(Train.shape[0]) , [missing_]*Train.shape[0]))
						values	  += [self.A[x,y]]
						
				#Test zone
				for x in range(self.xTr.shape[0], self.xTr.shape[0] + self.xTe.shape[0]):
					# test set
					for y in total_zone_idx:
						points_to_predict += [(x,y)]
						

			griddata_type = "linear" if self.interpolation_method == "griddata-linear" else "cubic"

			ip2_pred = griddata(point_lst, values, points_to_predict, method = "linear")#, rescale = True)#, method="linear")#"nearest")#"linear")#'cubic')
			ip2_pred = ip2_pred.reshape(self.xTe.shape)
			#ip2_resid = ip2_pred - self.xTe
			#points we can see in the training set

			if show_prediction:
				plt.imshow(ip2_pred, aspect = 0.1)
				plt.show()
				
				plt.imshow(self.xTe, aspect = 0.1)
				plt.show() 

			###plots:
			self.ip_res = {"prediction": ip2_pred, 
						   "nrmse" : nrmse(pred_ = ip2_pred, truth = self.xTe, columnwise = columnwise)} 


		elif self.interpolation_method == "rbf":
			print("STARTING INTERPOLATION")

			self.xs_known, self.ys_known, self.values, self.xs_unknown, self.ys_unknown  = [], [], [], [], []

			for x in range(self.xTr.shape[0]):
				# resonse points : train
				for y in total_zone_idx:
					self.rbf_add_point((x, y))

			#Test zone
			for x in range(self.xTr.shape[0], self.A.shape[0]):
				for y in resp_idx:		# test set
					self.rbf_add_point((x,y), test_set = True)
					
				for y in obs_idx:		# test set observers
					self.rbf_add_point((x,y))

			x, y, values = [np.array(i) for i in [self.xs_known, self.ys_known, self.values]]
			

			ALL_POINTS_AT_ONCE = False

			self.rbfi = Rbf(x, y, values, function='cubic') 

			print("RBF SET")

			if ALL_POINTS_AT_ONCE:

				xi, yi = [np.array(i) for i in [self.xs_unknown, self.ys_unknown]]

				#epsilon=2, #function = "cubic")  # radial basis function interpolato
				di   = rbfi(xi, yi) 				# interpolated values
				di   = di.reshape(self.xTe.shape)
				diR  = nrmse(pred_ = di, truth = self.xTe, columnwise = columnwise)

				self.ip_res = {"prediction" : di, 
									"nrmse" : diR} 
				print("FINISHED INTERPOLATION: R = " + str(diR))

			else:
				values_rbf_output = []
				LEN = len(self.xs_unknown)
				print(LEN)
				for i, xi in enumerate(self.xs_unknown):
					print( i / LEN * 100)
					yi = np.array(self.ys_unknown[i])
					xi = np.array(xi)
					di = self.rbfi(xi, yi) 

					values_rbf_output +=[di]#[0]
				values_rbf_output = np.array(values_rbf_output).reshape(self.xTe.shape)
				diR  = nrmse(pred_ = di, truth = self.xTe, columnwise = columnwise)

				self.ip_res = {"prediction" : di, 
									"nrmse" : diR} 
				print("FINISHED INTERPOLATION: R = " + str(diR))
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

	def currTime():
		now = datetime.now()

		current_time = now.strftime("%H:%M:%S")
		print("Current Time =", current_time)
	currTime()
	"""