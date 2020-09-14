from tqdm.notebook import trange, tqdm

def check_for_duplicates(lst, UnqLst = True, verbose = True):
    """ return the duplicate items in a list
    
    Args:
        UnqLst: if True return the duplicates
        verbose: if True print the duplicates
    
    """
    lst_tmp = []
    duplicates = []
    for i in lst:
        if i in lst_tmp:   
            duplicates += [i]
        else:
            lst_tmp += [i]
    if verbose == True:
        print(duplicates)
    if UnqLst:
        return(lst_tmp)

def build_string(message, *values, sep = ""): 
    """
    example_usage: build_string("bp_", 5, 6, 7, "blah", sep = "/")
    """
    if not values:
        return message
    else:
        return message.join(str(x) + sep for x in values)

class EchoStateAnalysis:
    """ #Spectrogram class for training, testing, and splitting data for submission to reservoir nueral network code.
    
    Args: #TODO Incomplete
        bounds:
        size: a string in ["small", "medium", "publish"] that refer to different dataset sizes.
        file_path: a string that describes the directory where the data is located. (load from)
        out_path: where to save the data
        target_frequency: in Hz which frequency we want to target as the center of a block experiment or the only frequency in the case of a simple prediction.

    """
    def __init__(self, path_list, bp = "experiment_results/medium/", force = False):
        self.path_list = path_list
        self.bp = bp
        self.force = force
        
    def get_experiment(self, json_obj, compare_ = False, plot_split = False,
                   librosa = False, verbose = False, model = "exponential"):
        """ This function retrieves, from a json dictionary file, an EchoStateExperiment object.

        Args:
            compare_: if True run the compare function above (plot the NRMSE comparison)

            plot_split: if True plot the 2D train-test split

            librosa: If True, load a pickle file (instead of json), perhaps other things specific to 
                the spectrograms that were created with the librosa package.

        """

        experiment_ = EchoStateExperiment(**json_obj["experiment_inputs"], librosa = librosa)

        obs_inputs = json_obj["get_observer_inputs"]
        obs_inputs["method"] = "exact"

        experiment_.obs_idx, experiment_.resp_idx  = json_obj["obs_idx"], json_obj["resp_idx"]

        experiment_.get_observers(**obs_inputs, plot_split = plot_split)
        
        if verbose == True:
            print("experiment inputs: " + str(json_obj["experiment_inputs"]))
            print("get_obs_inputs: " + str(obs_inputs))
            print("Train.shape: " + str(experiment_.Train.shape))
            print("Saved_prediction.shape: " + str(np.array(json_obj["prediction"]["uniform"]).shape))

        if model == "uniform":
            experiment_.already_trained(json_obj["best arguments"]["uniform"], exponential = False)
        elif model == "exponential": 
            experiment_.already_trained(json_obj["best arguments"]["exponential"], exponential = True)

        experiment_.Train, experiment_.Test = experiment_.xTr, experiment_.xTe#self.recover_test_set(json_obj)

        ### what is the range of the test set?
        xx = range(experiment_.xTe.shape[0])

        #experiment_.plot_timeseries(method = "avg")
        if compare_:

            compare_inputs = {}

            #this is a hacky solution.
            changing_terminology = {"uniform" : "unif_w_pred", "exponential" : "exp_w_pred"}

            for model in list(json_obj["prediction"].keys()):

                specific_prediction = json_obj[model]
                specific_key = changing_terminology[model]
                compare_inputs[specific_key] = json_obj["prediction"][model]
            #print(compare_inputs)

            unif_w_pred, exp_w_pred = json_obj["prediction"]["uniform"], json_obj["prediction"]["exponential"]
            ip_pred = json_obj["prediction"]["interpolation"]
            unif_w_pred, exp_w_pred, ip_pred = [np.array(i) for i in [unif_w_pred, exp_w_pred, ip_pred]]
            """
            if verbose == True:
                for i in [[unif_w_pred, "unif pred"],
                          [exp_w_pred, "exp pred"],
                          [ip_pred, "ip pred"],
                          [np.array(experiment_.Test), "Test" ]]:
                    Shape(i)
            """
            ### make this flexible. if uniform isn't there, don't break. IF exp ditto.

            compare( truth = np.array(experiment_.Test), unif_w_pred = unif_w_pred, ip_pred = ip_pred,
                exp_w_pred  = exp_w_pred, columnwise  = False, verbose = False)
            if n_keys == 2:
                compare(
                    truth       = np.array(experiment_.Test), 
                    unif_w_pred = np.array(json_obj["prediction"]["uniform"]),
                    ip_pred = np.array(json_obj["prediction"]["interpolation"]),
                    exp_w_pred  = None,#np.array(json_obj["prediction"]["exponential"]), 
                    columnwise  = False,
                    verbose = False)
        return(experiment_)
    
    def change_interpolation_json(self,  ip_method = "all"):
        """ Imports a set of experiments and returns an altered list of jsons with a new interpolation method.
            Also plots the comparison.

        Args: 
            path_lst: a path list to the json experiments.
            ip_method: the interpolation method, a string chosen from the following set
                options: { "linear", "cubic", "nearest"}

        """
        path_lst = self.path_list
        results_tmp = []
        results_rel = []
        for i in trange(len(path_lst), desc='experiment list, fixing interpolation...'): 
            if not i:
                experiment_lst = []
            experiment_dict = self.load_data(path_lst[i], bp = self.bp)

            if ip_method == "all":
                for ip_method_ in ["linear", "cubic", "nearest"]:
                    tmp_dict = self.fix_interpolation(experiment_dict, method = ip_method_)
                    experiment_dict["nrmse"]["ip: " + ip_method_] = tmp_dict["nrmse"]["interpolation"]
                    experiment_dict["prediction"]["ip: " + ip_method_] = tmp_dict["prediction"]["interpolation"]

                rel_spec = { key: experiment_dict["nrmse"][key] / experiment_dict["nrmse"]["ip: linear"] 
                            for key in experiment_dict["nrmse"].keys()}
                
                results_rel.append(rel_spec)
                
                #removing interpolation
                pred_dict_temp = experiment_dict["prediction"]
                pred_dict_temp = ifdel(pred_dict_temp, "interpolation")
                
                nrmse_dict_temp = experiment_dict["nrmse"]
                nrmse_dict_temp = ifdel(nrmse_dict_temp, "interpolation")
                
                experiment_dict["nrmse"] = nrmse_dict_temp
                experiment_dict["prediction"] = pred_dict_temp
                
                results_tmp.append(experiment_dict["nrmse"])
                
                experiment_lst.append(experiment_dict)
            else:
                rel_spec = { key: experiment_dict["nrmse"][key] / experiment_dict["nrmse"]["interpolation"] 
                            for key in experiment_dict["nrmse"].keys()}
                results_rel.append(rel_spec)
                experiment_dict = fix_interpolation(experiment_dict, method = ip_method) 
                experiment_lst.append(experiment_dict)
                results_tmp.append(experiment_dict["nrmse"])

        results_df = pd.DataFrame(results_tmp)
        results_df = results_df.melt()
        results_df.columns = ["model", "R"]

        results_df_rel = pd.DataFrame(results_rel)
        results_df_rel = results_df_rel.melt()
        results_df_rel.columns = ["model", "R"]
        
        self.experiment_lst = experiment_lst
        self.R_results_df = results_df
        self.R_results_df_rel = results_df_rel
        #return(experiment_lst, results_df, results_df_rel)
    
    def load_data(self, file = "default", print_lst = None, bp = None, verbose = False, enforce_exp = False):
        """
        print_lst can contain a list of keys to print, for example ["nrmse"]
        """
        if bp != None:
            file = bp + file
        if file == "default":
            nf = get_new_filename(exp = exp, current = True)
        else:
            nf = file
        with open(nf) as json_file: # 'non_exp_w.txt'
            datt = json.load(json_file)
        print(datt["nrmse"])
            
        assert len(list(datt["nrmse"].keys())) >= 3, "at least one model not found: " + file
        
        if verbose:
            print(datt["nrmse"])
        
        return(datt)
    
    def compare(self, truth, unif_w_pred = None, exp_w_pred = None, ip_pred = None, 
                columnwise = False, verbose = False):
        """ This function compares the NRMSE of the various models: RC [exp, unif] and interpolation

        Args:

            columnwise: This function provides two things, conditional on the columnwise variable.
                False: cross-model comparison of nrmse
                True: model nrmse correlary for each point.

            ip_pred: The interpolation prediction over the test set
            exp_w_pred: Exponential weight RC prediction
            unif_w_pred: Uniform weight RC prediction
        """
        nrmse_inputs = {"truth" : truth, "columnwise" : columnwise}
        pred_list = [unif_w_pred, exp_w_pred, ip_pred]
        pred_dicts = [{"pred_": prediction, **nrmse_inputs} for prediction in pred_list]

        def conditional_nrmse(inputs):
            """ If the prediction exists, calculate the NRMSE

            Args: 
                inputs: a dictionary {truth, pred_}
            """
            return None if not inputs["pred_"] else nrmse(**inputs) # check if prediction is empty


        unif_nrmse, exp_nrmse, ip_nrmse = [nrmse(**i) for i in pred_dicts]


        hi = """
        if type(unif_w_pred) != type(None):
            unif_nrmse = nrmse(pred_ = unif_w_pred, **nrmse_inputs)

        if type(exp_w_pred) != type(None):
            exp_nrmse  = nrmse(pred_  = exp_w_pred , **nrmse_inputs)

        if type(ip_pred) != type(None):
            ip_nrmse   = nrmse(pred_  = ip_pred , **nrmse_inputs)
        """
        ip_res = {"nrmse" : ip_nrmse, "pred" : ip_pred}


        assert type(columnwise) == bool, "columnwise must be a boolean"

        if not columnwise:
            if verbose:
                print("cubic spline interpolation nrmse: " + str(ip_res["nrmse"]))
                print("uniform weights rc nrmse: " + str(unif_nrmse))
                print("exponential weights rc nrmse: " + str(exp_nrmse))
                print("creating barplot")

            unif_and_ip_dict = {"interpolation" : ip_res["nrmse"], "uniform rc" : unif_nrmse}
            exp_dict = {"exponential rc" : exp_nrmse}

            if type(exp_w_pred) != type(None):
                df = pd.DataFrame({"interpolation" : ip_res["nrmse"], 
                                   "uniform rc" : unif_nrmse, 
                                   "exponential rc" : exp_nrmse}, index = [0])
            else:
                df = pd.DataFrame({"interpolation" : ip_res["nrmse"], 
                                   "uniform rc" : unif_nrmse}, index = [0])
            display(df)

            plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            sns.catplot(data = df, kind = "bar")
            plt.title("model vs nrmse")
            plt.ylabel("nrmse")
            improvement = []
            for rc_nrmse in[unif_nrmse, exp_nrmse]:
                impr_spec = ((ip_res["nrmse"] - rc_nrmse)/ip_res["nrmse"]) * 100
                impr_spec = [round(impr_spec,1)]
                improvement += impr_spec

            pct_improve_unif, pct_improve_exp = improvement
            if pct_improve_unif > 0:
                print("unif improvement vs interpolation: nrmse " + str(-pct_improve_unif) + "%")
            else:
                print("rc didn't beat interpolation: nrmse +" + str(-pct_improve_unif) + "%")

            if pct_improve_exp > 0:
                print("exp improvement vs interpolation: nrmse " + str(-pct_improve_exp) + "%")
            else:
                print("rc didn't beat interpolation: nrmse +" + str(-pct_improve_exp) + "%")

            impr_rc_compare = round(((unif_nrmse - exp_nrmse)/unif_nrmse) * 100,1)

            if impr_rc_compare > 0:
                print("exp rc improvement vs unif rc: nrmse " + str(-impr_rc_compare) + "%")
            else:
                print("exp weights didn't improve rc: nrmse +" + str(-impr_rc_compare) + "%")
        else:
            print("creating first figure")
            model_names = ["interpolation", "uniform rc", "exponential rc"]
            for i, model_rmse_np in enumerate([ip_res["nrmse"], unif_nrmse, exp_nrmse]):
                model_rmse_pd = pd.melt(pd.DataFrame(model_rmse_np.T))
                model_rmse_pd.columns = ["t","y"]
                model_rmse_pd["model"] = model_names[i]
                if not i: # check if i == 0
                    models_pd = model_rmse_pd
                else:
                    models_pd = pd.concat([models_pd, model_rmse_pd ], axis = 0)
            fig, ax = plt.subplots(1,1, figsize = (11, 6))
            sns.lineplot(x = "t", y = "y", hue = "model", data = models_pd, ax = ax)
            ax.set_title("model vs rmse")
            ax.set_ylabel("nrmse")
            ax.set_xlabel("Test idx")

    def recover_test_set(self, json_obj):
        """ Recover the test_set. This function is probably vestigal.

        """

        experiment_ = EchoStateExperiment(**json_obj["experiment_inputs"])

        obs_inputs = json_obj["get_observer_inputs"]
        obs_inputs["method"] = "exact"

        obs_idx, resp_idx = json_obj["obs_idx"], json_obj["resp_idx"]
        A_subset = experiment_.A.copy()

        # pred shape
        pred_shape = np.array(json_obj["prediction"]["interpolation"]); pred_shape = pred_shape.shape[0]                   

        A = experiment_.A

        train_len = (A.shape[0] - pred_shape)
        Train_Tmp, Test_Tmp  = A[:train_len, resp_idx], A[train_len:, resp_idx]

        return(Train_Tmp, Test_Tmp)

    def load_p_result (path : str, bp = ""):
        """ Load a pickle spectrogram result.

        Args:
            path: the path to the file
            bp: base_path
        """
        path = bp + path
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        return(b)


    def fix_interpolation(self, exper_, method):
        """ Change the interpolation method of the inputed experiment.

        Args:
            exper_: the json experiment dictionary which you will to alter.
            method: the type of interpolation method (chosen from 'linear', 'cubic', 'nearest')
        """
        hiObj = self.get_experiment(exper_, verbose = False, plot_split = False, compare_ = False)
        if method == "cubic":
            hiObj.interpolation_method = "griddata-cubic"
        elif method == "nearest":
            hiObj.interpolation_method = "griddata-nearest"
        hiObj.runInterpolation()
        exper_["prediction"]["interpolation"] = hiObj.ip_res["prediction"]
        exper_["nrmse"]["interpolation"] = hiObj.ip_res["nrmse"]
        return(exper_)
    
    def make_R_barplots(self, label_rotation = 45):
        """
        Self.Parameters:
            self.R_results_df is the nrmse pd dataframe, non-columnwise for all experiments in path_lst
            self.R_resulsts_df_rel is the relative R pd dataframe
        """
        fig, ax = plt.subplots(1, 2, figsize = (14, 5))
        ax = ax.ravel()

        sns.violinplot(x = "model", y = "R", data = self.R_results_df, ax=ax[0])
        sns.violinplot(x = "model", y = "R", data = self.R_results_df_rel, ax=ax[1])
        for i in range(len(ax)):
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=label_rotation)
        ax[0].set_ylim(0, 1.05)
        ax[1].set_ylim(0.5, 2.0)

#recover_test_set(hi)