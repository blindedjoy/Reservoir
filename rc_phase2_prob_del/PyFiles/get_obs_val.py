# validation version
def get_observers(method, 
                  num_observers = 20,
                  missing = 100, 
                  split = 0.2, 
                  observer_range = None,
                  response_range = None,
                  plot_split = True,
                  validation = False,
                  dataset = A, 
                  aspect = 6, transpose = False):
    """
    arguments:
        method: either random or equal
        missing: either 
            (+) any integer:  (standing for column of the spectogram) or 
            (+) "all" : which stands for all of the remaining target series.
    """
    #if transpose == True:
    #    dataset = dataset.T
    n_rows = dataset.shape[0]
    n_cols = dataset.shape[1]
    
    train_len = int(n_rows * split)
    
    if validation != False:
        val_split = int(n_rows * validation)
    else:
        val_split = dataset.shape[0]
    
    test_len =  n_rows - train_len
    col_idx = list(range(n_cols))
    
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
    
    
    ### The following is problematic because you haven't dealt with the case where they overlap.
    ### BLOCK
    elif method == "block":
        """
        This method either blocks observers and/or the response area.
        """
        print("you selected the block method")
        if response_range == None:
            response_idx  = [missing]
            response      = dataset[ : , missing].reshape( -1, 1)
        else:
            response_idx =  my_range2lst(response_range)
            response = dataset[ : , response_idx].reshape( -1, len( response_idx))
        #print("response_index: " + str(response_idx))
            
        for resp_idx_spec in response_idx:
            col_idx.remove( resp_idx_spec)
        
        if observer_range == None:
            col_idx.remove( missing)
            obs_idx = np.sort( np.random.choice( col_idx, 
                                                num_observers, 
                                                replace = False))
        else:
            obs_idx = my_range2lst(observer_range)
        

            
        # check for problems with the block method:
        union_obs_resp_set = set(obs_idx) & set(response_idx)
        union_obs_resp_lst = list(union_obs_resp_set)
        
        if len(union_obs_resp_lst) != 0:
        
            print("Error: overlap in obs_idx and response_idx: ")
            print("overlap: " + str(union_obs_resp_lst))

            return(1)
            
    
    observers = dataset[ : val_split, obs_idx]

    observers_tr = observers[ :train_len, : ]
    observers_te = observers[ train_len : val_split, : ]

    response_tr  = response[   :train_len, : ]
    response_te  = response[   train_len:val_split, : ]

    
    ### Visualize the train test split, observers and (validation set?)
    if plot_split == True:
        red, yellow, blue, black = [255, 0, 0], [255, 255, 0], [0, 255, 255], [0, 0, 0]
        orange, green, white = [255, 165, 0], [ 0, 128, 0], [255, 255, 255]
        if validation == False:
            
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
        
        else:
            
            #preprocess:
            if transpose == True:
                split_img = np.full(( n_cols, n_rows, 3), black)
            else:
                split_img = np.full(( n_rows, n_cols, 3), black)

            # assign observer lines
            for i in obs_idx:
                split_img[ : , i] = np.full(( 1, n_rows, 3), yellow)
            
            
            # assign target area
            for i in response_idx:
                split_img[ :train_len, i] = np.full(( 1, train_len, 3), blue)
                split_img[ train_len:val_split, i] = np.full(( 1, val_len,  3), white)
                split_img[ val_split:, i] = np.full(( 1, test_len,  3), red)
                
            legend_elements = [Patch(facecolor='cyan', edgecolor='blue', label='Train'),
                               Patch(facecolor='white', edgecolor='red', label='Validation'),
                           Patch(facecolor='red', edgecolor='red', label='Test'),
                           Patch(facecolor='yellow', edgecolor='orange', label='Observers')]
        
            
        # Create the figure
        fig, ax = plt.subplots( 1, 2, figsize = ( 12, 6))
        ax = ax.flatten()
        
        
        solid_color_np =  np.transpose(split_img.T, axes = (1,2,0))
        
        #solid_color_pd.index = freq_idx
        
       
        """
        oA = np.rot90(A_orig, k = 0, axes = (0, 1))
        #oA stands for other lab A
        oA = pd.DataFrame(oA).copy()
        freq_idx = [ int(i / 100) * 100 for i in f['f'].reshape(-1,).tolist()]
        oA.index = freq_idx

        yticks = list(range(0,10800,1000))

        y_ticks = [ int(i) for i in yticks]

        my_heat = sns.heatmap(oA, center=0, cmap=sns.color_palette("CMRmap"), 
                              yticklabels = 100, ax = axis)
        #, cmap = sns.color_palette("RdBu_r", 7))
        axis.set_ylabel('Frequency (Hz)')#,rotation=0)
        axis.set_xlabel('time')

        my_heat.invert_yaxis()
        plt.yticks(rotation=0)
        """
        
        # The legend:
        #https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
        
        
        
        olab_display(ax[1])
        
        #### INTERVENTION for correct labels on first plot
       
        ax[0].imshow(solid_color_np, aspect = aspect)
        ax[0].set_ylabel('Frequency (Hz)')
        ax[0].set_xlabel('time')
        ax[0].legend(handles=legend_elements, loc='lowerright')
        #ax[0].invert_yaxis()
        old_labels  = ax[0].get_yticks()
        
        plt.sca(ax[1])
        locs, labels = plt.yticks()  
        freq_labels = [int(label.get_text()) for label in labels]
        max_freq = np.max(freq_labels)
        freq_labels = range(0, np.max(freq_labels), 1000)
        
        max_idx = A.shape[0]
        new_positions = (freq_labels/max_freq)*max_idx
        new_positions = [int(max_idx - i) for i in new_positions]
        #print("new_positions " +  str(new_positions))
        #print("len new_positions " +  str(len(new_positions)))
        
        plt.sca(ax[0])
        
        plt.ylabel('Frequency (Hz)')#,rotation=0)
        plt.xlabel('time')
        
        max_idx = int(round(np.max(old_labels)))
        #print("max_idx " + str(max_idx))
        max_freq = int(round(np.max(freq_labels)))
        nticks = len(freq_labels)
        #pos_nlabels = list(range(0, old_labels, max_old//(nticks+1) ))
        #val_nlabels = list(range(0, max_new, max_new//nticks + 1))
        
        plt.yticks(ticks = new_positions, labels = freq_labels)
        plt.sca(ax[1])
        new_pos_rev = [max_idx - i for i in new_positions]
        new_pos_rev = new_pos_rev - np.min(new_pos_rev)
        plt.yticks(ticks = new_pos_rev, labels = freq_labels)
        xx = spect_xrange[size]
        plt.xticks(ticks = xx, labels = xx, rotation = 0)
        #plt.ylim((0, max_old))
        ax[0].set_title("Dataset Split Visualization")
        ax[1].set_title("Spectogram Data")
        plt.show()
        #plt.imshow(solid_color_np, aspect = aspect)
        #plt.show()
    
    if validation == False:
        return({"obs_tr"  : observers_tr, 
                "obs_te"  : observers_te,
                "resp_tr" : response_tr,
                "resp_te" : response_te,
                "obs_idx" : obs_idx,
                "resp_idx" : response_idx})
    else:
        return({"obs_tr"  : observers_tr, 
                "obs_val" : observers_val,
                "obs_te"  : observers_te,
                
                "resp_tr" : response_tr,
                "resp_val" : observers_val,
                "resp_te" : response_te,
                
                "obs_idx" : obs_idx,
                "resp_idx" : response_idx})