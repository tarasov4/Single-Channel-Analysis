import numpy as np
import matplotlib.pyplot as plt
import pyabf
import os
from pathlib import Path
import pybaselines
from collections import namedtuple
import pandas as pd
from distfit import distfit
from scipy.stats import *
import math
import seaborn as sns

class ProcessABF:
    
    """
    Compensate capasitive transient with Joint Baseline Correction and Denoising (jbcd) Algorithm
    using pybaselines library.
    
    Reference: Liu, H., et al. Joint Baseline-Correction and Denoising for Raman Spectra. 
               Applied Spectroscopy, 2015, 69(9), 1013-1022.
               
    Rewrites ABF file with no capacitive transient.
        
    """
    
    def __init__(self, file:str, start_time:float, end_time:float, decay_time:float):
        
        """
        file (str) - path to row .abf file.
        start_time (float) - start of the region of interests in seconds.
        end_time (float) - end of the region of interests in seconds.
        fit_time (float) - duration of capasitive transient in seconds.
        
        """
        
        self._file = file
        self._start_time = start_time
        self._end_time = end_time
        self._decay_time = decay_time
        self._jbcd_current = None
        
    @property
    def file(self):
        return self._file
    
    @property
    def ABF_object(self):
        return pyabf.ABF(self.file)
    
    @property
    def start_time(self):
        return self._start_time
    
    @property
    def end_time(self):
        return self._end_time
    
    @property
    def decay_time(self):
        return self._decay_time
    
    @property
    def frequency(self):
        time_unit = self.ABF_object.sweepX[1]-self.ABF_object.sweepX[0]
        return time_unit**-1
    
    @property
    def new_name(self):
        return self.file.split('.')[0]+'_new.abf'
    
    @property
    def jbcd_current(self):
        return self._jbcd_current

    @jbcd_current.setter
    def jbcd_current(self, array):
        self._jbcd_current = array
        
    def _jbcd(self, sweep_n:int, baseline:list, start_ix:int, decay_ix:int, end_ix:int):
        ABF = self.ABF_object
        def signal_foo(sweep_n):
            ABF.setSweep(sweep_n,baseline=baseline)
            I_peak = ABF.sweepY[start_ix:decay_ix]
            I_late = ABF.sweepY[decay_ix:end_ix+1]

            return I_peak, I_late

        Signal = namedtuple('Signal',['I_peak', 'I_late'])

        signal = Signal._make(signal_foo(sweep_n))

        jbcd_peak = pybaselines.morphological.jbcd(
                signal.I_peak, beta_mult=1.05, gamma_mult=0.95)[1]['signal']
        #jbcd_late = pybaselines.morphological.jbcd(
                #signal.I_late, beta_mult=1.05, gamma_mult=0.95)[1]['signal']

        jbcd_peak = jbcd_peak - np.median(jbcd_peak[int(jbcd_peak.size - jbcd_peak.size/10):])
        #jbcd_late = jbcd_late - np.median(jbcd_late)
        #jbcd = np.concatenate((jbcd_peak, jbcd_late))
        jbcd = np.concatenate((jbcd_peak, signal.I_late))

        I = np.concatenate((signal.I_peak, signal.I_late))
        t = ABF.sweepX[start_ix:end_ix+1] - ABF.sweepX[start_ix]


        plt.ioff()
        fig = plt.figure(figsize=(10, 4))
        ax1,ax2 = plt.subplot(211), plt.subplot(212)
        ax1.set_title(f'sweep number: {sweep_n}')
        ax1.plot(t, I, color='black', label='row signal')
        ax1.plot(t, jbcd, color = 'blue', label='jbcd signal')
        ax1.plot(t, 0*t, '--', color='red')
        ax1.grid()
        ax1.legend()
        ax1.set_xlim(left = -.001, right=0.040)
        ax1.plot(t, I, color='black')
        ax2.plot(t, I, color='black')
        ax2.plot(t, jbcd, color = 'blue')
        ax2.plot(t, 0*t, '--', color='red')
        ax2.grid()
        plt.tight_layout()
        plt.savefig(Path('.').joinpath('Fig').joinpath(f'{self.file.split(".")[0]} sweep {sweep_n}'))
        plt.close(fig)

        return jbcd


    def correct_baseline(self):

        ABF = self.ABF_object

        decay_time = self.start_time + self.decay_time
        time_unit = ABF.sweepX[1]-ABF.sweepX[0]
        baseline=[self.start_time, self.end_time]

        start_ix = int(round(self.start_time/time_unit))
        end_ix = int(round(self.end_time/time_unit))
        decay_ix = int(round(decay_time/time_unit))
        sweep_list = ABF.sweepList

        I_list = []
        for sweep_n in sweep_list:
            I_list.append(self._jbcd(sweep_n, baseline, start_ix, decay_ix, end_ix))

        I_array = np.row_stack(I_list)
        self.jbcd_current = I_array
                
    def re_write_ABF(self, excluded:list=None):
        
        """excluded (list): list of sweeps to be excluded"""
        
        if self.jbcd_current is not None:
            sweepData = self.jbcd_current
        else:
            raise ValueError('self.correct_baseline() must be run first')
        
        if excluded:
            sweepData = np.delete(sweepData, np.array(excluded).astype('int'), axis=0)
        wr = pyabf.abfWriter
        wr.writeABF1(sweepData=sweepData, filename=self.new_name, sampleRateHz=self.frequency)

            
def concat_abf(abf_files:list,  t1:float, resulting_file_name='concat.abf'):
    
    """
    Concatenate .abf files with equal sampling rate.
    
    abf_files (list of stringes): list of abf file name to be concatenated
    t1 (float): duration of recording in seconds.
    
    """

    def current_stack(abf_file):
        ABF = pyabf.ABF(abf_file)
        t0 = 0

        time_unit = ABF.sweepX[1]-ABF.sweepX[0]
        ix0 = int(t0//time_unit)
        ix1 = int(t1//time_unit)

        currents_stack = []
        for sweep_number in ABF.sweepList:
            ABF.setSweep(sweep_number)
            currents_stack.append(ABF.sweepY[ix0:ix1+1])
        return np.row_stack(currents_stack), time_unit
    
    lst = []
    
    for file in abf_files:
        i = current_stack(file)[0]
        lst.append(i)
        print(i.shape)
    
    I = np.concatenate(lst, axis=0)
    
    time_unit = current_stack(abf_files[0])[1]
    sampleRateHz = 1/time_unit            
    wr = pyabf.abfWriter
    wr.writeABF1(sweepData=I, filename=os.getcwd()+'\\'+resulting_file_name, sampleRateHz=sampleRateHz)

    
class EventsDetection:

    """Writes single channels search (events sheet) into .csv file and idealize path clamp recording"""
    
    def __init__(self, abf_file:str, atf_file:str):
        
        """
        Params:
        abf_file: name .abf file
        atf_file: name of.atf containing events search results from Clamfit
            
        """
        self._abf_file = abf_file
        self._atf_file = atf_file
                        
    @property
    def abf_file(self):
        return self._abf_file
    
    @property
    def atf_file(self):
        return self._atf_file
    
    @property
    def events_csv(self):
        return self.abf_file.split('.')[0]+'_events.csv'
    
    @property
    def idealized_csv(self):
        return self.abf_file.split('.')[0]+'_idealized.csv'
    
    @property
    def n_sweeps(self):
        ABF = pyabf.ABF(self.abf_file)
        return len(ABF.sweepList)
           
    def get_events(self, save:bool=True):
        data = pd.DataFrame()
        with open(self._atf_file, mode='r') as events:
            line = 0
            for record in events:
                if line == 2:
                    cols = record.split('\t')[:11]
                    cols = [s.strip('"') for s in cols]

                if line > 2:
                    row = record.split('\t')[:11]
                    processed_row = []

                    for item in row:
                        try:
                            item = float(item)
                        except:
                            processed_row.append(item)
                        else:
                            processed_row.append(float(item))

                    data = pd.concat([data, pd.DataFrame(data=processed_row).T], axis=0, ignore_index=True)

                line += 1
        data = data.rename(columns={old_col:new_col for old_col, new_col in zip(data.columns, cols)})
        
        if save:
            data.to_csv(self.events_csv, header=data.columns, index=False)
                
    
    def idealize(self, save:bool=True, 
                 plot_idealization:bool=False, rep_sweeps:list=[]):
        
        """
        Idealizes events search.
        """
        
        dataset = pd.read_csv(self.events_csv)
        abf = pyabf.ABF(self.abf_file)
        time = np.arange(abf.sweepX[0]*1000, abf.sweepX[-1]*1000+0.01, 0.01)
          
        count_rows = 0
        count_states = 0
        states = np.zeros(time.size)
        STATES = np.zeros(time.size)
        sweep_list = np.unique(dataset.loc[:,"Trace"])

        for sweep in sweep_list:

            for ix,t in np.ndenumerate(time):
                
                if ix[0]!=0:
                    states[ix[0]] = states[ix[0] - 1]
                    
                if math.isclose(t, dataset.at[count_rows + count_states, "Event Start Time (ms)"], abs_tol=0.01):
                    states[ix[0]] = dataset.at[count_rows + count_states, "Level"]
                    
                if math.isclose(t, dataset.at[count_rows + count_states, "Event End Time (ms)"], abs_tol=0.01):
                    
                    if count_rows + count_states + 1 < dataset.shape[0]: 
                        if  dataset.at[count_rows + count_states+1, "Trace"] != dataset.at[count_rows + count_states, "Trace"]:
                            states[ix[0]] = 0
                            STATES = np.vstack((STATES, states))
                            states = np.zeros(time.size)
                            count_rows = count_rows + count_states + 1
                            count_states = 0
                            break

                        else: 
                            states[ix[0]] = dataset.at[count_rows + count_states+1, "Level"]
                            count_states +=1
                    else:
                        states[ix[0]] = 0
                        STATES = np.vstack((STATES, states))
                        break

        STATES = np.delete(STATES, 0, axis=0)
        empty_sweeps_states = np.zeros((self.n_sweeps-sweep_list.size, STATES.shape[1]))
        STATES = np.vstack([STATES, empty_sweeps_states])
        STATES = np.vstack([time, STATES])
                
        #STATES structure: row 0 - time, ecah next row - states in 1 sweep
        
        if save:
                                        #saving STATES in .csv file
        #=========================================================================================#

            DF = pd.DataFrame(data=STATES.T)
            DF.to_csv(self.idealized_csv, 
                      header=(['time, ms']+[f'sweep {i}' for i in range(STATES.T.shape[1]-1)]), index=False)
            
        if plot_idealization:
            
            rep_traces = rep_sweeps
            fig, axis = plt.subplots(ncols=3, nrows=3, figsize = (14, 6))
            for axis, rep_trace in zip(axis.ravel(), rep_traces):
                axis.plot(STATES[0], STATES[rep_trace])
                axis.plot(STATES[0], np.zeros(STATES[0].size), '--', color='black')
                axis.plot(STATES[0], np.full(STATES[0].size, 1), '--', color='green')
                axis.plot(STATES[0], np.full(STATES[0].size, 2), '--', color='red')
                axis.set_xlim(left=-2)
                axis.set_ylim(bottom=-0.2)
                axis.set_ylabel('level', fontsize = 14)
                axis.set_xlabel('ms', fontsize = 14)
            plt.tight_layout()
            plt.show()
            
class EventsAnalysis:
    
    """Analizes single channels recordings"""
    
    def __init__(self, abf_file, idealized_csv):
        
        """
        Params:
        
        abf_file - name of analized abf.file
        idealized_csv - name of idealized_csv

        """
        
        self.abf_file = abf_file
        self.idealized_csv = idealized_csv
        self.time, self.dataset = self._get_dataset()
        self.binom_fit_results_csv = self.abf_file.split('.')[0]+'_binom_fit.csv'
        
        
    def _get_dataset(self):
        dataset = pd.read_csv(self.idealized_csv, header=None).to_numpy()
        dataset = np.delete(dataset, 0, axis=0)
        dataset = dataset.astype('float')
        time = dataset[:,0].copy()
        dataset = np.delete(dataset, 0, axis=1)
        dataset = dataset.astype('int') # each row - one time point, each column - one sweep
        return time, dataset
        
    
    def _pool(self, exp_fr, obs_fr):

        new_exp_fr = []
        new_obs_fr = []
        removed_ix = []

        for ix, fr in enumerate(exp_fr):
            if ix not in removed_ix:
                if fr>=5:
                    new_exp_fr.append(fr)
                    new_obs_fr.append(obs_fr[ix])
                else:
                    if len(exp_fr)>ix+1:
                        new_exp_fr.append(fr+exp_fr[ix+1])
                        new_obs_fr.append(obs_fr[ix]+obs_fr[ix+1])
                        removed_ix.append(ix+1)

        if exp_fr[-1]<5:
            new_exp_fr[-1] += exp_fr[-1]
            new_obs_fr[-1] += obs_fr[-1]

        if len(new_exp_fr)>1:
                return new_exp_fr, new_obs_fr


    def _single_binom_fit(self, time_point):
        obs_categs, obs_fr = np.unique(time_point, return_counts=True)
        if obs_categs.size>1:
            result = distfit(method='discrete').fit_transform(time_point, verbose=0)
            N_fit, po_fit, RSS = result['model']['n'], result['model']['p'], result['model']['score']
            exp_fr = []
            for cat in obs_categs:
                exp_fr.append(time_point.size*binom.pmf(cat, N_fit, po_fit))


            while not np.all(np.array(exp_fr)>=5):
                if self._pool(exp_fr, obs_fr) is not None: 
                    exp_fr, obs_fr = self._pool(exp_fr, obs_fr)
                else:
                    exp_fr, obs_fr = None, None
                    break

            if exp_fr is not None and obs_fr is not None:
                try:
                    ch_sq, pval = chisquare(obs_fr, exp_fr)
                    S_obs = entropy(obs_fr)
                    S_exp = entropy(exp_fr)
                    delta_S = S_obs - S_exp
                except:
                    raise ValueError('sums of observed adn expected frequencies is not equale')
                    
                else:
                    return pval, N_fit, RSS, delta_S #chi sq pval

    def binom_fit(self):
        time = self.time
        dataset = self.dataset
        npo_array = np.array([])
        pval_array = np.array([])
        N_array = np.array([])
        RSS_array = []
        t = np.array([])
        delta_S_array = np.array([])
        for t_value, time_point in zip(time, dataset):
            if self._single_binom_fit(time_point) is not None:
                levels, counts = np.unique(time_point, return_counts=True)
                npo = counts[np.where(levels!=0)].sum()/time_point.size
                npo_array = np.append(npo_array, npo)
                pval, N_fit, RSS, delta_S = self._single_binom_fit(time_point)
                pval_array = np.append(pval_array, pval)
                N_array = np.append(N_array, N_fit)
                RSS_array = np.append(RSS_array, RSS)
                t = np.append(t, t_value)
                delta_S_array = np.append(delta_S_array, delta_S)

        npo_crit = 0.75*npo_array.max()
        good_RSS = RSS_array[np.where(npo_array>=npo_crit)]
        good_N = N_array[np.where(npo_array>=npo_crit)]
        RSS_median = np.median(good_RSS)
        best_N = good_N[np.where(good_RSS<RSS_median)]
        good_chisq_pval = pval_array[np.where(npo_array>=npo_crit)]
        good_t = t[np.where(npo_array>=npo_crit)]
        best_t = good_t[np.where(good_RSS<RSS_median)]
        best_chisq_pval = good_chisq_pval[np.where(good_RSS<RSS_median)]
        best_ns, counts = np.unique(best_N, return_counts=True)
        BEST_N = best_ns[np.where(counts==counts.max())]

        dct = {'time': pd.Series(t, index = np.arange(t.size)),
               'npo': pd.Series(npo_array, index = np.arange(npo_array.size)),
               'pval': pd.Series(pval_array, index=np.arange(pval_array.size)),
               'delta_S': pd.Series(delta_S_array, index=np.arange(delta_S_array.size)),
               'best_time': pd.Series(best_t, index=np.arange(best_t.size)), 
               'best_chisq_pval': pd.Series(best_chisq_pval, index=np.arange(best_chisq_pval.size)), 
               'Best fit N of channels': pd.Series(int(BEST_N[0]), index=[0])}
        DF = pd.DataFrame(data=dct, index=np.arange(t.size)).to_csv(self.binom_fit_results_csv) 
    
    
class CoupledGatingAnalysis:
        
    def __init__(self, abf_file, events_csv, idealized_csv):
            self.abf_file = abf_file
            self.events_csv = events_csv
            self.idealized_csv = idealized_csv
            
    
    def latency(self, save=True):
        
        """Calculates latency between successive openinga and closings"""
        
        search = pd.read_csv(self.events_csv)
        
        sweeps = search.loc[:,"Trace"].to_numpy()
        events = search.loc[:,"Level"].to_numpy()
        starts = search.loc[:,"Event Start Time (ms)"].to_numpy() 
        ends = search.loc[:,"Event End Time (ms)"].to_numpy() 

        lat_o = np.array([])
        lat_c = np.array([])

        for ix, event in np.ndenumerate(events):
            if event > 1:        
                if sweeps[ix[0]] != sweeps[ix[0]-1]:            
                    lat_o = np.append(lat_o, [0])            
                elif event - events[ix[0]-1] > 1:
                    lat_o = np.append(lat_o, [0])
                elif event - events[ix[0]-1] == 1:
                    if events[ix[0]-2]<events[ix[0]-1]:
                        lat_o = np.append(lat_o, [starts[ix[0]]-starts[ix[0]-1]])
       
        for ix, event in np.ndenumerate(events):
            if event > 1:        
                if sweeps[ix[0]+1] != sweeps[ix[0]]:            
                    lat_c = np.append(lat_c, [0])
                elif event - events[ix[0]+1] > 1:
                    lat_c = np.append(lat_c, [0])
                elif event - events[ix[0]+1] == 1:
                    if events[ix[0]+2]<events[ix[0]+1]:
                        lat_c = np.append(lat_c, [ends[ix[0]+1]-ends[ix[0]]])
                        
        if save:
            DF = pd.DataFrame([lat_o, lat_c]).T
            file_name = self.abf_file.split('.')[0]+'_latency.csv'
            DF.to_csv(file_name, header=['lat_o', 'lat_c'], index=False)
                                
        return lat_o, lat_c
  
    def latency_hists(self):
        
        """Plots latency"""
        
        lat_o, lat_c = self.latency()

        lat_o_heights, lat_o_bins = np.histogram(lat_o, bins=np.arange(0, 3.05, 0.05))
        lat_c_heights, lat_c_bins = np.histogram(lat_c, bins=np.arange(0, 3.05, 0.05))

        width_o =lat_o_bins[1]-lat_o_bins[0]
        width_c = lat_c_bins[1]-lat_c_bins[0]

        lat_o_norm = lat_o_heights/float(lat_o.size)
        lat_o_norm_cum = np.cumsum(lat_o_norm, axis=0)
        lat_c_norm = lat_c_heights/float(lat_c.size)
        lat_c_norm_cum = np.cumsum(lat_c_norm, axis=0)

        plt.figure(figsize=(12, 10))

        ax1 = plt.subplot(411)
        plt.bar(lat_o_bins[:-1], lat_o_norm, width=width_o)
        plt.plot([np.mean(lat_o)]*2, [0, lat_o_norm.max()], color='orange')
        plt.scatter([np.mean(lat_o)], lat_o_norm.max(), marker='v', s=200, label=f'mean={round(np.mean(lat_o),2)} ms')
        plt.plot([np.median(lat_o)]*2, [0, lat_o_norm.max()], color='red')
        plt.scatter([np.median(lat_o)], lat_o_norm.max(), marker='v', color='red', s=200, label=f'median={round(np.median(lat_o),2)} ms')

        plt.legend(fontsize=14)
        plt.title('Latency between successive openings', fontsize=14)
        plt.xlabel('Latency, ms', fontsize=14)
        plt.ylabel('Fraction of totat', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        ax2 = plt.subplot(412)
        plt.plot(lat_o_bins[:-1],lat_o_norm_cum)
        plt.title('Cumulative distribution of latency \nbetween successive openings', fontsize=14)
        plt.xlabel('Latency, ms', fontsize=14)
        plt.ylabel('Fraction of total', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        ax3 = plt.subplot(413)
        plt.bar(lat_c_bins[:-1], lat_c_norm, width=width_c)
        plt.plot([np.mean(lat_c)]*2, [0, lat_c_norm.max()], color='orange')
        plt.scatter([np.mean(lat_c)], lat_c_norm.max(), marker='v', s=200, label=f'mean={round(np.mean(lat_c),2)} ms')
        plt.plot([np.median(lat_c)]*2, [0, lat_c_norm.max()], color='red')
        plt.scatter([np.median(lat_c)], lat_c_norm.max(), marker='v', color='red', s=200, label=f'median={round(np.median(lat_c),2)} ms')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(fontsize=14)
        plt.title('Latency between successive closings', fontsize=14)
        plt.xlabel('Latency, ms', fontsize=14)
        plt.ylabel('Fraction of total', fontsize=14)

        ax4 = plt.subplot(414)
        plt.plot(lat_c_bins[:-1],lat_c_norm_cum)
        plt.title('Cumulative distribution of latency \nbetween successive closings', fontsize=14)
        plt.xlabel('Latency, ms', fontsize=14)
        plt.ylabel('Fraction of total', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.tight_layout()
        
     
    
    def f1f2(self, show_validation=False, chi_sq=False, plot_f1f2=False, 
             plot_chi_sq=False, save=True):
        
        STATES = pd.read_csv(self.idealized_csv, header=None).to_numpy().T
        STATES = np.delete(STATES, 0, axis=1)
        STATES = STATES.astype('float')
        
        time = STATES[0]
        level_count = np.zeros((4, time.size)) #row levels
        level_count[0] = time
        level_count[1:] = level_count[1:].astype('int')

        for n in range(STATES.shape[1]):
            states, L = np.unique(STATES[1:, n], return_counts=True)
            for i, j in enumerate(list(states)):
                level_count[int(j)+1,n] = L[i]

        n_sweeps = STATES.shape[0]-1
        fraction_count = level_count.copy().astype('float')
        fraction_count[1:] /= n_sweeps # row fractions

        f0 = fraction_count[1]
        f1 = fraction_count[2]
        f2 = fraction_count[3]
        indi_fraction_count = fraction_count.copy()
        indi_fraction_count[1] = np.power((f0 + f1/2), 2)
        indi_fraction_count[2] = 2*(f1/2+f2)*(f0+f1/2)
        indi_fraction_count[3] = np.power((f1/2 + f2), 2) #expected fractions
        indi_level_count = indi_fraction_count.copy()
        indi_level_count[1:] *= n_sweeps #expected levels
        #_count stracture: row0 - time, row1 - L0, row2 - L1, row3 - L2
        L0 = level_count[1]
        L1 = level_count[2]
        L2 = level_count[3]
        L0i = indi_level_count[1]
        L1i = indi_level_count[2]
        L2i = indi_level_count[3]
        f0i = indi_fraction_count[1]
        f1i = indi_fraction_count[2]
        f2i = indi_fraction_count[3]
        
        if show_validation:
            #validation of L values: the sum of all Ls should be = number of sweeps
            #====================================#
            print(f'n_sweeps = {n_sweeps}')
            print(f'Is L0+L1+L2 = n_sweeps: {np.all((L0+L1+L2) == n_sweeps)}')
            print(f'min of sum of all Li: {(L0i+L1i+L2i).min()}')
            print(f'max of sum of all Li: {(L0i+L1i+L2i).max()}')
            
        if chi_sq:
            
            ###calculation of p for chi2 test
            pval_chi2 = np.zeros(STATES.shape[1]).astype('float')
            for ix in range(pval_chi2.size):
                obst_NoInter = np.array([L0[ix], L1[ix]/2, L1[ix]/2, L2[ix]]).astype('float64')
                expt_NoInter = np.array([L0i[ix], L1i[ix]/2, L1i[ix]/2, L2i[ix]]).astype('float64')
                X2_NoInter = chisquare(obst_NoInter, expt_NoInter)[0]
                if np.isnan(X2_NoInter):
                    X2_NoInter = 0
                df=1
                pval = 1 - chi2.cdf(X2_NoInter, df)
                pval_chi2[ix] = pval
        
        if plot_f1f2:
            
            plt.figure(figsize=(14,6))

            ax1 = plt.subplot(321)
            plt.plot(time, f1, label='f1 observed')
            plt.plot(time, f1i, label='f1 expected')
            plt.xlabel('ms', fontsize=14)
            plt.ylabel('fraction of sweeps', fontsize=14)

            ax2 = plt.subplot(322)
            plt.plot(time, f1, label='f1 observed')
            plt.plot(time, f1i, label='f1 expected')
            ax2.legend(loc='upper center', fontsize=14)
            plt.xlim(0, 10)
            plt.xlabel('ms', fontsize=14)
            plt.ylabel('fraction of sweeps', fontsize=14)

            ax3 = plt.subplot(323)
            plt.plot(time, f2, label='f2 observed', color='green')
            plt.plot(time, f2i, label='f2 expected', color='red')
            plt.xlabel('ms', fontsize=14)
            plt.ylabel('fraction of sweeps', fontsize=14)

            ax4 = plt.subplot(324)
            plt.plot(time, f2, label='f2 observed', color='green')
            plt.plot(time, f2i, label='f2 expected', color='red')
            ax4.legend(loc='upper center', fontsize=14)
            plt.xlim(0, 10)
            plt.xlabel('ms', fontsize=14)
            plt.ylabel('fraction of sweeps', fontsize=14)

            plt.tight_layout()
            plt.show()
            
        if plot_chi_sq:
            
            plt.figure(figsize=(14,6))

            ax5 = plt.subplot(211)
            plt.plot(time, pval_chi2, label='p value of chi2 test', color='black')
            plt.plot(time, np.full(time.size, 0.05), color='gray')
            plt.xlabel('ms', fontsize=14)
            plt.ylabel('p value', fontsize=14)

            ax6 = plt.subplot(212)
            plt.plot(time, pval_chi2, label='p value of chi2 test', color='black')
            plt.plot(time, np.full(time.size, 0.05), label='p = 0.05', color='gray')
            ax6.legend(loc='center', fontsize=14)
            plt.xlim(0, 10)
            plt.xlabel('ms', fontsize=14)
            plt.ylabel('p value', fontsize=14)
            plt.tight_layout()
            plt.show()
            
        if save:
            DF = pd.DataFrame([time, f0, f1, f2, f0i, f1i, f2i, pval_chi2]).T
            file_name = self.abf_file.split('.')[0]+'_f1f2.csv'
            DF.to_csv(file_name, header=['time, ms', 'f0', 'f1', 'f2', 'f0i', 'f1i', 'f2i', 'pval_chi2'], index=False)