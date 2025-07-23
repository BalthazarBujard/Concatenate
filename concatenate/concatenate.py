# file for concatenate class

from typing import Any, List, Tuple, Optional, Union
import numpy as np
from .utils.utils import detect_onsets, find_elems_in_range
from librosa.onset import onset_detect
import matplotlib.pyplot as plt


class TimeStamp():
    def __init__(self,times:Tuple[Union[int,float],Union[int,float]], 
                 new_times : Optional[Tuple[Union[int,float],Union[int,float]]] = None):
        
        self.__times = times #original timestamp before reassignment
        self.__new_times = new_times #timestamp in output
    
    @property
    def times(self):
        return self.__times
    
    @times.setter
    def times(self,times:Tuple[Union[int,float],Union[int,float]]):
        self.__times=times
    
    
    @property
    def new_times(self):
        return self.__new_times
    
    @new_times.setter 
    def new_times(self,new_times:Tuple[Union[int,float],Union[int,float]]):
        self.__new_times = new_times
    
    def from_seconds_to_samples(self,sampling_rate : int):
        self.times = [int(t*sampling_rate) for t in self.times]
        try:
            self.new_times = [int(t*sampling_rate) for t in self.new_times]
        except TypeError as e: #if new_times is None dont change it
            pass

    def from_samples_to_seconds(self,sampling_rate : int):
        self.times = [t/sampling_rate for t in self.times]
        try:
            self.new_times = [t/sampling_rate for t in self.new_times]
        except TypeError as e: #if new_times is None dont change it
            pass
    
    @property
    def duration(self):
        return self.times[1]-self.times[0]

class Concatenator():
    def __init__(self,verbose:bool=False):
        self.verbose=verbose
    
    def _generate_crossfade_window(self, fade_time : float, sampling_rate : int, in_out : str):
        f=1/(4*fade_time) #frequency of cos and sine windows (sin=1 and cos=0 at tmax=fade_time)
        t=np.linspace(0,fade_time,int(fade_time*sampling_rate))
        
        if in_out == "in":
            return np.sin(2*np.pi*f*t)**2

        elif in_out == "out":
            return np.cos(2*np.pi*f*t)**2
        
        else : raise ValueError()
    
    
    #TODO : add "new_times" argument
    def __call__(self, audio : np.ndarray, markers : List[TimeStamp], 
                 sampling_rate : int, fade_time : float, 
                 clean : bool, max_backtrack : float = None, new_times : Optional[List[Union[int,float]]] = None) -> np.ndarray:
        
        output = []
        start=0
        stop=1
        
        if clean:
            onsets, backtrack = detect_onsets(audio,sampling_rate,True) #compute onsets and backtrack once over whole memeory
            #to samples
            onsets = (onsets*sampling_rate).astype(int)
            backtrack = (backtrack*sampling_rate).astype(int)
        
        else :
            onsets, backtrack = None, None
        
        # fade_out_cp_time = None #fade out crossing point (in samples). None means no fade out crossing point (first segment)
        # delta_r = 0 #shift of right crossing point (fade out crossing point)
        
        if max_backtrack==None : max_backtrack = fade_time/2 #si plus grand que fade_t/2 il faudrait recalculer la fenetre 
        max_backtrack*=sampling_rate #to samples
        
        if new_times is not None : 
            assert len(new_times)==len(markers), "new_times and markers should have the same number of elements"
            
            for m, t0 in zip(markers, new_times) :
                m.new_times = [t0, t0+m.duration]
        
        #if no temporal sync is provided with new_times
        if all(m.new_times is None for m in markers):
            t0 = 0
            for m in markers:
                t1 = t0 + m.duration
                m.new_times = [t0,t1]
                t0 = t1
        
        #generate empty output that can contain all chunks
        output = np.zeros(markers[-1].new_times[1])
        
        while start < len(markers):
            output, stop = self._concatenate_one_step(audio, markers, output,
                                                                fade_time, sampling_rate, 
                                                                start, stop,
                                                                clean, onsets, backtrack, max_backtrack)
            
            #update fade out params
            # delta_r = t.times[1]-new_t.times[1] # right crossing point shift = t1 - t1'
            # fade_out_cp_time = new_t.times[1]
            #print("xr",delta_r)
            
            
            #update counters
            start = stop
            stop += 1
        
        return output 
    
    
    def _concatenate_one_step(self, audio : np.ndarray, markers : List[TimeStamp], output:np.ndarray,
                              fade_time : float, sampling_rate:int,
                              start:int, stop:int,
                              clean : bool, 
                              onsets : np.ndarray, backtrack:np.ndarray, max_backtrack:float)->Tuple[np.ndarray,int,TimeStamp,TimeStamp]:
        
        t0 = markers[start]
            
        #compute next continous segment. stop is the index of the non-consecutive index
        # if stop<len(markers):
        #     stop = self._generate_continous(markers, start, stop)
            
        t1 = markers[stop-1]
        
        first_slice = start==0
        last_slice = stop>len(markers)-1
        #print(first_slice,last_slice)
        
        #fade in crossing point is beginning of continous segment (start marker's t0) 
        #fade_in_cp_time = t0.times[0] 
        
        t = TimeStamp((t0.times[0],t1.times[1]),(t0.new_times[0],t1.new_times[1])) #timestamp of continous segment to concatenate
        
        new_t = t
        delta_l = 0
        delta_r = 0
        
        #clean continous segment of early/late attacks by shifting crossing points to closest backtrack time
        if clean :
            print("Finding best markers")
            new_t = self._find_best(t,onsets,backtrack,max_backtrack) 
            
            #update markers original time. only update borders of continous segment
            markers[start].times = [new_t.times[0],markers[start].times[1]] 
            markers[stop-1].times = [markers[stop-1].times[0],new_t.times[1]]
            
            #right shift computed after crossfade because we want the right shift of the last continous segment
            delta_l = t.times[0]-new_t.times[0]
            delta_r = t.times[1]-new_t.times[1]
            #print('xl',delta_l)
            
            #update fade_in_time
            #fade_in_cp_time = new_t.times[0]
        
        #crossfade between output and new segment
        output = self._process_crossfade_v2(output, audio, new_t, fade_time, sampling_rate, delta_l, delta_r, first_slice, last_slice)
        
        #print("output:",len(output)/sampling_rate)
        
        return output, stop#, t, new_t
    
    #find longest consecutive segments in markers   
    def _generate_continous(self, markers : List[TimeStamp], start : int, stop: int) -> int:
        
        t0,t1 = markers[start], markers[stop] #timestamps of markers "start" and "stop"
        
        #compute consecutive segments
        while t1.times[0] == t0.times[1]+1:
            t0 = t1
            stop += 1
            if stop == len(markers) : break
            t1 = markers[stop]
        
        return stop # stop is the index of the first NON-consecutive slice
    
    #THIS FUNCTION IS PROBABLY SPECIFIC TO DICY3.
    #IF YOU DONT WANT EARLY OR LATE ATTACKS WITH YOUR CONCATENATION, PROPERLY SEGMENT YOUR AUDIO BEFOREHAND
    def _find_best(self, t: TimeStamp, 
               onsets : np.ndarray[int], backtrack : np.ndarray[int], 
               max_backtrack : int) -> TimeStamp:
        
        t0,t1 = t.times #begin and end of segment
        
        lower = t1 - max_backtrack 
        onsets_ = find_elems_in_range(onsets,lower,t1) #look for onset in max_backtrack window
        if len(onsets_)>0:
            print("found onset close to end")
            onset = onsets_[0] #first onset above thresh
            #find backtrack before onset
            back = backtrack[onsets<=onset][-1] #get matching backtrack to onset as new end
            print("backtrack found at :", back)
            if abs(back-t1)<max_backtrack: #dont go too far away
                print("valid backtrack close to end found")
                #delta_r = t1-back #>=0
                t1 = back #assign new end of segment
        
        #close to beginning onset
        upper = t0 + max_backtrack
        onsets_ = find_elems_in_range(onsets,t0,upper)
        if len(onsets_)>0:
            print("found onset close to beginning")
            onset = onsets_[0] #first onset above thresh
            back = backtrack[onsets<=onset][-1] #get matching backtrack to onset as new end
            print("backtrack found at :", back)
            if abs(back-t0)<max_backtrack: #dont go too far away
                print("valid backtrack close to beginning found")
                #delta_l = t0-back # >0 : left shift, <0 : right shift
                t0 = back
        
        new_t = TimeStamp((t0,t1))
        
        return new_t
    
    # no longer extracting crossfade segements to avoid losing samples
    # def __extract_fade_segment(self,audio,fade_t,r,x):
    #     t0 = fade_t - (r+x)
    #     pad_l=0
    #     if t0<0:
    #         pad_l = abs(t0)
    #         t0 = 0
        
    #     t1 = fade_t + (r+x)
    #     pad_r = 0
    #     if t1>=len(audio):
    #         pad_r = t1-len(audio)+1
    #         t1 = len(audio)-1
        
    #     to_fade = audio[t0:t1]
    #     if pad_l>0:
    #         to_fade = np.concatenate([np.zeros(pad_l),to_fade])
    #     if pad_r>0:
    #         to_fade = np.concatenate([to_fade,np.zeros(pad_r)])
        
    #     return to_fade
    
    #method to add extra segments to output and new segment for crossfade
    def _prepare_segments_for_crossfade(self, audio : np.ndarray, output:np.ndarray, new_segment : np.ndarray, t : TimeStamp, fade_in_cp_time : int, fade_out_cp_time : int, delta : int):
        #output processing
        t0_out = fade_out_cp_time
        t1_out = t0_out+delta
        pad_r=0
        if t1_out >= len(audio):
            pad_r = t1_out-len(audio)+1
            t1_out = len(audio)-1 #limit to audio duration
            
        append = np.concatenate([audio[t0_out:t1_out],np.zeros(pad_r)]) #(right) extra segment for crossfade output 
        #print("output before append:",len(output)/sampling_rate)
        output = np.concatenate([output,append])
        #print("output after append:",len(output)/sampling_rate)
        
        #new_segment processing
        t0_in = fade_in_cp_time - delta
        t1_in = fade_in_cp_time
        pad_l=0
        if t0_in<0:
            pad_l = abs(t0_in)
            t0_in = 0 #limit to beginning of audio
        
        prepend = np.concatenate([np.zeros(pad_l),audio[t0_in:t1_in]]) #(left) extra segment for crossfade new segment
        #print("new segment before prepend:",len(new_segment)/sampling_rate)
        new_segment = np.concatenate([prepend,new_segment])
        #print("new segment after prepend:",len(new_segment)/sampling_rate)
        
        assert len(append)==len(prepend), print(len(append),len(prepend)) #security and debugging
        
        #half_length = len(prepend) #duration of half crossfade window
        
        return output, new_segment#, half_length #half_length is delta
            
    def _prepare_segment_for_crossfade(self, audio : np.ndarray, new_segment : np.ndarray, fade_in_cp_time : int, fade_out_cp_time : int, delta : int, first_slice : bool, last_slice : bool):
        
        append = []
        if not last_slice:
            #fade_out extra segment
            t0_out = fade_out_cp_time
            t1_out = t0_out+delta
            pad_r=0
            if t1_out >= len(audio):
                pad_r = t1_out-len(audio)+1
                t1_out = len(audio)-1 #limit to audio duration
                
            append = np.concatenate([audio[t0_out:t1_out],np.zeros(pad_r)]) #(right) extra segment
        
        prepend = []
        if not first_slice:
            #fade_in extra segment
            t0_in = fade_in_cp_time - delta
            t1_in = fade_in_cp_time
            pad_l=0
            if t0_in<0:
                pad_l = abs(t0_in)
                t0_in = 0 #limit to beginning of audio
            
            prepend = np.concatenate([np.zeros(pad_l),audio[t0_in:t1_in]]) #(left) extra segment
        
        #assert len(append)==len(prepend), print(len(append),len(prepend)) #security and debugging
        
        #print("new segment before prepend:",len(new_segment)/sampling_rate)
        new_segment = np.concatenate([prepend,new_segment, append])
        
        return new_segment
    
    #method to place new segment given the new_times attribute of the timestamp
    def _place_segment(self, output : np.ndarray, new_segment : np.ndarray, t : TimeStamp, delta : int):
        t0 = max(0,t.new_times[0] - delta) #position of segemnt in output
        t1 = min(len(output),t.new_times[1] + delta)
        
        #print(t0, t1, len(output))
        
        #pad new segment
        pad_l = t0 
        pad_r = len(output) - t1
        
        new_segment = np.concatenate([np.zeros(pad_l),new_segment,np.zeros(pad_r)])
        
        output += new_segment
        
        return output
    
    def _process_crossfade_v2(self, output : np.ndarray, audio : np.ndarray, t : TimeStamp,
                   fade_time : float, sampling_rate : int,
                   delta_l : int, delta_r : int,
                   first_slice : bool, last_slice : bool):
        
        if not t.duration > 0:
            raise RuntimeError("New chunk to crossfade is empty")
        
        if fade_time < 0.01 or fade_time > t.duration/sampling_rate:
            raise ValueError(f"Fade time should be gretaer than 10ms and shorter than the chunk duration ({t.duration/sampling_rate:.4f}s) !")
        
        fade_time = max(fade_time,0.01)
        r = int((fade_time/2) * sampling_rate) #delta
        
        #extract new segment to concatenate
        new_segment = audio[t.times[0]:t.times[1]]
        
        fade_in_cp_time = t.times[0]
        fade_out_cp_time = t.times[1]
        
        #TODO : IF CLEAN METHOD IS APPLIED LEFT AND RIGHT SHIFTS HAVE TO BE TAKEN INTO ACCOUNT WHEN SUMMING OUTPTU AND NEW_SEGMENT
        # d_max = max(delta_r,-delta_l)
        # delta = r+d_max #utiliser au lieu de r±delta_l/r
        
        #add extra segments for crossfade
        new_segment = self._prepare_segment_for_crossfade(audio, new_segment, fade_in_cp_time, fade_out_cp_time, r, first_slice, last_slice)
        #print("new_segment duration",len(new_segment)/sampling_rate)
        
        # generate and apply windows
        T_samples = 2*r #half_length
        T = T_samples/sampling_rate #crossfade effective duration

        cos = self._generate_crossfade_window(T,sampling_rate,'out')
        sin = self._generate_crossfade_window(T,sampling_rate,'in')
        
        new_segment[:T_samples] *= sin #apply fade in to beginning of new_segment
        new_segment[-T_samples:] *= cos #apply fade out to end of output
        
        # if output != []:
        #     #pad output and new segment before summing
        #     pad_output = len(new_segment)-T_samples #2*delta
        #     pad_new_segment = len(output)-T_samples #2*delta
        #     #print("pad_output:",pad_output/sampling_rate)
            
        #     output = np.concatenate([output,np.zeros(pad_output)])
        #     new_segment = np.concatenate([np.zeros(pad_new_segment),new_segment])
            
        #     output = new_segment+output
        
        # #first segment
        # else : 
        #     output = new_segment
        
        #print("output after crossfade:",len(output)/sampling_rate)
        
        output = self._place_segment(output, new_segment, t, r)
        
            
        return output
        
    
    def _process_crossfade(self, output : np.ndarray, audio : np.ndarray, t : TimeStamp,
                   fade_in_cp_time : int, fade_out_cp_time : int, #crossfade crossing point for fade in and fade out in samples
                   fade_time : float, sampling_rate : int,
                   delta_l : int, delta_r : int):
        
        if not t.duration > 0:
            raise RuntimeError("New chunk to crossfade is empty")
        
        if fade_time < 0.01 or fade_time > t.duration/sampling_rate:
            raise ValueError(f"Fade time should be gretaer than 10ms and shorter than the chunk duration ({t.duration/sampling_rate:.4f}s) !")
        
        fade_time = max(fade_time,0.01)
        r = int((fade_time/2) * sampling_rate) #delta
        
        #extract new segment to concatenate
        new_segment = audio[t.times[0]:t.times[1]]
        
        if fade_out_cp_time != None:
            
            d_max = max(delta_r,-delta_l)
            delta = r+d_max #utiliser au lieu de r±delta_l/r
            
            #add extra segment for crossfade to output and new_segment
            output, new_segment = self._prepare_segments_for_crossfade(audio, output, new_segment, t, fade_in_cp_time, fade_out_cp_time, delta)
            
            # generate and apply windows
            T_samples = 2*delta #half_length
            T = T_samples/sampling_rate #crossfade effective duration

            cos = self._generate_crossfade_window(T,sampling_rate,'out')
            sin = self._generate_crossfade_window(T,sampling_rate,'in')
            
            output[-T_samples:] *= cos #apply fade out to end of output
            new_segment[:T_samples] *= sin #apply fade in to beginning of new_segment
            
            if self.verbose:
                plt.plot(cos)
                plt.plot(output[-T_samples:]+0.1,label='fade out')
                plt.plot(sin)
                plt.plot(new_segment[:T_samples]-0.1,label='fade in')
                plt.plot(output[-T_samples:]+new_segment[:T_samples],label='output')
                plt.vlines(T_samples//2,ymin=0,ymax=1,label=f"cp @ {T_samples/2}", colors='r',linestyles='--')
                plt.legend(fontsize=9)
                plt.show()
            
            #pad output and new segment before summing
            pad_output = len(new_segment)-T_samples #2*delta
            pad_new_segment = len(output)-T_samples #2*delta
            #print("pad_output:",pad_output/sampling_rate)
            
            output = np.concatenate([output,np.zeros(pad_output)])
            new_segment = np.concatenate([np.zeros(pad_new_segment),new_segment])
            
            output = new_segment+output
            
            #print("output after crossfade:",len(output)/sampling_rate)
            
        
        #first segment
        elif fade_out_cp_time == None :
            #print("First segment")
            sin = self._generate_crossfade_window(fade_time,sampling_rate,'in')
            new_segment[:len(sin)] *= sin
            
            output = new_segment
            
            
        return output

class ConcatenateWithSilence():
    def __init__(self):
        pass
    
    def _generate_crossfade_window(self, fade_time : float, sampling_rate : int, in_out : str):
        f=1/(4*fade_time) #frequency of cos and sine windows (sin=1 and cos=0 at tmax=fade_time)
        t=np.linspace(0,fade_time,int(fade_time*sampling_rate))
        if in_out == "in":
            return np.sin(2*np.pi*f*t)
        elif in_out == "out":
            return np.cos(2*np.pi*f*t)
        
        else : raise ValueError()
    
    def __call__(self, audio : np.ndarray, markers : List[TimeStamp], sampling_rate : int, fade_time : float, clean : bool, max_backtrack : float = None):
        output = []
        start=0
        stop=1
        continious_lens=[]
        new_index = 0 # index for slice count in output
        
        #memory = np.concatenate(memory_chunks)
        onsets, backtrack = detect_onsets(audio.astype(np.float32),sampling_rate,True) #compute onsets and backtrack once over whole memeory
        #to samples
        onsets = (onsets*sampling_rate).astype(int)
        backtrack = (backtrack*sampling_rate).astype(int)
        
        fade_in_cp_time,fade_out_cp_time = None,None #fade in and out timestamps (in samples)
        delta_l, delta_r = 0,0 #left and right shift of crossing point
        
        if max_backtrack==None : max_backtrack = fade_time/2 #si plus grand que fade_t/2 il faudrait recalculer la fenetre 
        
        while start < len(markers):
            
            #check if silence
            t0 = markers[start]
            is_silence = t0.index==-1
            
            #compute next continous segment
            continous, stop = self._generate_continous(audio, markers, start, stop)
            
            continious_lens.append(len(continous)) #for statistics
            continous = np.concatenate(continous) #flatten
            
            #compute fade_in_time 
            fade_in_cp_time = t0.times[0] if not is_silence else None 
            
            t = TimeStamp((t0.times[0],t0.times[0]+len(continous)),new_index) #timestamp of segment to concatenate
            new_t = t
            #clean continous segment of early/late attacks
            if clean and not is_silence:
                
                new_t = self._find_best(t,onsets,backtrack,max_backtrack)
                
                continous = audio[new_t.times[0]:new_t.times[1]]
                
                #delta_l, delta_r = t.times[0]-new_t.times[0], t.times[1]-new_t.times[1] #left and right shift after cleaning onsets
                #right shift computed after crossfade because we want the right shift of the last continous segment
                delta_l = t.times[0]-new_t.times[0]
                
                #update fade_in_time
                fade_in_cp_time = new_t.times[0]
            
            #crossfade between output and new segment (continous)
            output = self._process_crossfade(output, audio, new_t, fade_in_cp_time, fade_out_cp_time, fade_time, sampling_rate, delta_l, delta_r)
            
            #update fade out params
            delta_r = t.times[1]-new_t.times[1]
            fade_out_cp_time = new_t.times[1] if not is_silence else None #if silence then no fade out
            
            #update counters
            start = stop
            stop += 1
            new_index += 1
        
        return output #and other variables ?
            
                
                
    #TODO : PAS BESOIN DE GENERER CONTINOUS ICI MEME ON PEUT JUSTE UTILISER LES TIMESTAMPS MAIS FAUT GERER SILENCE        
    def _generate_continous(self, audio : np.ndarray, markers : List[TimeStamp], start : int, stop: int) -> Tuple[List[List], int]:
        
        #border case where we end with an isolated segment
        if stop == len(markers):
            t0 = markers[start]
            is_silence = t0.index == -1
            continous = [audio[t0.times[0]:t0.times[1]].tolist()] if not is_silence else [[0]*t0.duration]
            return continous, stop
        
        t0,t1 = markers[start], markers[stop] #timestamps of markers "start" and "stop"
        
        is_silence = t0.index == -1 #flag if current slice is silence
        
        continous = [audio[t0.times[0]:t0.times[1]].tolist()] if not is_silence else [[0]*t0.duration] #init continous with first slice
        
        #compute consecutive silence
        if is_silence :
            while t1.index == -1:
                continous.append([0]*t1.duration)
                stop += 1
                if stop == len(markers) : break
                t1 = markers[stop]
        
        #compute consecutive segments
        else :
            while t1.index == t0.index+1 :
                continous.append(audio[t1.times[0]:t1.times[1]].tolist())
                t0 = t1
                stop += 1
                if stop == len(markers) : break
                t1 = markers[stop]
        
        
        return continous, stop #need stop value 
    
    def _find_best(self, t: TimeStamp, 
               onsets : np.ndarray[int], backtrack : np.ndarray[int], 
               max_backtrack : int) -> TimeStamp:
        
        t0,t1 = t.times #begin and end of segment
        #delta_r, delta_l = 0,0 #right and left shift after cleaning
            
        lower = t1 - max_backtrack 
        onsets_ = find_elems_in_range(onsets,lower,t1) #look for onset in max_backtrack window
        if len(onsets_)>0:
            onset = onsets_[0] #first onset above thresh
            #find backtrack before onset
            back = backtrack[onsets<=onset][-1] #get matching backtrack to onset as new end
            if abs(back-t1)<max_backtrack: #dont go too far away
                #delta_r = t1-back #>=0
                t1 = back #assign new end of segment
        
        #close to beginning onset
        upper = t0 + max_backtrack
        onsets_ = find_elems_in_range(onsets,t0,upper)
        if len(onsets_)>0:
            onset = onsets_[0] #first onset above thresh
            back = backtrack[onsets<=onset][-1] #get matching backtrack to onset as new end
            if abs(back-t0)<max_backtrack: #dont go too far away
                #delta_l = t0-back # >0 : left shift, <0 : right shift
                t0 = back
        
        new_t = TimeStamp((t0,t1),t.index)
        
        return new_t
    
    def __extract_fade_segment(self,audio,fade_t,r,x):
        t0 = fade_t - (r+x)
        pad_l=0
        if t0<0:
            pad_l = abs(t0)
            t0 = 0
        
        t1 = fade_t + (r+x)
        pad_r = 0
        if t1>=len(audio):
            pad_r = t1-len(audio)+1
            t1 = len(audio)-1
        
        to_fade = audio[t0:t1]
        if pad_l>0:
            to_fade = np.concatenate([np.zeros(pad_l),to_fade])
        if pad_r>0:
            to_fade = np.concatenate([to_fade,np.zeros(pad_r)])
        
        return to_fade
    
    #TODO : surement moyen d'eviter de faire les 4 cas et plutot faire cross en quand fade != None et a la fin concatener ?
    def _process_crossfade(self, output : np.ndarray, audio : np.ndarray, t : TimeStamp,
                   fade_in_cp_time : int, fade_out_cp_time : int, #times in samples
                   fade_time : float, sampling_rate : int,
                   delta_l : int, delta_r : int):
        
        #fade_in, fade_out = cross_fade_windows(fade_time, sampling_rate)
        r = int((fade_time/2) * sampling_rate) #delta
        
        if fade_in_cp_time != None and fade_out_cp_time != None:
            #-----extract segments to crossfade taking shift into account-------#
            
            #fade in segment
            to_fade_in = self.__extract_fade_segment(audio, fade_in_cp_time, r, -delta_l) # -delta_l cuz defined the other way
            
            #fade out segment
            to_fade_out = self.__extract_fade_segment(audio, fade_out_cp_time, r, delta_r)
                
            #-----generate crossfade windows-----#
            fade_time_in = len(to_fade_in)/sampling_rate
            fade_in = self._generate_crossfade_window(fade_time_in,sampling_rate,'in')
            
            fade_time_out = len(to_fade_out)/sampling_rate
            fade_out = self._generate_crossfade_window(fade_time_out, sampling_rate, 'out')
            
            #apply windows
            to_fade_in*=fade_in
            to_fade_out*=fade_out
            
            #---------sum crossfade segments of different size---------#
            delta = len(to_fade_in)-len(to_fade_out) #difference in crossfade windows size
            
            #ATTENTION IL PEUT Y AVOIR PROBLEME DANS LA GESTION DU PADDING QUAND T2 OU T0 DEPASSE BORNES [0,LEN(AUDIO)]
            
            if delta<0: #fade_out>fade_in
                #pad beginning of to_fade_in with d/2 zeros and append d/2 of continous to it 
                pad = np.zeros(delta//2)
                
                t1_in = min(len(audio)-1,fade_in_cp_time + (r-delta_l))
                t2 = t1_in + (delta-delta//2)
                pad_r=0
                if t2 >= len(audio):
                    pad_r = t2 - len(audio) +1
                    t2=len(audio)-1
                    append = np.concatenate([audio[t1_in:t2],np.zeros(pad_r)]) #take everything till end and pad with 0
                    
                else : append = audio[t1_in:t2]
                
                to_fade_in = np.concatenate([pad,to_fade_in,append])
                
            elif delta > 0: #fade_in>fade_out
                #pad end of to_fade_out and prepend d/2 of output
                pad = np.zeros(delta//2)
                
                t0_out = max(0,fade_out_cp_time-(r+delta_r))
                t0 = t0_out - (delta-delta//2)
                pad_l=0
                if t0<0:
                    pad_l = abs(t0)
                    t0=0
                    prepend = np.concatenate([np.zeros(pad_l),audio[t0:t0_out]])
                
                else : prepend = audio[t0:t0_out]
                
                to_fade_out = np.concatenate([prepend,to_fade_out,pad])
            
            #security & debugging
            assert len(to_fade_out)==len(to_fade_in)
            
            crossfade = to_fade_in+to_fade_out
            
            #------concatenate all together------#
            T = len(crossfade)
            
            print("output :-T//2",len(output)/sampling_rate,len(output[:-T//2])/sampling_rate)
            print("continous",len(audio[t.times[0]+T//2:t.times[1]])/sampling_rate)
            print("crossfade",len(crossfade)/sampling_rate)
            
            output = np.concatenate([output[:-T//2],crossfade,audio[t.times[0]+T//2:t.times[1]]])
            
            print("output", len(output)/sampling_rate)
        
        #new segment is silence
        elif fade_in_cp_time == None and fade_out_cp_time != None:
            print('on ne devrait pas rentrer ici !')
            #fade out segment
            to_fade_out = self.__extract_fade_segment(audio, fade_out_cp_time, r, delta_r)
            fade_time_out = len(to_fade_out)/sampling_rate
            fade_out = self._generate_crossfade_window(fade_time_out, sampling_rate, 'out')
            to_fade_out*=fade_out
            
            crossfade = to_fade_out
            T = len(crossfade)
            
            output = np.concatenate([output[:-T//2],crossfade,[0]*(t.duration-T//2)])
        
        #previous segment is silence or first segment
        elif fade_in_cp_time != None and fade_out_cp_time == None :
            print("First segment or previous was silent")
            to_fade_in = self.__extract_fade_segment(audio, fade_in_cp_time, r, -delta_l)
            fade_time_in = len(to_fade_in)/sampling_rate
            fade_in = self._generate_crossfade_window(fade_time_in,sampling_rate,'in')
            to_fade_in *= fade_in
            
            crossfade = to_fade_in
            T = len(crossfade)
            
            if len(output)>0:
                output = np.concatenate([output[:-T//2],crossfade,audio[t.times[0]+T//2:t.times[1]]])
            
            else :
                output = np.concatenate([crossfade,audio[t.times[0]+T:t.times[1]]])
            
            print(len(output)/sampling_rate)
            
            
        else :
            raise RuntimeError("There should not be a case where fade_in and fade_out are None")
            
        return output
                 
            
            
            
            
            
            
            
            
            
            
    
    