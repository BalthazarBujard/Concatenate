# file for concatenate class

from typing import Any, List, Tuple
import numpy as np
from .utils.utils import detect_onsets, find_elems_in_range
from librosa.onset import onset_detect

def cross_fade_windows(fade_time,sampling_rate):
    f=1/(4*fade_time) #frequency of cos and sine windows (sin=1 and cos=0 at tmax=fade_time)
    t=np.linspace(0,fade_time,int(fade_time*sampling_rate))
    fade_in = np.sin(2*np.pi*f*t)
    fade_out = np.cos(2*np.pi*f*t)
    
    return fade_in,fade_out

#TODO : Rajouter timestamp avant/apres ?
class TimeStamp():
    def __init__(self,times:Tuple[int,int],index:int):
        # times should be represented in samples and index as slice index
        self.__times = times
        self.__index = index
    
    @property
    def times(self):
        return self.__times
    @property 
    def index(self):
        return self.__index
    
    @property
    def duration(self):
        return self.times[1]-self.times[0]

class Concatenate():
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
    
    def __call__(self, audio : np.ndarray, queries : List[TimeStamp], 
                 sampling_rate : int, fade_time : float, 
                 clean : bool, max_backtrack : float = None) -> np.ndarray:
        
        response = []
        start=0
        stop=1
        continious_lens=[]
        new_index = 0 # index for slice count in response
        
        #memory = np.concatenate(memory_chunks)
        #onsets, backtrack = detect_onsets(audio.astype(np.float32),sampling_rate,True) #compute onsets and backtrack once over whole memeory
        #to samples
        onsets = onset_detect(y=audio,sr=sampling_rate,backtrack=False,units='samples')
        backtrack = onset_detect(y=audio,sr=sampling_rate,backtrack=True,units='samples')
        #onsets = (onsets*sampling_rate).astype(int)
        #backtrack = (backtrack*sampling_rate).astype(int)
        
        fade_in_t,fade_out_t = None,None #fade in and out timestamps (in samples)
        x_l, x_r = 0,0 #left and right shift of crossing point
        
        if max_backtrack==None : max_backtrack = fade_time/2 #si plus grand que fade_t/2 il faudrait recalculer la fenetre 
        max_backtrack*=sampling_rate #to samples
        
        while start < len(queries):
            
            #check if silence
            t0 = queries[start]
            #is_silence = t0.index==-1
            
            #compute next continous segment
            continous, stop = self._generate_continous(audio, queries, start, stop)
            
            continious_lens.append(len(continous)) #for statistics
            continous = np.concatenate(continous) #flatten
            
            #compute fade_in_time 
            fade_in_t = t0.times[0] #if not is_silence else None 
            
            t = TimeStamp((t0.times[0],t0.times[0]+len(continous)),new_index) #timestamp of segment to concatenate
            new_t = t
            #clean continous segment of early/late attacks
            if clean :#and not is_silence:
                print("Cleaning")
                new_t = self._clean(t,onsets,backtrack,max_backtrack)
                
                continous = audio[new_t.times[0]:new_t.times[1]]
                
                #x_l, x_r = t.times[0]-new_t.times[0], t.times[1]-new_t.times[1] #left and right shift after cleaning onsets
                #right shift computed after crossfade because we want the right shift of the last continous segment
                x_l = t.times[0]-new_t.times[0]
                print('xl',x_l)
                
                #update fade_in_time
                fade_in_t = new_t.times[0]
            
            #crossfade between response and new segment (continous)
            response = self._crossfade(response, audio, new_t, fade_in_t, fade_out_t, fade_time, sampling_rate, x_l, x_r)
            
            #update fade out params
            x_r = t.times[1]-new_t.times[1]
            fade_out_t = new_t.times[1] #if not is_silence else None #if silence then no fade out
            print("xr",x_r)
            
            #update counters
            start = stop
            stop += 1
            new_index += 1
        
        return response #and other variables ?
            
                
                
    #TODO : PAS BESOIN DE GENERER CONTINOUS ICI MEME ON PEUT JUSTE UTILISER LES TIMESTAMPS       
    def _generate_continous(self, audio : np.ndarray, queries : List[TimeStamp], start : int, stop: int) -> Tuple[List[List], int]:
        
        #border case where we end with an isolated segment
        if stop == len(queries):
            t0 = queries[start]
            #is_silence = t0.index == -1
            continous = [audio[t0.times[0]:t0.times[1]].tolist()] #if not is_silence else [[0]*t0.duration]
            return continous, stop
        
        t0,t1 = queries[start], queries[stop] #timestamps of queries "start" and "stop"
        
        #is_silence = t0.index == -1 #flag if current slice is silence
        
        continous = [audio[t0.times[0]:t0.times[1]].tolist()] #if not is_silence else [[0]*t0.duration] #init continous with first slice
        
        #compute consecutive silence
        # if is_silence :
        #     while t1.index == -1:
        #         continous.append([0]*t1.duration)
        #         stop += 1
        #         if stop == len(queries) : break
        #         t1 = queries[stop]
        
        #compute consecutive segments
        while t1.times[0] == t0.times[1] :
            continous.append(audio[t1.times[0]:t1.times[1]].tolist())
            t0 = t1
            stop += 1
            if stop == len(queries) : break
            t1 = queries[stop]
        
        
        return continous, stop #need stop value 
    
    def _clean(self, t: TimeStamp, 
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
                #x_r = t1-back #>=0
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
                #x_l = t0-back # >0 : left shift, <0 : right shift
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
    def _crossfade(self, response : np.ndarray, audio : np.ndarray, t : TimeStamp,
                   fade_in_t : int, fade_out_t : int, #times in samples
                   fade_time : float, sampling_rate : int,
                   x_l : int, x_r : int):
        
        #fade_in, fade_out = cross_fade_windows(fade_time, sampling_rate)
        r = int((fade_time/2) * sampling_rate) #delta
        
        if fade_in_t != None and fade_out_t != None:
            #-----extract segments to crossfade taking shift into account-------#
            
            #fade in segment
            to_fade_in = self.__extract_fade_segment(audio, fade_in_t, r, -x_l) # -x_l cuz defined the other way
            
            #fade out segment
            to_fade_out = self.__extract_fade_segment(audio, fade_out_t, r, x_r)
                
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
            
            if delta<0: #fade_out>fade_in --> pad fade_in 
                print("pad fade_in")
                delta = abs(delta)
                #pad beginning of to_fade_in with d/2 zeros and append d/2 of continous to it 
                pad = np.zeros(delta//2)
                
                t1_in = fade_in_t + (r-x_l) #min(len(audio)-1,fade_in_t + (r-x_l))
                t2 = t1_in + (delta-delta//2)
                pad_r=0
                if t2 >= len(audio):
                    #get audio before pad_r (if there is any left)
                    c=[]
                    if t1_in < len(audio):
                        c=audio[t1_in:]
                    
                    pad_r = t2 - len(audio)+1 -len(c)
                    
                    append = np.concatenate([c,np.zeros(pad_r)])
                    
                else : append = audio[t1_in:t2]
                
                to_fade_in = np.concatenate([pad,to_fade_in,append])
                
            elif delta > 0: #fade_in>fade_out --> pad fade out
                print("pad fade_out")
                #pad end of to_fade_out and prepend d/2 of response
                pad = np.zeros(delta//2)
                
                t0_out = fade_out_t-(r+x_r) #max(0,fade_out_t-(r+x_r))
                t0 = t0_out - (delta-delta//2)
                pad_l=0
                if t0<0:
                    #get audio after pad_l (if there is any)
                    c=[]
                    if t0_out>0: #>= ?
                        c=audio[:t0_out]
                    
                    pad_l = t0_out - t0 - len(c)
                    
                    prepend=np.concatenate([np.zeros(pad_l),c])
                
                else : prepend = audio[t0:t0_out]
                
                to_fade_out = np.concatenate([prepend,to_fade_out,pad])
            
            #security & debugging
            assert len(to_fade_out)==len(to_fade_in)
            
            crossfade = to_fade_in+to_fade_out
            
            #------concatenate all together------#
            T = len(crossfade)
            
            print("response :-T//2",len(response)/sampling_rate,len(response[:-T//2])/sampling_rate)
            print("continous",len(audio[t.times[0]+T//2:t.times[1]])/sampling_rate)
            print("crossfade",T/sampling_rate)
            
            response = np.concatenate([response[:-T//2],crossfade,audio[t.times[0]+T//2:t.times[1]]])
            
            print("response", len(response)/sampling_rate)
        
        # #new segment is silence
        # elif fade_in_t == None and fade_out_t != None:
        #     print('on ne devrait pas rentrer ici !')
        #     #fade out segment
        #     to_fade_out = self.__extract_fade_segment(audio, fade_out_t, r, x_r)
        #     fade_time_out = len(to_fade_out)/sampling_rate
        #     fade_out = self._generate_crossfade_window(fade_time_out, sampling_rate, 'out')
        #     to_fade_out*=fade_out
            
        #     crossfade = to_fade_out
        #     T = len(crossfade)
            
        #     response = np.concatenate([response[:-T//2],crossfade,[0]*(t.duration-T//2)])
        
        #first segment
        elif fade_in_t != None and fade_out_t == None :
            print("First segment or previous was silent")
            to_fade_in = self.__extract_fade_segment(audio, fade_in_t, r, -x_l)
            fade_time_in = len(to_fade_in)/sampling_rate
            fade_in = self._generate_crossfade_window(fade_time_in,sampling_rate,'in')
            
            #ATTENYTION CA ARRIVE QU'IL Y AIT PROBLEME D'ARRONDI : TROUVER SOLUTION PLUS PROPRE
            if len(to_fade_in)!=len(fade_in):
                d=len(to_fade_in)-len(fade_in)
                if d>0:
                    fade_in=np.concatenate([fade_in,np.ones(abs(d))])
                else :
                    to_fade_in = np.concatenate([to_fade_in,np.zeros(d)])
            
            to_fade_in *= fade_in
            
            crossfade = to_fade_in
            T = len(crossfade)
            
            response = np.concatenate([crossfade,audio[t.times[0]+T:t.times[1]]])
            
        else :
            raise RuntimeError("There should not be a case where fade_in and fade_out are None")
            
        return response

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
    
    def __call__(self, audio : np.ndarray, queries : List[TimeStamp], sampling_rate : int, fade_time : float, clean : bool, max_backtrack : float = None):
        response = []
        start=0
        stop=1
        continious_lens=[]
        new_index = 0 # index for slice count in response
        
        #memory = np.concatenate(memory_chunks)
        onsets, backtrack = detect_onsets(audio.astype(np.float32),sampling_rate,True) #compute onsets and backtrack once over whole memeory
        #to samples
        onsets = (onsets*sampling_rate).astype(int)
        backtrack = (backtrack*sampling_rate).astype(int)
        
        fade_in_t,fade_out_t = None,None #fade in and out timestamps (in samples)
        x_l, x_r = 0,0 #left and right shift of crossing point
        
        if max_backtrack==None : max_backtrack = fade_time/2 #si plus grand que fade_t/2 il faudrait recalculer la fenetre 
        
        while start < len(queries):
            
            #check if silence
            t0 = queries[start]
            is_silence = t0.index==-1
            
            #compute next continous segment
            continous, stop = self._generate_continous(audio, queries, start, stop)
            
            continious_lens.append(len(continous)) #for statistics
            continous = np.concatenate(continous) #flatten
            
            #compute fade_in_time 
            fade_in_t = t0.times[0] if not is_silence else None 
            
            t = TimeStamp((t0.times[0],t0.times[0]+len(continous)),new_index) #timestamp of segment to concatenate
            new_t = t
            #clean continous segment of early/late attacks
            if clean and not is_silence:
                
                new_t = self._clean(t,onsets,backtrack,max_backtrack)
                
                continous = audio[new_t.times[0]:new_t.times[1]]
                
                #x_l, x_r = t.times[0]-new_t.times[0], t.times[1]-new_t.times[1] #left and right shift after cleaning onsets
                #right shift computed after crossfade because we want the right shift of the last continous segment
                x_l = t.times[0]-new_t.times[0]
                
                #update fade_in_time
                fade_in_t = new_t.times[0]
            
            #crossfade between response and new segment (continous)
            response = self._crossfade(response, audio, new_t, fade_in_t, fade_out_t, fade_time, sampling_rate, x_l, x_r)
            
            #update fade out params
            x_r = t.times[1]-new_t.times[1]
            fade_out_t = new_t.times[1] if not is_silence else None #if silence then no fade out
            
            #update counters
            start = stop
            stop += 1
            new_index += 1
        
        return response #and other variables ?
            
                
                
    #TODO : PAS BESOIN DE GENERER CONTINOUS ICI MEME ON PEUT JUSTE UTILISER LES TIMESTAMPS MAIS FAUT GERER SILENCE        
    def _generate_continous(self, audio : np.ndarray, queries : List[TimeStamp], start : int, stop: int) -> Tuple[List[List], int]:
        
        #border case where we end with an isolated segment
        if stop == len(queries):
            t0 = queries[start]
            is_silence = t0.index == -1
            continous = [audio[t0.times[0]:t0.times[1]].tolist()] if not is_silence else [[0]*t0.duration]
            return continous, stop
        
        t0,t1 = queries[start], queries[stop] #timestamps of queries "start" and "stop"
        
        is_silence = t0.index == -1 #flag if current slice is silence
        
        continous = [audio[t0.times[0]:t0.times[1]].tolist()] if not is_silence else [[0]*t0.duration] #init continous with first slice
        
        #compute consecutive silence
        if is_silence :
            while t1.index == -1:
                continous.append([0]*t1.duration)
                stop += 1
                if stop == len(queries) : break
                t1 = queries[stop]
        
        #compute consecutive segments
        else :
            while t1.index == t0.index+1 :
                continous.append(audio[t1.times[0]:t1.times[1]].tolist())
                t0 = t1
                stop += 1
                if stop == len(queries) : break
                t1 = queries[stop]
        
        
        return continous, stop #need stop value 
    
    def _clean(self, t: TimeStamp, 
               onsets : np.ndarray[int], backtrack : np.ndarray[int], 
               max_backtrack : int) -> TimeStamp:
        
        t0,t1 = t.times #begin and end of segment
        #x_r, x_l = 0,0 #right and left shift after cleaning
            
        lower = t1 - max_backtrack 
        onsets_ = find_elems_in_range(onsets,lower,t1) #look for onset in max_backtrack window
        if len(onsets_)>0:
            onset = onsets_[0] #first onset above thresh
            #find backtrack before onset
            back = backtrack[onsets<=onset][-1] #get matching backtrack to onset as new end
            if abs(back-t1)<max_backtrack: #dont go too far away
                #x_r = t1-back #>=0
                t1 = back #assign new end of segment
        
        #close to beginning onset
        upper = t0 + max_backtrack
        onsets_ = find_elems_in_range(onsets,t0,upper)
        if len(onsets_)>0:
            onset = onsets_[0] #first onset above thresh
            back = backtrack[onsets<=onset][-1] #get matching backtrack to onset as new end
            if abs(back-t0)<max_backtrack: #dont go too far away
                #x_l = t0-back # >0 : left shift, <0 : right shift
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
    def _crossfade(self, response : np.ndarray, audio : np.ndarray, t : TimeStamp,
                   fade_in_t : int, fade_out_t : int, #times in samples
                   fade_time : float, sampling_rate : int,
                   x_l : int, x_r : int):
        
        #fade_in, fade_out = cross_fade_windows(fade_time, sampling_rate)
        r = int((fade_time/2) * sampling_rate) #delta
        
        if fade_in_t != None and fade_out_t != None:
            #-----extract segments to crossfade taking shift into account-------#
            
            #fade in segment
            to_fade_in = self.__extract_fade_segment(audio, fade_in_t, r, -x_l) # -x_l cuz defined the other way
            
            #fade out segment
            to_fade_out = self.__extract_fade_segment(audio, fade_out_t, r, x_r)
                
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
                
                t1_in = min(len(audio)-1,fade_in_t + (r-x_l))
                t2 = t1_in + (delta-delta//2)
                pad_r=0
                if t2 >= len(audio):
                    pad_r = t2 - len(audio) +1
                    t2=len(audio)-1
                    append = np.concatenate([audio[t1_in:t2],np.zeros(pad_r)]) #take everything till end and pad with 0
                    
                else : append = audio[t1_in:t2]
                
                to_fade_in = np.concatenate([pad,to_fade_in,append])
                
            elif delta > 0: #fade_in>fade_out
                #pad end of to_fade_out and prepend d/2 of response
                pad = np.zeros(delta//2)
                
                t0_out = max(0,fade_out_t-(r+x_r))
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
            
            print("response :-T//2",len(response)/sampling_rate,len(response[:-T//2])/sampling_rate)
            print("continous",len(audio[t.times[0]+T//2:t.times[1]])/sampling_rate)
            print("crossfade",len(crossfade)/sampling_rate)
            
            response = np.concatenate([response[:-T//2],crossfade,audio[t.times[0]+T//2:t.times[1]]])
            
            print("response", len(response)/sampling_rate)
        
        #new segment is silence
        elif fade_in_t == None and fade_out_t != None:
            print('on ne devrait pas rentrer ici !')
            #fade out segment
            to_fade_out = self.__extract_fade_segment(audio, fade_out_t, r, x_r)
            fade_time_out = len(to_fade_out)/sampling_rate
            fade_out = self._generate_crossfade_window(fade_time_out, sampling_rate, 'out')
            to_fade_out*=fade_out
            
            crossfade = to_fade_out
            T = len(crossfade)
            
            response = np.concatenate([response[:-T//2],crossfade,[0]*(t.duration-T//2)])
        
        #previous segment is silence or first segment
        elif fade_in_t != None and fade_out_t == None :
            print("First segment or previous was silent")
            to_fade_in = self.__extract_fade_segment(audio, fade_in_t, r, -x_l)
            fade_time_in = len(to_fade_in)/sampling_rate
            fade_in = self._generate_crossfade_window(fade_time_in,sampling_rate,'in')
            to_fade_in *= fade_in
            
            crossfade = to_fade_in
            T = len(crossfade)
            
            if len(response)>0:
                response = np.concatenate([response[:-T//2],crossfade,audio[t.times[0]+T//2:t.times[1]]])
            
            else :
                response = np.concatenate([crossfade,audio[t.times[0]+T:t.times[1]]])
            
            print(len(response)/sampling_rate)
            
            
        else :
            raise RuntimeError("There should not be a case where fade_in and fade_out are None")
            
        return response
                 
            
            
            
            
            
            
            
            
            
            
    
    