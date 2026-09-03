from .capsule_migration import CAPSULE_ROOT
import numpy as np
from scipy import stats
import statsmodels.api as sm
from joblib import Parallel, delayed
import re
from PyPDF2 import PdfMerger
import pandas as pd
import os
# from aind_dynamic_foraging_basic_analysis.lick_analysis import load_nwb
from scipy.stats import norm
import ast
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session, plot_foraging_session_nwb
from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_data
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
try:
    from .capsule_migration import capsule_directories
    from .mat_converter import load_df_from_mat
except ImportError:
    from capsule_migration import capsule_directories
    from mat_converter import load_df_from_mat
from uuid import uuid4
import json
from datetime import datetime
import logging
from dateutil.tz import tzlocal
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject
from hdmf_zarr import NWBZarrIO
from scipy.io import loadmat
import pickle
import shutil

import itertools

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from sklearn.metrics import r2_score

import spikeinterface as si

# save_folder=CAPSULE_ROOT + R'/scratch/check_rwd_licks'

logger = logging.getLogger(__name__)

def _get_field(obj, field_list, reject_list=[None, np.nan,'',[]], index=None, default=np.nan):
    """get field from obj, if not found, return default

    Parameters
    ----------
    obj : the object to get the field from
    field : str or list
            if is a list, try one by one until one is found in the obj (for back-compatibility)
    reject_list: list, optional
            if the value is in the reject_list, reject this value and continue to search the next field name in the field_list
    index: int, optional
            if index is not None and the field is a list, return the index-th element of the field
            otherwise, return default
    default : _type_, optional
        _description_, by default np.nan
    """
    if type(field_list) is not list:
        field_list = [field_list]
        
    for f in field_list:
            has_field=0
            if type(obj) is type:
                if hasattr(obj, f):
                    value = getattr(obj, f)
                    has_field=1
            # the obj.Opto_dialog is a dictionary
            elif type(obj) is dict:
                if f in obj:
                    value = obj[f]
                    has_field=1
            if has_field==0:
                continue
            if value in reject_list:
                continue
            if index is None:
                return value
            # If index is int, try to get the index-th element of the field
            try:
                if value[index] in reject_list:
                    continue
                return value[index]
            except:
                logger.debug(f"Field {field_list} is iterable or index {index} is out of range")
                return default
    else:
        logger.debug(f"Field {field_list} not found in the object")
        return default
    

######## load the Json/Mat file #######
def bonsai_to_nwb(fname, save_file=None):
    if fname.endswith('.mat'):
        Obj = loadmat(fname)
        for key in Obj.keys():
            Value=Obj[key]
            if key=='B_SuggestedWater':
                pass
            if isinstance(Value, np.ndarray):
                if Value.shape[0]==1:
                    try:
                        Value=Value.reshape(Value.shape[1],)
                    except:
                        pass
                if Value.shape==(1,):
                    Obj[key]=Value.tolist()[0]
                else:
                    Obj[key]=Value.tolist()
    elif fname.endswith('.json'):
        f = open (fname, "r")
        Obj = json.loads(f.read())
        f.close()
        
    # transfer dictionary to class
    class obj:
        pass
    for attr_name in Obj.keys():
        setattr(obj, attr_name, Obj[attr_name])
     
    # Early return if missing some key fields
    if any([not hasattr(obj, field) for field in ['B_TrialEndTime', 'TP_BaseRewardSum']]):
        logger.warning(f"Missing key fields! Skipping {fname}")
        return 'incomplete_json'
    
    if not hasattr(obj, 'Other_SessionStartTime'):
        session_start_timeC=datetime.strptime('2023-04-26', "%Y-%m-%d") # specific for LA30_2023-04-27.json
    else:
        session_start_timeC=datetime.strptime(obj.Other_SessionStartTime, '%Y-%m-%d %H:%M:%S.%f')
    
    # add local time zone explicitly
    session_start_timeC = session_start_timeC.replace(tzinfo=tzlocal())

    ### session related information ###
    nwbfile = NWBFile(
        session_description='Session end time:'+_get_field(obj, 'Other_CurrentTime', default='None'),  
        identifier=str(uuid4()),  # required
        session_start_time= session_start_timeC,  # required
        session_id=os.path.basename(fname),  # optional
        experimenter=[
            _get_field(obj, 'Experimenter', default='None'),
        ],  # optional
        lab="",  # optional
        institution="Allen Institute for Neural Dynamics",  # optional
        ### add optogenetics description (the target brain areas). 
        experiment_description="",  # optional
        related_publications="",  # optional
        notes=obj.ShowNotes,
        protocol=obj.Task
    )

    #######  Animal information #######
    # need to get more subject information through subject_id
    nwbfile.subject = Subject(
        subject_id=obj.ID,
        description='Animal name:'+obj.ID,
        species="Mus musculus",
        weight=_get_field(obj, 'WeightAfter',default=np.nan),
    )
    # print(nwbfile)
    
    ### Add some meta data to the scratch (rather than the session description) ###
    # Handle water info (with better names)
    BS_TotalReward = _get_field(obj, 'BS_TotalReward')
    # Turn uL to mL if the value is too large
    water_in_session_foraging = BS_TotalReward / 1000 if BS_TotalReward > 5.0 else BS_TotalReward 
    # Old name "ExtraWater" goes first because old json has a wrong Suggested Water
    water_after_session = float(_get_field(obj, 
                                           field_list=['ExtraWater', 'SuggestedWater'], default=np.nan
                                           ))
    water_day_total = float(_get_field(obj, 'TotalWater'))
    water_in_session_total = water_day_total - water_after_session
    water_in_session_manual = water_in_session_total - water_in_session_foraging
    Opto_dialog=_get_field(obj, 'Opto_dialog',default='None')
    metadata = {
        # Meta
        'box': _get_field(obj, ['box', 'Tower']),
        'session_end_time': _get_field(obj, 'Other_CurrentTime'),
        'session_run_time_in_min': _get_field(obj, 'Other_RunningTime'),
        
        # Water (all in mL)
        'water_in_session_foraging': water_in_session_foraging, 
        'water_in_session_manual': water_in_session_manual,
        'water_in_session_total':  water_in_session_total,
        'water_after_session': water_after_session,
        'water_day_total': water_day_total,

        # Weight
        'base_weight': float(_get_field(obj, 'BaseWeight')),
        'target_weight': float(_get_field(obj, 'TargetWeight')),
        'target_weight_ratio': float(_get_field(obj, 'TargetRatio')),
        'weight_after': float(_get_field(obj, 'WeightAfter')),
        
        # Performance
        'foraging_efficiency': _get_field(obj, 'B_for_eff_optimal'),
        'foraging_efficiency_with_actual_random_seed': _get_field(obj, 'B_for_eff_optimal_random_seed'),

        # Optogenetics
        'laser_1_calibration_power': float(_get_field(Opto_dialog, 'laser_1_calibration_power')),
        'laser_2_calibration_power': float(_get_field(Opto_dialog, 'laser_2_calibration_power')),
        'laser_1_target_areas': _get_field(Opto_dialog, 'laser_1_target',default='None'),
        'laser_2_target_areas': _get_field(Opto_dialog, 'laser_2_target',default='None'),

        # Behavior control software version
        'commit_ID':_get_field(obj, 'commit_ID',default='None'),
        'repo_url':_get_field(obj, 'repo_url',default='None'),
        'current_branch':_get_field(obj, 'current_branch',default='None'),
    }

    # Turn the metadata into a DataFrame in order to add it to the scratch
    df_metadata = pd.DataFrame(metadata, index=[0])

    # Are there any better places to add arbitrary meta data in nwb?
    # I don't bother creating an nwb "extension"...
    # To retrieve the metadata, use:
    # nwbfile.scratch['metadata'].to_dataframe()
    nwbfile.add_scratch(df_metadata, 
                        name="metadata",
                        description="Some important session-wise meta data")


    #######       Add trial     #######
    ## behavior events (including trial start/end time; left/right lick time; give left/right reward time) ##
    nwbfile.add_trial_column(name='animal_response', description=f'The response of the animal. 0, left choice; 1, right choice; 2, no response')
    nwbfile.add_trial_column(name='rewarded_historyL', description=f'The reward history of left lick port')
    nwbfile.add_trial_column(name='rewarded_historyR', description=f'The reward history of right lick port')
    nwbfile.add_trial_column(name='delay_start_time', description=f'The delay start time')
    nwbfile.add_trial_column(name='goCue_start_time', description=f'The go cue start time')
    nwbfile.add_trial_column(name='reward_outcome_time', description=f'The reward outcome time (reward/no reward/no response) Note: This is in fact time when choice is registered.')
    ## training paramters ##
    # behavior structure
    nwbfile.add_trial_column(name='bait_left', description=f'Whether the current left lickport has a bait or not')
    nwbfile.add_trial_column(name='bait_right', description=f'Whether the current right lickport has a bait or not')
    nwbfile.add_trial_column(name='base_reward_probability_sum', description=f'The summation of left and right reward probability')
    nwbfile.add_trial_column(name='reward_probabilityL', description=f'The reward probability of left lick port')
    nwbfile.add_trial_column(name='reward_probabilityR', description=f'The reward probability of right lick port')
    nwbfile.add_trial_column(name='reward_random_number_left', description=f'The random number used to determine the reward of left lick port')
    nwbfile.add_trial_column(name='reward_random_number_right', description=f'The random number used to determine the reward of right lick port')
    nwbfile.add_trial_column(name='left_valve_open_time', description=f'The left valve open time')
    nwbfile.add_trial_column(name='right_valve_open_time', description=f'The right valve open time')
    # block
    nwbfile.add_trial_column(name='block_beta', description=f'The beta of exponential distribution to generate the block length')
    nwbfile.add_trial_column(name='block_min', description=f'The minimum length allowed for each block')
    nwbfile.add_trial_column(name='block_max', description=f'The maxmum length allowed for each block')
    nwbfile.add_trial_column(name='min_reward_each_block', description=f'The minimum reward allowed for each block')
    # delay duration
    nwbfile.add_trial_column(name='delay_beta', description=f'The beta of exponential distribution to generate the delay duration(s)')
    nwbfile.add_trial_column(name='delay_min', description=f'The minimum duration(s) allowed for each delay')
    nwbfile.add_trial_column(name='delay_max', description=f'The maxmum duration(s) allowed for each delay')
    nwbfile.add_trial_column(name='delay_duration', description=f'The expected time duration between delay start and go cue start')
    # ITI duration
    nwbfile.add_trial_column(name='ITI_beta', description=f'The beta of exponential distribution to generate the ITI duration(s)')
    nwbfile.add_trial_column(name='ITI_min', description=f'The minimum duration(s) allowed for each ITI')
    nwbfile.add_trial_column(name='ITI_max', description=f'The maxmum duration(s) allowed for each ITI')
    nwbfile.add_trial_column(name='ITI_duration', description=f'The expected time duration between trial start and ITI start')
    # response duration
    nwbfile.add_trial_column(name='response_duration', description=f'The maximum time that the animal must make a choce in order to get a reward')
    # reward consumption duration
    nwbfile.add_trial_column(name='reward_consumption_duration', description=f'The duration for the animal to consume the reward')
    # reward delay
    nwbfile.add_trial_column(name='reward_delay', description=f'The delay between choice and reward delivery')
    # auto water
    nwbfile.add_trial_column(name='auto_waterL', description=f'Autowater given at Left')
    nwbfile.add_trial_column(name='auto_waterR', description=f'Autowater given at Right')
    # optogenetics
    nwbfile.add_trial_column(name='laser_on_trial', description=f'Trials with laser stimulation')
    nwbfile.add_trial_column(name='laser_wavelength', description=f'The wavelength of laser or LED')
    nwbfile.add_trial_column(name='laser_location', description=f'The target brain areas')
    nwbfile.add_trial_column(name='laser_1_power', description=f'The laser power of the laser 1(mw)')
    nwbfile.add_trial_column(name='laser_2_power', description=f'The laser power of the laser 2(mw)')
    nwbfile.add_trial_column(name='laser_on_probability', description=f'The laser on probability')
    nwbfile.add_trial_column(name='laser_duration', description=f'The laser duration')
    nwbfile.add_trial_column(name='laser_condition', description=f'The laser on is conditioned on LaserCondition')
    nwbfile.add_trial_column(name='laser_condition_probability', description=f'The laser on is conditioned on LaserCondition with a probability LaserConditionPro')
    nwbfile.add_trial_column(name='laser_start', description=f'Laser start is aligned to an event')
    nwbfile.add_trial_column(name='laser_start_offset', description=f'Laser start is aligned to an event with an offset')
    nwbfile.add_trial_column(name='laser_end', description=f'Laser end is aligned to an event')
    nwbfile.add_trial_column(name='laser_end_offset', description=f'Laser end is aligned to an event with an offset')
    nwbfile.add_trial_column(name='laser_protocol', description=f'The laser waveform')
    nwbfile.add_trial_column(name='laser_frequency', description=f'The laser waveform frequency')
    nwbfile.add_trial_column(name='laser_rampingdown', description=f'The ramping down time of the laser')
    nwbfile.add_trial_column(name='laser_pulse_duration', description=f'The pulse duration for Pulse protocol')
    nwbfile.add_trial_column(name='session_wide_control', description=f'Control the optogenetics session wide (e.g. only turn on opto in half of the session)')
    nwbfile.add_trial_column(name='fraction_of_session', description=f'Turn on/off opto in a fraction of the session (related to session_wide_control)')
    nwbfile.add_trial_column(name='session_start_with', description=f'The session start with opto on or off (related to session_wide_control)')
    nwbfile.add_trial_column(name='session_alternation', description=f'Turn on/off opto in every other session (related to session_wide_control)')
    nwbfile.add_trial_column(name='minimum_opto_interval', description=f'Minimum interval between two optogenetics trials (number of trials)')

    # auto training parameters
    nwbfile.add_trial_column(name='auto_train_engaged', description=f'Whether the auto training is engaged')
    nwbfile.add_trial_column(name='auto_train_curriculum_name', description=f'The name of the auto training curriculum')
    nwbfile.add_trial_column(name='auto_train_curriculum_version', description=f'The version of the auto training curriculum')
    nwbfile.add_trial_column(name='auto_train_curriculum_schema_version', description=f'The schema version of the auto training curriculum')
    nwbfile.add_trial_column(name='auto_train_stage', description=f'The current stage of auto training')
    nwbfile.add_trial_column(name='auto_train_stage_overridden', description=f'Whether the auto training stage is overridden')
    
    # add lickspout position
    nwbfile.add_trial_column(name='lickspout_position_x', description=f'x position (um) of the lickspout position (left-right)')
    nwbfile.add_trial_column(name='lickspout_position_z', description=f'z position (um) of the lickspout position (up-down)')

    # determine lickspout keys based on stage position keys
    stage_positions = getattr(obj, 'B_StagePositions', [{}])
    if list(stage_positions[0].keys()) == ['x', 'y1', 'y2', 'z']:   # aind stage
        nwbfile.add_trial_column(name='lickspout_position_y1',
                                 description=f'y position (um) of the left lickspout position (forward-backward)')
        nwbfile.add_trial_column(name='lickspout_position_y2',
                                 description=f'y position (um) of the right lickspout position (forward-backward)')
    else:
        nwbfile.add_trial_column(name='lickspout_position_y',
                                 description=f'y position (um) of the lickspout position (forward-backward)')
    # add reward size
    nwbfile.add_trial_column(name='reward_size_left', description=f'Left reward size (uL)')
    nwbfile.add_trial_column(name='reward_size_right', description=f'Right reward size (uL)')

    ## start adding trials ##
    # to see if we have harp timestamps
    if not hasattr(obj, 'B_TrialEndTimeHarp'):
        Harp = ''
    elif obj.B_TrialEndTimeHarp == []: # for json file transferred from mat data
        Harp = ''
    else:
        Harp = 'Harp'
    for i in range(len(obj.B_TrialEndTime)):
        Sc = obj.B_SelectedCondition[i] # the optogenetics conditions
        if Sc == 0:
            LaserWavelengthC = np.nan
            LaserLocationC = 'None'
            Laser1Power = np.nan
            Laser2Power = np.nan
            LaserOnProbablityC = np.nan
            LaserDurationC = np.nan
            LaserConditionC = 'None'
            LaserConditionProC = np.nan
            LaserStartC = 'None'
            LaserStartOffsetC = np.nan
            LaserEndC = 'None'
            LaserEndOffsetC = np.nan
            LaserProtocolC = 'None'
            LaserFrequencyC = np.nan
            LaserRampingDownC = np.nan
            LaserPulseDurC = np.nan

        else:
            laser_color=_get_field(obj, field_list=[f'TP_Laser_{Sc}',f'TP_LaserColor_{Sc}'],index=i)
            if laser_color == 'Blue':
                LaserWavelengthC = float(473)
            elif laser_color == 'Red':
                LaserWavelengthC = float(647)
            elif laser_color == 'Green':
                LaserWavelengthC = float(547)
            LaserLocationC = str(getattr(obj, f'TP_Location_{Sc}')[i])
            Laser1Power=float(eval(_get_field(obj, field_list=[f'TP_Laser1_power_{Sc}',f'TP_LaserPowerLeft_{Sc}'],index=i,default='[np.nan,np.nan]'))[1])
            Laser2Power=float(eval(_get_field(obj, field_list=[f'TP_Laser2_power_{Sc}',f'TP_LaserPowerRight_{Sc}'],index=i,default='[np.nan,np.nan]'))[1]) 
            LaserOnProbablityC = float(getattr(obj, f'TP_Probability_{Sc}')[i])
            LaserDurationC = float(getattr(obj, f'TP_Duration_{Sc}')[i])
            LaserConditionC = str(getattr(obj, f'TP_Condition_{Sc}')[i])
            LaserConditionProC = float(getattr(obj, f'TP_ConditionP_{Sc}')[i])
            LaserStartC = str(getattr(obj, f'TP_LaserStart_{Sc}')[i])
            LaserStartOffsetC = float(getattr(obj, f'TP_OffsetStart_{Sc}')[i])
            LaserEndC = str(getattr(obj, f'TP_LaserEnd_{Sc}')[i])
            LaserEndOffsetC = float(getattr(obj, f'TP_OffsetEnd_{Sc}')[i])
            LaserProtocolC = str(getattr(obj, f'TP_Protocol_{Sc}')[i])
            LaserFrequencyC = float(getattr(obj, f'TP_Frequency_{Sc}')[i])
            LaserRampingDownC = float(getattr(obj, f'TP_RD_{Sc}')[i])
            LaserPulseDurC = float(getattr(obj, f'TP_PulseDur_{Sc}')[i])
         
        if Harp == '':
            goCue_start_time_t = getattr(obj, f'B_GoCueTime')[i]  # Use CPU time
        else:
            if hasattr(obj, f'B_GoCueTimeHarp'):
                goCue_start_time_t = getattr(obj, f'B_GoCueTimeHarp')[i]  # Use Harp time, old format
            else:
                goCue_start_time_t = getattr(obj, f'B_GoCueTimeSoundCard')[i]  # Use Harp time, new format

        trial_kwargs = {
        'start_time' : getattr(obj, f'B_TrialStartTime{Harp}')[i],
        'stop_time' : getattr(obj, f'B_TrialEndTime{Harp}')[i],
        'animal_response' : obj.B_AnimalResponseHistory[i],
        'rewarded_historyL' : obj.B_RewardedHistory[0][i],
        'rewarded_historyR' : obj.B_RewardedHistory[1][i],
        'reward_outcome_time' : obj.B_RewardOutcomeTime[i],
        'delay_start_time' : _get_field(obj, f'B_DelayStartTime{Harp}', index=i, default=np.nan),
        'goCue_start_time' : goCue_start_time_t,
        'bait_left' : obj.B_BaitHistory[0][i],
        'bait_right' : obj.B_BaitHistory[1][i],
        'base_reward_probability_sum' : float(obj.TP_BaseRewardSum[i]),
        'reward_probabilityL' : float(obj.B_RewardProHistory[0][i]),
        'reward_probabilityR' : float(obj.B_RewardProHistory[1][i]),
        'reward_random_number_left' : _get_field(obj, 'B_CurrentRewardProbRandomNumber', index=i, default=[np.nan] * 2)[
                                        0],
        'reward_random_number_right' : _get_field(obj, 'B_CurrentRewardProbRandomNumber', index=i, default=[np.nan] * 2)[
                                         1],
        'left_valve_open_time' : float(obj.TP_LeftValue[i]),
        'right_valve_open_time' : float(obj.TP_RightValue[i]),
        'block_beta' : float(obj.TP_BlockBeta[i]),
        'block_min' : float(obj.TP_BlockMin[i]),
        'block_max' : float(obj.TP_BlockMax[i]),
        'min_reward_each_block' : float(obj.TP_BlockMinReward[i]),
        'delay_beta' : float(obj.TP_DelayBeta[i]),
        'delay_min' : float(obj.TP_DelayMin[i]),
        'delay_max' : float(obj.TP_DelayMax[i]),
        'delay_duration' : obj.B_DelayHistory[i],
        'ITI_beta' : float(obj.TP_ITIBeta[i]),
        'ITI_min' : float(obj.TP_ITIMin[i]),
        'ITI_max' : float(obj.TP_ITIMax[i]),
        'ITI_duration' : obj.B_ITIHistory[i],
        'response_duration' : float(obj.TP_ResponseTime[i]),
        'reward_consumption_duration' : float(obj.TP_RewardConsumeTime[i]),
        'reward_delay' : float(_get_field(obj, 'TP_RewardDelay', index=i, default=0)),
        'auto_waterL' : obj.B_AutoWaterTrial[0][i] if type(obj.B_AutoWaterTrial[0]) is list else obj.B_AutoWaterTrial[
            i],  # Back-compatible with old autowater format
        'auto_waterR' : obj.B_AutoWaterTrial[1][i] if type(obj.B_AutoWaterTrial[0]) is list else obj.B_AutoWaterTrial[i],
        # optogenetics
        'laser_on_trial' : obj.B_LaserOnTrial[i],
        'laser_wavelength' : LaserWavelengthC,
        'laser_location' : LaserLocationC,
        'laser_1_power' : Laser1Power,
        'laser_2_power' : Laser2Power,
        'laser_on_probability' : LaserOnProbablityC,
        'laser_duration' : LaserDurationC,
        'laser_condition' : LaserConditionC,
        'laser_condition_probability' : LaserConditionProC,
        'laser_start' : LaserStartC,
        'laser_start_offset' : LaserStartOffsetC,
        'laser_end' : LaserEndC,
        'laser_end_offset' : LaserEndOffsetC,
        'laser_protocol' : LaserProtocolC,
        'laser_frequency' : LaserFrequencyC,
        'laser_rampingdown' : LaserRampingDownC,
        'laser_pulse_duration' : LaserPulseDurC,

        'session_wide_control' : _get_field(obj, 'TP_SessionWideControl', index=i, default='None'),
        'fraction_of_session' : float(_get_field(obj, 'TP_FractionOfSession', index=i, default=np.nan)),
        'session_start_with' : _get_field(obj, 'TP_SessionStartWith', index=i, default='None'),
        'session_alternation' : _get_field(obj, 'TP_SessionAlternating', index=i, default='None'),
        'minimum_opto_interval' : float(_get_field(obj, 'TP_MinOptoInterval', index=i, default=0)),

        # add all auto training parameters (eventually should be in session.json)
        'auto_train_engaged' : _get_field(obj, 'TP_auto_train_engaged', index=i, default='None'),
        'auto_train_curriculum_name' : _get_field(obj, 'TP_auto_train_curriculum_name', index=i, default='None'),
        'auto_train_curriculum_version' : _get_field(obj, 'TP_auto_train_curriculum_version', index=i, default='None'),
        'auto_train_curriculum_schema_version' : _get_field(obj, 'TP_auto_train_curriculum_schema_version', index=i,
                                                          default='None'),
        'auto_train_stage' : _get_field(obj, 'TP_auto_train_stage', index=i, default='None'),
        'auto_train_stage_overridden' : _get_field(obj, 'TP_auto_train_stage_overridden', index=i, default=np.nan),

        # reward size
        'reward_size_left' : float(_get_field(obj, 'TP_LeftValue_volume', index=i)),
        'reward_size_right' : float(_get_field(obj, 'TP_RightValue_volume', index=i))
        }

        # populate lick spouts with correct format depending if using newscale or aind stage
        stage_positions = getattr(obj, 'B_StagePositions', [])  # If obj doesn't have attr, skip if since i !< len([])
        if i < len(stage_positions):    # index is valid for stage position lengths
            trial_kwargs['lickspout_position_x'] = stage_positions[i].get('x', np.nan)  # nan default if keys are wrong
            trial_kwargs['lickspout_position_z'] = stage_positions[i].get('z', np.nan)  # nan default if keys are wrong
            if list(stage_positions[i].keys()) == ['x', 'y1', 'y2', 'z']:    # aind stage
                trial_kwargs['lickspout_position_y1'] = stage_positions[i]['y1']
                trial_kwargs['lickspout_position_y2'] = stage_positions[i]['y2']
            else:       # newscale stage
                trial_kwargs['lickspout_position_y'] = stage_positions[i].get('y', np.nan) # nan default if keys are wrong
        else:   # if i not valid index, populate values with nan for x, y, z
            trial_kwargs['lickspout_position_x'] = np.nan
            trial_kwargs['lickspout_position_y'] = np.nan
            trial_kwargs['lickspout_position_z'] = np.nan

        nwbfile.add_trial(**trial_kwargs)


    #######  Other time series  #######
    #left/right lick time; give left/right reward time
    B_LeftRewardDeliveryTime=_get_field(obj, f'B_LeftRewardDeliveryTime{Harp}',default=[np.nan])
    B_RightRewardDeliveryTime=_get_field(obj, f'B_RightRewardDeliveryTime{Harp}',default=[np.nan])
    B_LeftLickTime=_get_field(obj, 'B_LeftLickTime',default=[np.nan])
    B_RightLickTime=_get_field(obj, 'B_RightLickTime',default=[np.nan])
    B_PhotometryFallingTimeHarp=_get_field(obj, 'B_PhotometryFallingTimeHarp',default=[np.nan])
    B_PhotometryRisingTimeHarp=_get_field(obj, 'B_PhotometryRisingTimeHarp',default=[np.nan])

    LeftRewardDeliveryTime = TimeSeries(
        name="left_reward_delivery_time",
        unit="second",
        timestamps=B_LeftRewardDeliveryTime,
        data=np.ones(len(B_LeftRewardDeliveryTime)).tolist(),
        description='The reward delivery time of the left lick port'
    )
    nwbfile.add_acquisition(LeftRewardDeliveryTime)

    RightRewardDeliveryTime = TimeSeries(
        name="right_reward_delivery_time",
        unit="second",
        timestamps=B_RightRewardDeliveryTime,
        data=np.ones(len(B_RightRewardDeliveryTime)).tolist(),
        description='The reward delivery time of the right lick port'
    )
    nwbfile.add_acquisition(RightRewardDeliveryTime)

    LeftLickTime = TimeSeries(
        name="left_lick_time",
        unit="second",
        timestamps=B_LeftLickTime,
        data=np.ones(len(B_LeftLickTime)).tolist(),
        description='The time of left licks'
    )
    nwbfile.add_acquisition(LeftLickTime)

    RightLickTime = TimeSeries(
        name="right_lick_time",
        unit="second",
        timestamps=B_RightLickTime,
        data=np.ones(len(B_RightLickTime)).tolist(),
        description='The time of left licks'
    )
    nwbfile.add_acquisition(RightLickTime)

    # Add photometry time stamps
    PhotometryFallingTimeHarp = TimeSeries(
        name="FIP_falling_time",
        unit="second",
        timestamps=B_PhotometryFallingTimeHarp,
        data=np.ones(len(B_PhotometryFallingTimeHarp)).tolist(),
        description='The time of photometry falling edge (from Harp)'
    )
    nwbfile.add_acquisition(PhotometryFallingTimeHarp)

    PhotometryRisingTimeHarp = TimeSeries(
        name="FIP_rising_time",
        unit="second",
        timestamps=B_PhotometryRisingTimeHarp,
        data=np.ones(len(B_PhotometryRisingTimeHarp)).tolist(),
        description='The time of photometry rising edge (from Harp)'
    )
    nwbfile.add_acquisition(PhotometryRisingTimeHarp)
    
    # Add optogenetics time stamps
    ''' 
    There are two sources of optogenetics time stamps depending on which event it is aligned to. 
    The first source is the optogenetics time stamps aligned to the trial start time (from the 
    DO0 stored in B_TrialStartTimeHarp), and the second source is the optogenetics time stamps aligned to other events 
    (e.g go cue and reward outcome; from the DO3 stored in B_OptogeneticsTimeHarp).
    '''
    start_time=np.array(_get_field(obj, f'B_TrialStartTime{Harp}'))
    LaserStart=[]
    for i in range(len(obj.B_TrialEndTime)):
        Sc = obj.B_SelectedCondition[i] # the optogenetics conditions
        if Sc == 0:
            LaserStart.append('None')
            continue
        LaserStart.append(str(getattr(obj, f'TP_LaserStart_{Sc}')[i]))
    OptogeneticsTimeHarp_ITI_Stimulation=start_time[np.array(LaserStart) == 'Trial start'].tolist()
    OptogeneticsTimeHarp_other=_get_field(obj, 'B_OptogeneticsTimeHarp',default=[np.nan])
    B_OptogeneticsTimeHarp=OptogeneticsTimeHarp_ITI_Stimulation+OptogeneticsTimeHarp_other
    B_OptogeneticsTimeHarp.sort()
    OptogeneticsTimeHarp = TimeSeries(
        name="optogenetics_time",
        unit="second",
        timestamps=B_OptogeneticsTimeHarp,
        data=np.ones(len(B_OptogeneticsTimeHarp)).tolist(),
        description='Optogenetics start time (from Harp)'
    )
    nwbfile.add_acquisition(OptogeneticsTimeHarp)
    # save NWB file
    base_filename = os.path.splitext(os.path.basename(fname))[0] + '.nwb'
    if len(nwbfile.trials) > 0:
        if save_file is not None: 
            NWBName = save_file
            io = NWBHDF5IO(NWBName, mode="w")
            io.write(nwbfile)
            io.close()
            logger.info(f'Successfully converted: {NWBName}')
            print(f'NWB saved {NWBName}')
            return 'success', nwbfile
    else:
        logger.warning(f"No trials found! Skipping {fname}")
        return 'empty_trials', nwbfile

def parseSessionID(file_name):
    if len(re.split('[_.]', file_name)[0]) == 6 & re.split('[_.]', file_name)[0].isdigit():
        aniID = re.split('[_.]', file_name)[0]
        date = re.split('[_.]', file_name)[1]
        dateObj = datetime.strptime(date, "%Y-%m-%d")
        raw_id= session_id
    elif re.split('[_.]', file_name)[0] == 'behavior' or re.split('[_.]', file_name)[0] == 'ecephys':
        aniID = re.split('[_.]', file_name)[1]
        date = re.split('[_.]', file_name)[2]+'_'+re.split('[_.]', file_name)[3]
        dateObj = datetime.strptime(date, "%Y-%m-%d_%H-%M-%S")
        raw_id = '_'.join(re.split('[_.]', file_name)[1:])
    else:
        aniID = None
        dateObj = None
        raw_id = None
    
    return aniID, dateObj, raw_id

def get_stream_info(directory):
    """
    Get information about the streams in an Open Ephys data directory

    Parameters
    ----------
    directory : str
        The path to the Open Ephys data directory

    Returns
    -------
    pd.DataFrame
        A DataFrame with information about the streams

    """
    from open_ephys.analysis import Session
    session = Session(directory)

    stream_info = {
        "Record Node": [],
        "Rec Idx": [],
        "Exp Idx": [],
        "Stream": [],
        "Duration (s)": [],
        "Channels": [],
    }

    for recordnode in session.recordnodes:
        current_record_node = os.path.basename(recordnode.directory).split(
            "Record Node "
        )[1]

        for recording in recordnode.recordings:
            current_experiment_index = recording.experiment_index
            current_recording_index = recording.recording_index

            for stream in recording.continuous:
                sample_rate = stream.metadata["sample_rate"]
                data_shape = stream.samples.shape
                channel_count = data_shape[1]
                duration = data_shape[0] / sample_rate

                stream_info["Record Node"].append(current_record_node)
                stream_info["Rec Idx"].append(current_recording_index)
                stream_info["Exp Idx"].append(current_experiment_index)
                stream_info["Stream"].append(stream.metadata["stream_name"])
                stream_info["Duration (s)"].append(duration)
                stream_info["Channels"].append(channel_count)

    return pd.DataFrame(data=stream_info)

def session_dirs(session_id, model_name = None, data_dir = None, scratch_dir = None, hopkins = False):
    # set up directories if not provided
    if data_dir is None:
        data_dir = capsule_directories()['data_dir']
    if scratch_dir is None:
        scratch_dir = capsule_directories()['derived_dir']
    # parse session_id 

    aniID, date_obj, raw_id = parseSessionID(session_id)
    hopkins_list = ['672850', '669492', '669489', '754898', '754896', '754895', '749624',  '749472', '701707', '699472', '699461', '699462']
    # print(f"Parsing session_id: {session_id}, aniID: {aniID}")
    if aniID.startswith('ZS') or aniID in hopkins_list:
        # print('Old data, using hopkins formats')
        return session_dirs_hopkins(session_id, model_name, data_dir = data_dir, scratch_dir = scratch_dir)
    # raw dirs
    raw_dir = os.path.join(data_dir, session_id+'_raw_data')
    session_dir = os.path.join(raw_dir, 'ecephys_clipped')
    session_dir_raw = os.path.join(raw_dir, 'ecephys_compressed')
    if not os.path.exists(session_dir):
        session_dir = os.path.join(raw_dir, 'ecephys', 'ecephys_clipped')
        session_dir_raw = os.path.join(raw_dir, 'ecephys', 'ecephys_compressed')
    sorted_dir = os.path.join(data_dir, session_id+'_sorted_curated')
    sorted_raw_dir = os.path.join(data_dir, session_id+'_sorted')
    # nwb files
    nwb_dir_raw = None
    nwb_dir_curated = None
    # load recordings and decide which one is longer
    stream_name = None
    raw_recording_dir = None 
    experiment_id = None
    seg_id = None   
    max_len_ind = None
    all_rec_count = None
    stream_name_surface = None
    surface_exp_id = None
    surface_seg_id = None
    surface_recording_dir = None
    all_rec_count_surface = None
    if os.path.exists(session_dir):
        stream_info = get_stream_info(session_dir)
        stream_info = stream_info[(stream_info['Stream'].str.contains('ProbeA')) & (~stream_info['Stream'].str.contains('LFP'))]
        if len(stream_info)==1:
            # if there is only one recording, use it
            experiment = [f for f in os.listdir(session_dir_raw) if 'ProbeA' in f and 'LFP' not in f][0]
            rec_path = os.path.join(session_dir_raw, experiment)
            experiment_id = 1
            seg_id = 1
            # print(f'Single experiment found: experiment{experiment_id}, recording{seg_id}')
            stream_name = experiment.split('.zarr')[0] + '_recording1'
            all_rec_count = experiment_id-1 + seg_id-1
            raw_recording_dir = os.path.join(session_dir_raw, experiment)
            surface_exp_id = None
            surface_seg_id = None
            surface_recording_dir = None
            all_rec_count_surface = None
        else:
            # loop through experiments and segments to get the longest one
            lengths = []
            experiment_ids = []
            segment_inds = []
            stream_names = []

            for experiment in os.listdir(session_dir_raw):
                if 'ProbeA' in experiment: 
                    rec_path = os.path.join(session_dir_raw, experiment)
                    # print(rec_path)
                    exp_match = re.search(r'experiment(\d+)', experiment)
                    # experiment_inds.append(int(match.group(1)))
                    recording_curr = si.load(rec_path)
                    rec_num = recording_curr.get_num_segments()
                    # print(rec_num)
                    for rec_ind in range(rec_num):
                        lengths.append(recording_curr.get_num_samples(segment_index = rec_ind))
                        experiment_ids.append(int(exp_match.group(1)))
                        segment_inds.append(rec_ind)
                        stream_names.append(experiment.split('.zarr')[0])

            max_len_ind = np.argmax(np.array(lengths))
            experiment_id = experiment_ids[max_len_ind]
            seg_id = segment_inds[max_len_ind]+1
            # print(f'Selected experiment{experiment_id} recording{seg_id}, length:{lengths[max_len_ind]/32000 :.2f}')
            experiment_name = stream_names[max_len_ind]
            stream_name = experiment_name + f'_recording{seg_id}'
            all_rec_count = experiment_id-1 + seg_id-1
            raw_recording_dir = os.path.join(session_dir_raw, experiment_name+'.zarr')
            # check if there is a surface recording
            if np.sum(np.array(experiment_ids)>experiment_id) > 0:
                surface_inds = np.where(np.array(experiment_ids)>experiment_id)[0]
                surface_ind = surface_inds[np.argmax(np.array(lengths)[surface_inds])]
                surface_exp_id = experiment_ids[surface_ind]
                surface_seg_id = segment_inds[surface_ind]+1
                surface_experiment_name = stream_names[surface_ind]
                stream_name_surface = surface_experiment_name + f'_recording{surface_seg_id}'
                surface_recording_dir = os.path.join(session_dir_raw, surface_experiment_name+'.zarr')
                all_rec_count_surface = surface_exp_id-1 + surface_seg_id-1
                 
    # else:
    #     print(f'No raw session directory found for {session_id}.')

    # raw version
    nwb_dir_temp = os.path.join(sorted_raw_dir, 'nwb')
    nwb_dir_raw = None
    if os.path.exists(nwb_dir_temp):
        nwbs = [nwb for nwb in os.listdir(nwb_dir_temp) if nwb.endswith('.nwb')]
        nwb = [nwb for nwb in nwbs if (f'experiment{experiment_id}' in nwb) and (f'recording{seg_id}' in nwb)]
        if len(nwb) == 0:
            nwb = [nwb for nwb in nwbs if (f'experiment{experiment_id}' in nwb)]
        if len(nwb) == 1:
            nwb_dir_raw = os.path.join(nwb_dir_temp, nwb[0])
        elif len(nwb) > 1:
            # print('There are multiple recordings in the raw nwb directory. Picked one with units.')
            nwb_dir_raw = None
        else:
            nwb_dir_raw = None
            # print('There is no nwb file in the raw directory.')
    nwb_dir_curated = None
    # curated version
    if os.path.exists(sorted_dir):
        nwb_dir_temp = os.path.join(sorted_dir, 'nwb')
        if not os.path.exists(nwb_dir_temp):
            nwb_dir_temp = [path for path in os.listdir(sorted_dir) if path.endswith('.nwb')]
            if len(nwb_dir_temp) == 1:
                nwb_dir_temp = os.path.join(sorted_dir, nwb_dir_temp[0])
            elif len(nwb_dir_temp) > 1:
                nwb_dir_temp = os.path.join(sorted_dir, nwb_dir_temp[0])
                # print('There are multiple nwb files in the curated directory. Picked first one.')
            else:
                nwb_dir_temp = None
                nwb_dir_curated = None
                # print('There is no nwb file in the curated directory.')

        if nwb_dir_temp is not None:
            nwbs = [nwb for nwb in os.listdir(nwb_dir_temp) if nwb.endswith('.nwb')]
            nwb = [nwb for nwb in nwbs if (f'experiment{experiment_id}' in nwb) and (f'recording{seg_id}' in nwb)]
            if len(nwb) == 0:
                nwb = [nwb for nwb in nwbs if (f'experiment{experiment_id}' in nwb)]
            if len(nwb) == 1:
                nwb_dir_curated = os.path.join(nwb_dir_temp, nwb[0])
            elif len(nwb) > 1:
                # print('There are multiple recordings in the curated nwb directory. Picked one with units.')
                nwb_dir_curated = None
            else:
                nwb_dir_curated = None
                # print('There is no nwb file in the curated directory.')
    # postprocessed dirs
    postprocessed_dir_raw = None
    postprocessed_dir_curated = None

    if os.path.exists(sorted_dir):
        postprocessed_dir_temp = os.path.join(sorted_dir, 'postprocessed')
        if os.path.exists(postprocessed_dir_temp):
            postprocessed_sub_folders = os.listdir(postprocessed_dir_temp)
            postprocessed_sub_folder = [s for s in postprocessed_sub_folders if ('post' not in s) and (stream_name in s)]
            if len(postprocessed_sub_folder) == 0:
                postprocessed_sub_folder = [s for s in postprocessed_sub_folders if 'post' not in s and stream_name[:-1] in s]
            if len(postprocessed_sub_folder) > 0:
                postprocessed_dir_curated = os.path.join(postprocessed_dir_temp, postprocessed_sub_folder[0])
            else:
                postprocessed_dir_curated = None
                # print('No postprocessed dir found in curated sorted dir')

    if os.path.exists(sorted_raw_dir):
        postprocessed_dir_temp = os.path.join(sorted_raw_dir, 'postprocessed')
        if os.path.exists(postprocessed_dir_temp):
            postprocessed_sub_folders = os.listdir(postprocessed_dir_temp)
            postprocessed_sub_folder = [s for s in postprocessed_sub_folders if 'post' not in s and stream_name in s]
            if len(postprocessed_sub_folder) == 0:
                postprocessed_sub_folder = [s for s in postprocessed_sub_folders if 'post' not in s and stream_name[:-1] in s]
            if len(postprocessed_sub_folder) > 0:
                postprocessed_dir_raw = os.path.join(postprocessed_dir_temp, postprocessed_sub_folder[0])
            else:
                postprocessed_dir_raw = None
                # print('No postprocessed dir found in raw sorted dir')


    
    # curated dirs
    curated_dir_raw = None
    curated_dir_curated = None
    
    if os.path.exists(sorted_dir):
        curated_dir_temp = os.path.join(sorted_dir, 'curated')
        if os.path.exists(curated_dir_temp):
            curated_sub_folders = os.listdir(curated_dir_temp)
            curated_sub_folders = [s for s in curated_sub_folders if stream_name in s]
            if len(curated_sub_folders) == 0:
                curated_sub_folders = [s for s in os.listdir(curated_dir_temp) if stream_name[:-1] in s]
            if len(curated_sub_folders) > 0:
                curated_dir_curated = os.path.join(curated_dir_temp, curated_sub_folders[0])
            else:
                curated_dir_curated = None
                # print('No curated dir found in curated sorted dir') 
    
    if os.path.exists(sorted_raw_dir):
        curated_dir_temp = os.path.join(sorted_raw_dir, 'curated')
        if os.path.exists(curated_dir_temp):
            curated_sub_folders = os.listdir(curated_dir_temp)
            curated_sub_folders = [s for s in curated_sub_folders if stream_name in s]
            if len(curated_sub_folders) == 0:
                curated_sub_folders = [s for s in os.listdir(curated_dir_temp) if stream_name[:-1] in s]
            if len(curated_sub_folders) > 0:
                curated_dir_raw = os.path.join(curated_dir_temp, curated_sub_folders[0])
            else:
                curated_dir_raw = None
                # print('No curated dir found in raw sorted dir')

    # model dir
    models_dir = os.path.join(data_dir, f'{aniID}_model_stan')
    if model_name is not None:
        model_dir = os.path.join(models_dir, model_name)
        model_file = os.path.join(model_dir, session_id+'_session_model_dv.csv')
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, raw_id+'_session_model_dv.csv')
        session_curation_file = os.path.join(models_dir, f'{aniID}_session_data.csv')
    else:
        model_dir = None
        model_file = None
        session_curation_file = None
    
    # opto dirs
    opto_csv_dir = os.path.join(raw_dir, 'ecephys_clipped')
    opto_csvs = None
    if not os.path.exists(opto_csv_dir):
        opto_csv_dir = os.path.join(raw_dir, 'ecephys', 'ecephys_clipped')

    if os.path.exists(opto_csv_dir):
        temp_files = os.listdir(opto_csv_dir)
        opto_csvs = [os.path.join(opto_csv_dir, s) for s in temp_files if '.opto.csv' in s]

    # FP dir
    fp_dir = os.path.join(raw_dir, 'fib')
    photo_nwb_dir = None

    # processed dirs
    processed_dir = os.path.join(scratch_dir, aniID, session_id)
    alignment_dir = os.path.join(processed_dir, 'alignment')
    beh_fig_dir = os.path.join(processed_dir, 'behavior')
    ephys_dir = os.path.join(processed_dir, 'ephys')
    ephys_dir_raw = os.path.join(processed_dir, 'ephys', 'raw')
    ephys_processed_dir_raw = os.path.join(ephys_dir_raw, 'processed')
    ephys_fig_dir_raw = os.path.join(ephys_dir_raw, 'figures')
    ephys_dir_curated = os.path.join(processed_dir, 'ephys', 'curated')
    ephys_processed_dir_curated = os.path.join(ephys_dir_curated, 'processed')
    ephys_fig_dir_curated = os.path.join(ephys_dir_curated, 'figures')
    opto_dir = os.path.join(ephys_dir, 'opto')
    opto_dir_raw = os.path.join(opto_dir, 'raw')
    opto_dir_curated = os.path.join(opto_dir, 'curated')
    opto_dir_fig_raw = os.path.join(opto_dir, 'raw', 'figures')
    opto_dir_fig_curated = os.path.join(opto_dir, 'curated', 'figures')
    # pro   
    
    beh_nwb_dir = os.path.join(processed_dir, 'behavior', f'{session_id}.nwb')

    dir_dict = {'aniID': aniID,
                'raw_id': raw_id,
                'datetime': date_obj,
                'raw_dir': raw_dir,
                'raw_rec': raw_recording_dir,
                'session_dir': session_dir,
                'session_dir_raw': session_dir_raw,
                'processed_dir': processed_dir,
                'alignment_dir': alignment_dir,
                'beh_fig_dir': beh_fig_dir,
                'ephys_dir_raw': ephys_dir_raw,
                'ephys_processed_dir_raw': ephys_processed_dir_raw,
                'ephys_fig_dir_raw': ephys_fig_dir_raw,
                'ephys_dir_curated': ephys_dir_curated,
                'ephys_processed_dir_curated': ephys_processed_dir_curated,
                'ephys_fig_dir_curated': ephys_fig_dir_curated,
                'opto_dir': opto_dir,
                'opto_dir_raw': opto_dir_raw,
                'opto_dir_curated': opto_dir_curated,
                'opto_dir_fig_raw': opto_dir_fig_raw,
                'opto_dir_fig_curated': opto_dir_fig_curated,
                'nwb_dir_raw': nwb_dir_raw,
                'nwb_dir_curated': nwb_dir_curated,
                'postprocessed_dir_raw': postprocessed_dir_raw,
                'postprocessed_dir_curated': postprocessed_dir_curated,
                'curated_dir_raw': curated_dir_raw,
                'curated_dir_curated': curated_dir_curated,
                'model_dir': model_dir,
                'model_file': model_file,
                'session_curation_file': session_curation_file,
                'nwb_beh': beh_nwb_dir,
                'sorted_dir_curated': sorted_dir,
                'sorted_dir_raw': sorted_raw_dir,
                'opto_csvs': opto_csvs,
                'stream_name': stream_name,
                'stream_name_surface': stream_name_surface,
                'experiment_id': experiment_id,
                'seg_id': seg_id,
                'rec_id_all': all_rec_count,
                'surface_exp_id': surface_exp_id,
                'surface_seg_id': surface_seg_id,
                'surface_rec': surface_recording_dir,
                'surface_rec_id_all': all_rec_count_surface,
                'photometry_data_dir': fp_dir,
                'fp_nwb_dir': photo_nwb_dir}

    # make directories
    makedirs(dir_dict)

    return dir_dict

def session_dirs_hopkins(session_id, model_name = None, data_dir = None, scratch_dir = None):
    if data_dir is None:
        data_dir = capsule_directories()['data_dir']
    if scratch_dir is None:
        scratch_dir = capsule_directories()['derived_dir']
    # parse session_id
    aniID, date_obj, raw_id = parseSessionID(session_id)
    # raw dirs
    raw_dir = os.path.join(data_dir, session_id+'_raw_data')
    session_dir = os.path.join(raw_dir, 'ecephys', 'neuralynx', 'session')
    session_dir_raw = session_dir
    mat_dir = os.path.join(raw_dir, 'ecephys', 'sorted', 'session')
    if not os.path.exists(mat_dir):
        mat_dir = os.path.join(raw_dir, 'fib', 'sorted', 'session')
    if os.path.exists(mat_dir):
        beh_mat = [file for file in os.listdir(mat_dir) if file.endswith('sessionData_behav.mat')]
        if len(beh_mat) > 0:
            beh_mat = os.path.join(mat_dir, beh_mat[0])
        else:
            beh_mat = None
        ephys_mat = [file for file in os.listdir(mat_dir) if file.endswith('sessionData_nL.mat')]
        if len(ephys_mat) > 0:
            ephys_mat = os.path.join(mat_dir, ephys_mat[0])
        else:
            ephys_mat = None
    else:
        beh_mat = None
        ephys_mat = None
    

    if not os.path.exists(session_dir):
        session_dir = None
        session_dir_raw = None
    sorted_dir = os.path.join(raw_dir, 'ecephys', 'sorted')
    sorted_raw_dir = os.path.join(raw_dir, 'ecephys', 'sorted')
    # nwb files
    # find raw recording
    if session_dir is not None:
        raw_recording_dir = os.path.join(session_dir, 'raw_data.hdf5')
        if not os.path.exists(raw_recording_dir):
            # print(f'No raw session directory found for {session_id}.')
            stream_name = None
            raw_recording_dir = None 
            experiment_id = None
            seg_id = None   
            max_len_ind = None
            all_rec_count = None
    else:
        stream_name = None
        raw_recording_dir = None 
        experiment_id = None
        seg_id = None   
        max_len_ind = None
        all_rec_count = None       
    # raw version
    nwb_dir_temp = raw_dir
    nwb_dir_raw = None
    if os.path.exists(nwb_dir_temp):
        nwbs = [nwb for nwb in os.listdir(nwb_dir_temp) if nwb.endswith('.nwb.zarr')]
        nwb = nwbs
        if len(nwb) == 1:
            nwb_dir_raw = os.path.join(nwb_dir_temp, nwb[0])
        elif len(nwb) > 1:
            # print('There are multiple recordings in the raw nwb directory. Picked one with units.')
            nwb_dir_raw = None
        else:
            nwb_dir_raw = None
            # print('There is no nwb file in the raw directory.')
    nwb_dir_curated = None
    # curated version
    if os.path.exists(raw_dir):
        nwb_dir_temp = raw_dir
        if not os.path.exists(nwb_dir_temp):
            nwb_dir_temp = [path for path in os.listdir(sorted_dir) if path.endswith('.nwb.zarr')]
            if len(nwb_dir_temp) == 1:
                nwb_dir_temp = os.path.join(sorted_dir, nwb_dir_temp[0])
            elif len(nwb_dir_temp) > 1:
                nwb_dir_temp = os.path.join(sorted_dir, nwb_dir_temp[0])
                # print('There are multiple nwb files in the curated directory. Picked first one.')
            else:
                nwb_dir_temp = None
                nwb_dir_curated = None
                # print('There is no nwb file in the curated directory.')

        if nwb_dir_temp is not None:
            nwbs = [nwb for nwb in os.listdir(nwb_dir_temp) if nwb.endswith('.nwb.zarr')]
            nwb = nwbs
            if len(nwb) == 1:
                nwb_dir_curated = os.path.join(nwb_dir_temp, nwb[0])
            elif len(nwb) > 1:
                # print('There are multiple recordings in the curated nwb directory. Picked one with units.')
                nwb_dir_curated = None
            else:
                nwb_dir_curated = None
                # print('There is no nwb file in the curated directory.')
    # postprocessed dirs
    processed_dir = os.path.join(scratch_dir, aniID, session_id)
    postprocessed_dir_raw = os.path.join(processed_dir, 'ephys', 'curated', 'analyzer.zarr')
    postprocessed_dir_curated = os.path.join(processed_dir, 'ephys', 'curated', 'analyzer.zarr')

    # if os.path.exists(sorted_dir):
    #     postprocessed_dir_temp = os.path.join(sorted_dir, 'postprocessed')
    #     if os.path.exists(postprocessed_dir_temp):
    #         postprocessed_sub_folders = os.listdir(postprocessed_dir_temp)
    #         postprocessed_sub_folder = [s for s in postprocessed_sub_folders if ('post' not in s) and (stream_name in s)]
    #         postprocessed_dir_curated = os.path.join(postprocessed_dir_temp, postprocessed_sub_folder[0])

    # if os.path.exists(sorted_raw_dir):
    #     postprocessed_dir_temp = os.path.join(sorted_raw_dir, 'postprocessed')
    #     if os.path.exists(postprocessed_dir_temp):
    #         postprocessed_sub_folders = os.listdir(postprocessed_dir_temp)
    #         postprocessed_sub_folder = [s for s in postprocessed_sub_folders if 'post' not in s and stream_name in s]
    #         postprocessed_dir_raw = os.path.join(postprocessed_dir_temp, postprocessed_sub_folder[0])

    
    # curated dirs
    curated_dir_raw = os.path.join(sorted_raw_dir, 'session')
    curated_dir_curated = os.path.join(sorted_raw_dir, 'session')
    
    # if os.path.exists(sorted_dir):
    #     curated_dir_temp = os.path.join(sorted_dir, 'curated')
    #     if os.path.exists(curated_dir_temp):
    #         curated_sub_folders = os.listdir(curated_dir_temp)
    #         curated_sub_folders = [s for s in curated_sub_folders if stream_name in s]
    #         curated_dir_curated = os.path.join(curated_dir_temp, curated_sub_folders[0])    
    
    # if os.path.exists(sorted_raw_dir):
    #     curated_dir_temp = os.path.join(sorted_raw_dir, 'curated')
    #     if os.path.exists(curated_dir_temp):
    #         curated_sub_folders = os.listdir(curated_dir_temp)
    #         curated_sub_folders = [s for s in curated_sub_folders if stream_name in s]
    #         curated_dir_raw = os.path.join(curated_dir_temp, curated_sub_folders[0])

    # model dir
    models_dir = os.path.join(data_dir, f'{aniID}_model_stan')
    if model_name is not None:
        model_dir = os.path.join(models_dir, model_name)
        model_file = os.path.join(model_dir, session_id+'_session_model_dv.csv')
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, raw_id+'_session_model_dv.csv')
        session_curation_file = os.path.join(models_dir, f'{aniID}_session_data.csv')
    else:
        model_dir = None
        model_file = None
        session_curation_file = None
    
    # opto dirs
    opto_csv_dir = os.path.join(raw_dir, 'ecephys_clipped')
    opto_csvs = None
    # if not os.path.exists(opto_csv_dir):
    #     opto_csv_dir = os.path.join(raw_dir, 'ecephys', 'ecephys_clipped')

    # if os.path.exists(opto_csv_dir):
    #     temp_files = os.listdir(opto_csv_dir)
    #     opto_csvs = [os.path.join(opto_csv_dir, s) for s in temp_files if '.opto.csv' in s]
    # photometry dir
    photometry_data_dir = os.path.join(raw_dir, 'fib')
    # processed dirs
    processed_dir = os.path.join(scratch_dir, aniID, session_id)
    alignment_dir = os.path.join(processed_dir, 'alignment')
    beh_fig_dir = os.path.join(processed_dir, 'behavior')
    ephys_dir = os.path.join(processed_dir, 'ephys')
    ephys_dir_raw = os.path.join(processed_dir, 'ephys', 'raw')
    ephys_processed_dir_raw = os.path.join(ephys_dir_raw, 'processed')
    ephys_fig_dir_raw = os.path.join(ephys_dir_raw, 'figures')
    ephys_dir_curated = os.path.join(processed_dir, 'ephys', 'curated')
    ephys_processed_dir_curated = os.path.join(ephys_dir_curated, 'processed')
    ephys_fig_dir_curated = os.path.join(ephys_dir_curated, 'figures')
    opto_dir = os.path.join(ephys_dir, 'opto')
    opto_dir_raw = os.path.join(opto_dir, 'raw')
    opto_dir_curated = os.path.join(opto_dir, 'curated')
    opto_dir_fig_raw = os.path.join(opto_dir, 'raw', 'figures')
    opto_dir_fig_curated = os.path.join(opto_dir, 'curated', 'figures')
    # pro   
    
    beh_nwb_dir = os.path.join(processed_dir, 'behavior', f'{session_id}.nwb')

    # photometry nwb
    photometry_nwb_files = [f for f in os.listdir(raw_dir) if f.endswith('nwb.zarr')]
    if len(photometry_nwb_files) > 0:
        photometry_nwb_dir = os.path.join(raw_dir, photometry_nwb_files[0])
    else:
        photometry_nwb_dir = None

    dir_dict = {'aniID': aniID,
                'raw_id': raw_id,
                'datetime': date_obj,
                'raw_dir': raw_dir,
                'raw_rec': raw_recording_dir,
                'session_dir': session_dir,
                'session_dir_raw': session_dir_raw,
                'processed_dir': processed_dir,
                'alignment_dir': alignment_dir,
                'beh_fig_dir': beh_fig_dir,
                'ephys_dir_raw': ephys_dir_raw,
                'ephys_processed_dir_raw': ephys_processed_dir_raw,
                'ephys_fig_dir_raw': ephys_fig_dir_raw,
                'ephys_dir_curated': ephys_dir_curated,
                'ephys_processed_dir_curated': ephys_processed_dir_curated,
                'ephys_fig_dir_curated': ephys_fig_dir_curated,
                'opto_dir': opto_dir,
                'opto_dir_raw': opto_dir_raw,
                'opto_dir_curated': opto_dir_curated,
                'opto_dir_fig_raw': opto_dir_fig_raw,
                'opto_dir_fig_curated': opto_dir_fig_curated,
                'nwb_dir_raw': nwb_dir_raw,
                'nwb_dir_curated': nwb_dir_curated,
                'postprocessed_dir_raw': postprocessed_dir_raw,
                'postprocessed_dir_curated': postprocessed_dir_curated,
                'curated_dir_raw': curated_dir_raw,
                'curated_dir_curated': curated_dir_curated,
                'model_dir': model_dir,
                'model_file': model_file,
                'session_curation_file': session_curation_file,
                'nwb_beh': beh_nwb_dir,
                'sorted_dir_curated': sorted_dir,
                'sorted_dir_raw': sorted_raw_dir,
                'opto_csvs': opto_csvs,
                'stream_name': None,
                'experiment_id': 1,
                'seg_id': 1,
                'rec_id_all': 0,
                'beh_mat': beh_mat,
                'ephys_mat': ephys_mat,
                'fp_nwb_dir': photometry_nwb_dir,
                'photometry_data_dir': photometry_data_dir}

    # make directories
    makedirs(dir_dict)

    return dir_dict

def load_model_dv(session_id, model_name, data_dir = None, scratch_dir = None):
    if data_dir is None:
        data_dir = capsule_directories()['data_dir']
    if scratch_dir is None:
        scratch_dir = capsule_directories()['derived_dir']
    model_dirs = session_dirs(session_id, model_name, data_dir=data_dir, scratch_dir=scratch_dir)
    if model_dirs['model_file'] is not None and os.path.exists(model_dirs['model_file']):
        model_dv_temp = pd.read_csv(model_dirs['model_file'], index_col=0)
        # trial_df = get_session_tbl(session_id)
        # model_dv = pd.DataFrame(np.nan, index=range(len(trial_df)), columns=model_dv_temp.columns)
        session_curation = pd.read_csv(model_dirs['session_curation_file'])
        session_cut = session_curation.loc[session_curation['session_id'].str.contains(model_dirs['raw_id']), 'session_cut'].values[0]
        session_cut = ast.literal_eval(session_cut)
        # model_dv[curr_cut[0]:curr_cut[1]] = model_dv_temp
        model_dv = model_dv_temp.copy()
    else:
        model_dv = None
        session_cut = None
    return model_dv, session_cut

def makedirs(directories):
    for directory_name, directory in directories.items():
        if 'dir' in directory_name and directory is not None and 'scratch' in directory and 'data' not in directory:
            if not os.path.exists(directory):
                os.makedirs(directory)

def merge_pdfs(input_dir, output_filename='merged.pdf'):
    merger = PdfMerger()
    files = os.listdir(input_dir)
    files = sorted(files)
    # Iterate through all PDF files in the input directory
    for i, filename in enumerate(files):
        if filename.endswith('.pdf'):
            if i%50 == 0:
                print(f'Merging file {i} out of {len(os.listdir(input_dir))}')
            filepath = os.path.join(input_dir, filename)
            merger.append(filepath)

    # Write the merged PDF to the output file
    with open(output_filename, 'wb') as output_file:
        merger.write(output_file)

    print(f"PDF files in '{input_dir}' merged into '{output_filename}' successfully.")

    
def delete_files_without_name(folder_path, name):
    # Iterate through all files in the folder
    for i, filename in enumerate(os.listdir(folder_path)):
        # Check if the filename does not contain 'combined'
        if i%50 == 0:
            print(f'Deleting file {i} out of {len(os.listdir(folder_path))}')
        if name not in filename:
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Check if the path is a file
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)

def makeSessionDF(session, cut = [0, np.nan], model_name = None, cut_interruptions = False, load_glm = False):
    capsule_dirs = capsule_directories()
    session_dir = session_dirs(session)
    tblTrials = get_session_tbl(session)

    if model_name is not None:
        model_dv, cut = load_model_dv(session, model_name)
    if cut is not None:
        if np.isnan(cut[1]):
            tblTrials = tblTrials.iloc[cut[0]:].copy()
        else:
            tblTrials = tblTrials.iloc[cut[0]:cut[1]].copy()
        
    # tblTrials.reset_index(inplace=True)
    trialStarts = tblTrials.loc[tblTrials['animal_response']!=2, 'goCue_start_time'].values
    responseTimes = tblTrials[tblTrials['animal_response']!=2]
    responseTimes = responseTimes['reward_outcome_time'].values

    # responseInds
    responseInds = tblTrials['animal_response']!=2
    leftRewards = tblTrials.loc[responseInds, 'rewarded_historyL']

    # oucome
    leftRewards = tblTrials.loc[tblTrials['animal_response']!=2, 'rewarded_historyL']
    rightRewards = tblTrials.loc[tblTrials['animal_response']!=2, 'rewarded_historyR']
    outcomes = leftRewards | rightRewards
    outcomePrev = np.concatenate((np.full((1), np.nan), outcomes[:-1]))

    # choices
    choices = tblTrials.loc[tblTrials['animal_response']!=2, 'animal_response'] == 1
    choicesPrev = np.concatenate((np.full((1), np.nan), choices[:-1]))

    # go_cue_time
    go_cue_time = tblTrials.loc[tblTrials['animal_response']!=2, 'goCue_start_time']

    # choice_time
    choice_time = tblTrials.loc[tblTrials['animal_response']!=2, 'reward_outcome_time']

    # outcome_time
    outcome_time = tblTrials.loc[tblTrials['animal_response']!=2, 'reward_outcome_time'] + np.unique(tblTrials[~tblTrials['reward_delay'].isnull()]['reward_delay'])[0]

    
    # laser

    laserChoice = tblTrials.loc[tblTrials['animal_response']!=2, 'laser_on_trial'] == 1
    laser = tblTrials['laser_on_trial'] == 1
    laserPrev = np.concatenate((np.full((1), np.nan), laserChoice[:-1]))
    # trial_id = tblTrials.loc[tblTrials['animal_response']!=2, 'id']

    # stay vs switch
    svs = np.zeros(len(choices), dtype=bool)
    svs[choices!=choicesPrev] = True
    svs[0] = False

    # time since last trial
    time_between_trials = np.diff(tblTrials['goCue_start_time'], prepend = np.nan)
    time_between_trials = time_between_trials[responseInds]
    # create dataframe

    trialData = pd.DataFrame({
        # 'trial_id': trial_id,
        'outcome': outcomes.values.astype(float), 
        'choice': choices.values.astype(float),
        'laser': laserChoice.values.astype(float),
        'outcome_prev': outcomePrev,
        'laser_prev': laserPrev,
        'choices_prev': choicesPrev,
        'go_cue_time': go_cue_time.values,
        'choice_time': choice_time.values,
        'outcome_time': outcome_time.values,
        'svs': svs,
        'time_between_trials': time_between_trials,
        })
    # combine all columns of trialData and tblTrials
    choice_tbl = tblTrials.loc[tblTrials['animal_response']!=2].copy()
    choice_tbl.reset_index(inplace=True)
    trialData = pd.concat([trialData, choice_tbl], axis=1)
    if model_name is not None and load_glm:
        # load glm results if exist 
        glm_path = os.path.join(f"{capsule_dirs['derived_dir']}/{session_dir['aniID']}/ani_combinded", 'glm_choice_history_model_results.pkl')
        if os.path.exists(glm_path):
            with open(glm_path, 'rb') as f:
                glm_results = pickle.load(f)
            p_choice = glm_results['choice_prob_by_session'][session]
            session_interruptions = trialData['auto_manual_trial'] | trialData['extra_reward']
            longest_start, longest_end, longest_len = longest_zero_chunk(session_interruptions.tolist())
            trialData['pChoice_glm'] = np.nan
            trialData.loc[longest_start:longest_end, 'pChoice_glm'] = p_choice

            p_left_glm = p_choice.copy()
            p_right_glm = p_choice.copy()
            choice_cut = trialData['choice'].values.copy()[longest_start:longest_end+1]
            p_left_glm[choice_cut == 1] = 1 - p_left_glm[choice_cut == 1]
            p_right_glm[choice_cut == 0] = 1 - p_right_glm[choice_cut == 0]
            p_choice_updated = np.full(len(p_choice), np.nan)
            for t in range(len(p_choice)-1):
                if choice_cut[t] == 1:
                    p_choice_updated[t] = p_right_glm[t+1]
                else:
                    p_choice_updated[t] = p_left_glm[t+1]
            trialData['policy_glm_change'] = np.nan
            trialData.loc[longest_start:longest_end, 'policy_glm_change'] = p_choice_updated - p_choice
            trialData['policy_glm_change_log_odd'] = np.nan
            trialData.loc[longest_start:longest_end, 'policy_glm_change_log_odd'] = np.log(p_choice_updated / (1-p_choice_updated) ) - np.log(p_choice / (1-p_choice))
            svs_cut = trialData['svs'].values.copy()[longest_start:longest_end+1]
            policy_update_mean = (np.concatenate([
                                        1 - svs_cut[1:],
                                        [np.nan]
                                    ])
                                    - p_choice)
            trialData['policy_glm_update_mean'] = np.nan
            trialData.loc[longest_start:longest_end, 'policy_glm_update_mean'] = policy_update_mean

    if model_name is not None and model_dv is not None:
        session_df = pd.merge(trialData, model_dv, left_index=True, right_index=True, suffixes=('', '_model'))

        for column in session_df.columns:
            if '_model' in column:
                session_df.drop(column, axis=1, inplace=True)
        if 'Q_r' in session_df.columns:
            Qchosen = session_df['Q_l'].values.copy()
            Qchosen[np.where(session_df['choice']>0)] = session_df.loc[session_df['choice']>0, 'Q_r'].values.copy()
            session_df['Qchosen'] = Qchosen
        # policy update 
        p_choice = session_df['pChoice'].values
        p_choice_L = p_choice.copy()
        p_choice_L[session_df['choice'].values == 1] = 1-p_choice_L[session_df['choice'].values == 1]
        p_choice_R = p_choice.copy()
        p_choice_R[session_df['choice'].values == 0] = 1-p_choice_R[session_df['choice'].values == 0]
        policy_pre = session_df['pChoice'].values.copy()
        policy_post = np.full(len(policy_pre), np.nan)
        for t in range(len(policy_pre)-1):
            if session_df['choice'].values[t] == 1:
                policy_post[t] = p_choice_R[t+1]
            else:
                policy_post[t] = p_choice_L[t+1]
        session_df['policy_pre'] = policy_pre
        session_df['policy_post'] = policy_post
        session_df['policy_change'] = session_df['policy_post'] - session_df['policy_pre']
        session_df['policy_change_log_odd'] = np.log(policy_post / (1-policy_post) ) - np.log(policy_pre / (1-policy_pre))
        policy_update_mean = (
                                np.concatenate([
                                    1 - session_df['svs'].values[1:],
                                    [np.nan]
                                ])
                                - session_df['pChoice'].values
                            )
        session_df['policy_update_mean'] = policy_update_mean
        trialData = session_df.copy()
    
    if cut_interruptions:
        session_interruptions = trialData['auto_manual_trial'] | trialData['extra_reward']
        longest_start, longest_end, longest_len = longest_zero_chunk(session_interruptions.tolist())
        trialData = trialData.iloc[longest_start:longest_end+1]
        

    return trialData

def get_history_from_nwb(nwb, cut = [0, np.nan]):
    """Get choice and reward history from nwb file"""
    if not isinstance(nwb, pd.DataFrame):       
        df_trial = nwb.trials.to_dataframe()
    else:
        df_trial = nwb.copy()
    if np.isnan(cut[1]):
        df_trial = df_trial.iloc[cut[0]:].copy()
    else:
        df_trial = df_trial.iloc[cut[0]:cut[1]].copy()

    autowater_offered = (df_trial.auto_waterL == 1) | (df_trial.auto_waterR == 1)
    # autowater_offered[df_trial.auto_waterL == 1] = -1
    choice_history = df_trial.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
    reward_history = df_trial.rewarded_historyL | df_trial.rewarded_historyR
    p_reward = [
        df_trial.reward_probabilityL.values,
        df_trial.reward_probabilityR.values,
    ]

    trial_time = df_trial['goCue_start_time'].values
    return (
        choice_history,
        reward_history,
        p_reward,
        autowater_offered,
        trial_time,
    )

def plot_session_in_time_all(nwb, bin_size = 10, in_time = True, ax_ori = None):
    if ax_ori is None:
        fig, _ = plt.subplots(figsize=(12, 6))
        fig = plt.Figure(figsize=(12, 6))
        gs = GridSpec(3, 1, figure = fig, height_ratios=[6,1,1], hspace = 0.5)
    else:
        gs = GridSpecFromSubplotSpec(3, 1, height_ratios=[6,1,1], hspace = 0.5, subplot_spec = ax_ori.get_subplotspec())
    choice_history, reward_history, p_reward, autowater_offered, trial_time = get_history_from_nwb(nwb)
    ax_choice_reward = fig.add_subplot(gs[0])
    if in_time:
        plot_foraging_session(  # noqa: C901
                            choice_history,
                            reward_history,
                            p_reward = p_reward,
                            autowater_offered = autowater_offered,
                            trial_time = trial_time,
                            ax = ax_choice_reward,
                            )           
    else:
        plot_foraging_session(  # noqa: C901
                            choice_history,
                            reward_history,
                            p_reward = p_reward,
                            autowater_offered = autowater_offered,
                            ax = ax_choice_reward,
                            )
    # turn off x and y axis
    ax_choice_reward.set_frame_on(False)
    # plot licks
    data = load_data(nwb)   
    ax = fig.add_subplot(gs[2]) 
    # ax = plt.subplot(gs[2])
    bins = np.arange(np.min(data['all_licks']-data['tbl_trials']['goCue_start_time'][0]), np.max(data['all_licks']-data['tbl_trials']['goCue_start_time'][0]), bin_size)  
    ax.hist(data['left_licks']-data['tbl_trials']['goCue_start_time'][0], bins = bins, color = 'blue', alpha = 0.5, label = 'left licks', density = True)
    ax.set_xlim(0, -data['tbl_trials']['goCue_start_time'][0]+data['tbl_trials']['goCue_start_time'].values[-1])
    ax.legend(loc = 'upper right')
    ax.set_frame_on(False)
    ax.set_xlabel('Time in session (s)')
    ax = fig.add_subplot(gs[1])
    ax.hist(data['right_licks']-data['tbl_trials']['goCue_start_time'][0], bins = bins, color = 'red', alpha = 0.5, label = 'right licks', density = True)
    ax.set_xlim(0, -data['tbl_trials']['goCue_start_time'][0]+data['tbl_trials']['goCue_start_time'].values[-1])
    ax.legend(loc = 'upper right')
    ax.set_frame_on(False)
    plt.tight_layout()
    if ax_ori is None:
        return fig
    else:
        return ax_ori

def plot_session_glm(session, tMax = 10, cut = [0, np.nan], model_name = None):
    tbl = makeSessionDF(session, cut = cut, model_name=model_name)
    allChoices = 2 * (tbl['choice'].values - 0.5)
    allRewards = allChoices * tbl['outcome'].values
    allNoRewards = allChoices * (1 - tbl['outcome'].values)
    allChoice_R = tbl['choice'].values

    # Creating rwdMatx
    rwdMatx = []
    for i in range(1, tMax + 1):
        rwdMatx.append(np.concatenate([np.full(i, np.nan), allRewards[:len(allRewards) - i]]))

    rwdMatx = np.array(rwdMatx)

    # Creating noRwdMatx
    noRwdMatx = []
    for i in range(1, tMax + 1):
        noRwdMatx.append(np.concatenate([np.full(i, np.nan), allNoRewards[:len(allNoRewards) - i]]))

    noRwdMatx = np.array(noRwdMatx)

    # Combining rwdMatx and noRwdMatx
    X = np.vstack([rwdMatx, noRwdMatx]).T

    # Remove rows with NaN values
    valid_idx = ~np.isnan(X).any(axis=1)
    X = X[valid_idx]
    y = allChoice_R[valid_idx]

    # Adding a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fitting the GLM model
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Logit()))
    glm_result = glm_binom.fit()

    # R-squared calculation (pseudo R-squared)
    rsq = glm_result.pseudo_rsquared(kind="cs")

    # Coefficients and confidence intervals
    coef_vals = glm_result.params[1:tMax + 1]
    ci_bands = glm_result.conf_int()[1:tMax + 1]
    error_l = np.abs(coef_vals - ci_bands[:, 0])
    error_u = np.abs(coef_vals - ci_bands[:, 1])

    fig = plt.figure(figsize=(8, 6))
    # Plotting reward coefficients
    plt.errorbar(np.arange(1, tMax + 1) + 0.2, coef_vals, yerr=[error_l, error_u], fmt='o', color='c', linewidth=2, label='Reward')
    plt.plot(np.arange(1, tMax + 1) + 0.2, coef_vals, 'c-', linewidth=1)

    # Coefficients and confidence intervals for no reward
    coef_vals_no_rwd = glm_result.params[tMax + 1:]
    ci_bands_no_rwd = glm_result.conf_int()[tMax + 1:]
    error_l_no_rwd = np.abs(coef_vals_no_rwd - ci_bands_no_rwd[:, 0])
    error_u_no_rwd = np.abs(coef_vals_no_rwd - ci_bands_no_rwd[:, 1])

    # Plotting no reward coefficients
    plt.errorbar(np.arange(1, tMax + 1) + 0.2, coef_vals_no_rwd, yerr=[error_l_no_rwd, error_u_no_rwd], fmt='o', color='m', linewidth=2, label='No Reward')
    plt.plot(np.arange(1, tMax + 1) + 0.2, coef_vals_no_rwd, 'm-', linewidth=1)

    # Labels and legend
    plt.xlabel('Outcome n Trials Back')
    plt.ylabel('β Coefficient')
    plt.xlim([0.5, tMax + 0.5])
    plt.axhline(0, color='k', linestyle='--')

    # Adding R-squared and intercept information in the legend
    intercept_info = f'R² = {rsq:.2f} | Int: {glm_result.params[0]:.2f}'
    plt.legend(loc='upper right')

    coeffs_dict = {'coeff_reward': coef_vals.tolist(),
                   'ci-bands_reward': ci_bands.tolist(),
                   'coeff_no-reward': coef_vals_no_rwd.tolist(),
                   'ci-bands_no-reward': ci_bands_no_rwd.tolist(),
                   'rsq': rsq,
                   'intercept': glm_result.params[0],
                   }

    return fig, session, coeffs_dict

def longest_zero_chunk(binary_list):
    longest_len = 0
    longest_start = None
    longest_end = None

    # group consecutive values (e.g., [0,0,1,0,0,0,1] -> [(0,2), (1,1), (0,3), (1,1)])
    for value, group in itertools.groupby(enumerate(binary_list), key=lambda x: x[1]):
        if value == 0:
            indices = [idx for idx, _ in group]
            length = len(indices)
            if length > longest_len:
                longest_len = length
                longest_start = indices[0]
                longest_end = indices[-1]

    return longest_start, longest_end, longest_len

def get_session_tbl(session, cut_interruptions = False, load_raw = False):
    session_dir = session_dirs(session)
    if load_raw:
        nwb_file = os.path.join(session_dir['beh_fig_dir'], session + '_all_trials.nwb')
    else:
        nwb_file = os.path.join(session_dir['beh_fig_dir'], session + '.nwb')
    if os.path.exists(nwb_file):
        nwb = load_nwb_from_filename(nwb_file)
        tbl = nwb.trials.to_dataframe()
        tbl['lick_lat'] = tbl['reward_outcome_time'] - tbl['goCue_start_time']
        tbl['trial_ind'] = tbl.index
        right_rewards = nwb.acquisition['right_reward_delivery_time'].timestamps[:]
        left_rewards = nwb.acquisition['left_reward_delivery_time'].timestamps[:]
        # In all sessions, find reward times before reward_outcome_times
        right_rewards_times = [None] * len(tbl)
        left_rewards_times = [None] * len(tbl)

        for i, row in tbl.iterrows():
            right_rewards_trial = right_rewards[(right_rewards <= row['reward_outcome_time'] + row['reward_delay']+0.1) & (right_rewards > row['goCue_start_time'])]
            left_rewards_trial = left_rewards[(left_rewards <= row['reward_outcome_time'] + row['reward_delay']+0.1) & (left_rewards > row['goCue_start_time'])]
            if len(right_rewards_trial) > 0:
                right_rewards_times[i] = right_rewards_trial-row['goCue_start_time']
            if len(left_rewards_trial) > 0:
                left_rewards_times[i] = left_rewards_trial-row['goCue_start_time']
        tbl['right_reward_times'] = right_rewards_times
        tbl['left_reward_times'] = left_rewards_times
        tbl['choice_time_trial'] = tbl['reward_outcome_time'] - tbl['goCue_start_time']
        auto_manual_trials = [
            (
                (tbl['right_reward_times'].values[i] is not None and 
                np.any(tbl['right_reward_times'].values[i] < tbl['choice_time_trial'].values[i]))
                or
                (tbl['left_reward_times'].values[i] is not None and 
                np.any(tbl['left_reward_times'].values[i] < tbl['choice_time_trial'].values[i]))
            )
            for i in range(len(tbl))
        ]

        extra_reward = [
            (
                (tbl['right_reward_times'].values[i] is not None and 
                ~tbl['rewarded_historyR'].values[i])
                or
                (tbl['left_reward_times'].values[i] is not None and 
                ~tbl['rewarded_historyL'].values[i])
            )
            for i in range(len(tbl))

        ]
        tbl['auto_manual_trial'] = auto_manual_trials
        tbl['extra_reward'] = extra_reward
        if cut_interruptions:
            session_interruptions = tbl['auto_manual_trial'] | tbl['extra_reward']
            longest_start, longest_end, longest_len = longest_zero_chunk(session_interruptions.tolist())
            tbl = tbl.iloc[longest_start:longest_end+1]
    else:
        tbl = None
    return tbl

def get_unit_tbl(session, data_type, summary = True):
    session_dir = session_dirs(session)
    unit_tbl_summary = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_{data_type}_soma_opto_tagging_summary.pkl')
    # if 'ZS' in session:
    #     unit_tbl_summary = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_{data_type}_soma_opto_tagging_summary_raw_wf.pkl')
    unit_tbl_dir = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging_metrics.pkl')
    ccf_dir = os.path.join(session_dir[f'ephys_processed_dir_{data_type}'],'ccf_unit_locations.csv')
    if summary: 
        if os.path.exists(unit_tbl_summary):
            with open(unit_tbl_summary, 'rb') as f:
                  unit_data = pickle.load(f)
            unit_tbl = unit_data
            if os.path.exists(ccf_dir):
                ccf_data = pd.read_csv(ccf_dir)
                unit_tbl.drop(columns=['x_ccf', 'y_ccf', 'z_ccf'], inplace=True, errors='ignore')
                unit_tbl = pd.merge(unit_tbl, ccf_data, left_on='unit_id', right_on='unit_id', how='left')
        elif os.path.exists(unit_tbl_dir):
            with open(unit_tbl_dir, 'rb') as f:
                unit_data = pickle.load(f)
            unit_tbl = unit_data['opto_tagging_df']
        else:
            print(f'No unit table found for {session} in {data_type} data.')
            return None
        return unit_tbl
    elif os.path.exists(unit_tbl_dir):
        with open(unit_tbl_dir, 'rb') as f:
            unit_data = pickle.load(f)
        return unit_data['opto_tagging_df']
    else:
        print(f'No unit table found for {session} in {data_type} data.')
        return None

def transfer_nwb(session_id, hopkins=False, CSplus_only=True):
    session_dir = session_dirs(session_id, hopkins=hopkins)
    src_path = session_dir['nwb_dir_raw']
    with NWBZarrIO(src_path, mode="r") as io:
        src_nwb = io.read()
        
    # Get the trials DynamicTable
    trials_table = src_nwb.trials

    # --- Create a new NWBFile ---
    session_description = "CS_plus only copy" if CSplus_only else "All trials copy"
    new_nwb = NWBFile(
        session_description=session_description,
        identifier=src_nwb.identifier,
        session_start_time=src_nwb.session_start_time
    )
    new_nwb.session_id = session_id

    # Add all custom trial columns to new NWB with their original description
    for col in trials_table.colnames:
        if col not in ("start_time", "stop_time", "tags", "timeseries"):  # reserved
            desc = trials_table[col].description if hasattr(trials_table[col], "description") else ""
            new_nwb.add_trial_column(name=col, description=desc)
    new_nwb.add_trial_column(name='ITI_duration', description='Fake values to hack some analysis')
    new_nwb.add_trial_column(name='delay_max', description='Max no lick window')
    new_nwb.add_trial_column(name='delay_min', description='Min no lick window')
    
    metadata = {
        # Meta
        'box': 'Hopkins 295F nlyx'}
    df_metadata = pd.DataFrame(metadata, index=[0])
    new_nwb.add_scratch(df_metadata, 
                        name="metadata",
                        description="Some important session-wise meta data")
    # Add trials one by one
    # Convert trials table to DataFrame
    trials_df = trials_table.to_dataframe()
    pre_row = None

    if src_nwb.units is not None:
        session_df, lick_L, lick_R =load_df_from_mat(session_dir['ephys_mat'])
    else:
        session_df, lick_L, lick_R = load_df_from_mat(session_dir['beh_mat'])
    
    choice_side = np.full(len(session_df), 2)  # 0 for left, 1 for right, 2 for no response
    choice_side[~session_df['rewardL'].isna()]  = 0  # left choice
    choice_side[~session_df['rewardR'].isna()]  = 1  # right choice
    if len(session_df) != len(trials_df):
        print(f'Warning: The number of trials in the session ({len(session_df)}) does not match the number of trials in the NWB file ({len(trials_df)}).')
        raise ValueError("Mismatch in number of trials between session and NWB file.")
        
    for ind, row in trials_df.iterrows():
        if CSplus_only:
            if row['trial_type'] != 'CSplus':
                continue
            kwargs = row.to_dict()
            kwargs['ITI_duration'] = 0.5
            if ind > 0 and pre_row is not None:
                kwargs['start_time'] = pre_row['goCue_start_time'] + 3.1
            else:
                kwargs['start_time'] = row['goCue_start_time'] - 1.5
            kwargs['stop_time'] = row['goCue_start_time'] + 3.1
            kwargs['delay_max'] = kwargs['delay_duration']
            kwargs['delay_min'] = kwargs['delay_duration']
            kwargs['animal_response'] = choice_side[ind]
            new_nwb.add_trial(**kwargs)
            pre_row = row
        else:
            kwargs = row.to_dict()
            kwargs['ITI_duration'] = 0.5
            if ind > 0 and pre_row is not None:
                kwargs['start_time'] = pre_row['goCue_start_time'] + 3.1
            else:
                kwargs['start_time'] = row['goCue_start_time'] - 1.5
            kwargs['stop_time'] = row['goCue_start_time'] + 3.1
            kwargs['delay_max'] = kwargs['delay_duration']
            kwargs['delay_min'] = kwargs['delay_duration']
            kwargs['animal_response'] = choice_side[ind]
            new_nwb.add_trial(**kwargs)
            pre_row = row
    # add acquisition 
    from pynwb import TimeSeries

    def _materialize_array_like(value):
        """Load lazy datasets into memory to avoid external zarr links in output."""
        if value is None:
            return None
        try:
            return np.asarray(value[:])
        except Exception:
            return np.asarray(value)

    for name, acq in src_nwb.acquisition.items():
        if isinstance(acq, TimeSeries):
            new_acq_kwargs = {
                'name': acq.name,
                'data': _materialize_array_like(acq.data),
                'unit': acq.unit,
                'description': acq.description,
            }
            if acq.timestamps is not None:
                new_acq_kwargs['timestamps'] = _materialize_array_like(acq.timestamps)
            else:
                new_acq_kwargs['starting_time'] = acq.starting_time
                new_acq_kwargs['rate'] = acq.rate

            new_acq = TimeSeries(**new_acq_kwargs)
            new_nwb.add_acquisition(new_acq)
        else:
            new_nwb.add_acquisition(acq)  # fallback


    # for name, acq in src_nwb.acquisition.items():
    #     new_nwb.add_acquisition(acq)

    # --- Write the new NWB (Zarr) ---
    out_path = session_dir['nwb_beh']
    if not CSplus_only:
        base, ext = os.path.splitext(out_path)
        out_path = base + '_all_trials' + ext
    shutil.rmtree(out_path, ignore_errors=True)
    with NWBZarrIO(out_path, mode="w") as out_io:
        out_io.write(new_nwb)


# concatenate all session data for the certain list of sessions
def fit_glm_session_list(session_list, max_lag=5, plot=False):
    max_lag = 5
    all_reward_side_hist_mat = []
    all_noreward_side_hist_mat = []
    all_choice_hist_mat = []
    all_choices = []
    all_choices_prev = []
    session_lens = []
    session_list_valid = []
    for sess in session_list:
        if get_session_tbl(sess) is not None:
            sess_df = makeSessionDF(sess, cut_interruptions=True, model_name='stan_qLearning_5params')
            choice = 2*(sess_df['choice'].values.copy() - 0.5)
            all_choices.append(sess_df['choice'].values)
            all_choices_prev.append(sess_df['choices_prev'].values)
            outcome = sess_df['outcome'].values
            reward_side = choice * outcome
            noreward_side = choice * (1 - outcome)
            reward_side_hist_mat = np.full((len(choice), max_lag), np.nan)
            noreward_side_hist_mat = np.full((len(choice), max_lag), np.nan)
            choice_hist_mat = np.full((len(choice), max_lag), np.nan)

            for lag in range(1, max_lag+1):
                reward_side_hist_mat[lag:, lag-1] = reward_side[:-lag]
                noreward_side_hist_mat[lag:, lag-1] = noreward_side[:-lag]
                choice_hist_mat[lag:, lag-1] = choice[:-lag]

            all_reward_side_hist_mat.append(reward_side_hist_mat)
            all_noreward_side_hist_mat.append(noreward_side_hist_mat)
            all_choice_hist_mat.append(choice_hist_mat)
            session_lens.append(len(choice))
            session_list_valid.append(sess)
    if len(all_choices) == 0:
        raise ValueError("No valid sessions found in the provided session list.")
        return None, None
    all_reward_side_hist_mat = np.vstack(all_reward_side_hist_mat)
    all_noreward_side_hist_mat = np.vstack(all_noreward_side_hist_mat)
    all_choice_hist_mat = np.vstack(all_choice_hist_mat)
    all_choices = np.hstack(all_choices)
    all_choices_prev = np.hstack(all_choices_prev)
    # fit logistic regression model
    # reward side and choice history as predictors
    X = np.hstack([all_reward_side_hist_mat, all_choice_hist_mat])
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(all_choices, X, missing='drop')
    result_rwd_choice = model.fit()

    # reward side and no-reward side history as predictors
    X = np.hstack([all_reward_side_hist_mat, all_noreward_side_hist_mat])
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(all_choices, X, missing='drop')
    result_rwd_norwd = model.fit()

    # predict switch and stay probability, all regressors become interaction with previousl choice
    # reward side and no reward side history as predictors
    all_reward_side_hist_mat_inter = all_reward_side_hist_mat * all_choices_prev[:, np.newaxis]
    all_noreward_side_hist_mat_inter = all_noreward_side_hist_mat * all_choices_prev[:, np.newaxis]
    all_choices_switch = (all_choices != all_choices_prev).astype(float)
    # filter out where all_choices_prev is nan
    valid_idx = ~np.isnan(all_choices_prev)
    X = np.hstack([all_reward_side_hist_mat_inter[valid_idx], all_noreward_side_hist_mat_inter[valid_idx]])
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(all_choices_switch[valid_idx], X, missing='drop')
    result_switch_rwd_norwd = model.fit()

    # predict switch and stay probability, all regressors become interaction with previousl choice, with no reward side and choice history as predictors
    # all_reward_side_hist_mat_inter = all_reward_side_hist_mat * all_choices_prev[:, np.newaxis]
    all_noreward_side_hist_mat_inter = all_noreward_side_hist_mat * all_choices_prev[:, np.newaxis]
    all_choice_hist_mat_inter = all_choice_hist_mat * all_choices_prev[:, np.newaxis]
    all_choices_switch = (all_choices != all_choices_prev).astype(float)
    # filter out where all_choices_prev is nan
    valid_idx = ~np.isnan(all_choices_prev)
    X = np.hstack([all_noreward_side_hist_mat_inter[valid_idx], all_choice_hist_mat_inter[valid_idx]])
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(all_choices_switch[valid_idx], X, missing='drop')
    result_switch_norwd_choice = model.fit()

    # predict switch probability, all regressors become interaction with previousl choice, with reward side and choice history as predictors
    all_reward_side_hist_mat_inter = all_reward_side_hist_mat * all_choices_prev[:, np.newaxis]
    all_choice_hist_mat_inter = all_choice_hist_mat * all_choices_prev[:, np.newaxis]
    all_choices_switch = (all_choices != all_choices_prev).astype(float)
    # filter out where all_choices_prev is nan
    valid_idx = ~np.isnan(all_choices_prev)
    X = np.hstack([all_reward_side_hist_mat_inter[valid_idx], all_choice_hist_mat_inter[valid_idx]])
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(all_choices_switch[valid_idx], X, missing='drop')
    result_switch_rwd_choice = model.fit()


    # make predictions of choice probability using the first model
    X_pred = np.hstack([all_reward_side_hist_mat, all_choice_hist_mat])
    X_pred = sm.add_constant(X_pred, has_constant='add')
    choice_prob = result_rwd_choice.predict(X_pred)
    # flip choice probability for left choices
    left_choices = all_choices == 0
    choice_prob[left_choices] = 1 - choice_prob[left_choices]

    # make prediction of switch probability using the switch model
    valid_idx = ~np.isnan(all_choices_prev)
    X_pred_switch = np.hstack([all_noreward_side_hist_mat_inter[valid_idx], all_choice_hist_mat_inter[valid_idx]])
    X_pred_switch = sm.add_constant(X_pred_switch, has_constant='add')
    switch_prob = result_switch_norwd_choice.predict(X_pred_switch)
    # fill back the switch probability into the original array
    switch_prob_full = np.full(len(all_choices), np.nan)
    switch_prob_full[valid_idx] = switch_prob
    switch_prob = switch_prob_full


    # create a dictionary to hold results
    # cut choice_prob into sessions
    choice_prob_sess = []
    switch_prob_sess = []
    start_idx = 0
    for sess_len in session_lens:
        choice_prob_sess.append(choice_prob[start_idx:start_idx+sess_len])
        switch_prob_sess.append(switch_prob[start_idx:start_idx+sess_len])
        start_idx += sess_len

    results = {
        'reward_side_and_choice_history_model': result_rwd_choice,
        'reward_side_and_noreward_side_history_model': result_rwd_norwd,
        'switch_model_reward_noreward': result_switch_rwd_norwd,
        'switch_model_noreward_choice': result_switch_norwd_choice,
        'sitch_model_reward_choice': result_switch_rwd_choice,
        'choice_prob_by_session': {session: choice_prob_sess[idx] for idx, session in enumerate(session_list_valid)},
        'switch_prob_by_session': {session: switch_prob_sess[idx] for idx, session in enumerate(session_list_valid)}
    }

    # plot
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        ax = axes[0, 0]

        coef = result_rwd_choice.params[1:max_lag+1]
        err = result_rwd_choice.bse[1:max_lag+1]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Rewarded Side History Coefficients')

        ax = axes[0, 1]
        coef = result_rwd_choice.params[max_lag+1:]
        err = result_rwd_choice.bse[max_lag+1:]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Choice History Coefficients')

        ax = axes[1, 0]
        coef = result_rwd_norwd.params[1:max_lag+1]
        err = result_rwd_norwd.bse[1:max_lag+1]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Rewarded Side History Coefficients')
        ax = axes[1, 1]
        coef = result_rwd_norwd.params[max_lag+1:]
        err = result_rwd_norwd.bse[max_lag+1:]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')

        ax.set_ylabel('Coefficient')
        ax.set_title('No-Rewarded Side History Coefficients')
        plt.tight_layout()

        fig, axes = plt.subplots(3, 2, figsize=(10, 5))
        # first row: switch model with reward and no reward side history as predictors
        ax = axes[0, 0]
        coef = result_switch_rwd_norwd.params[1:max_lag+1]
        err = result_switch_rwd_norwd.bse[1:max_lag+1]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Switch Model - Reward Side History Coefficients')

        ax = axes[0, 1]
        coef = result_switch_rwd_norwd.params[max_lag+1:]
        err = result_switch_rwd_norwd.bse[max_lag+1:]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Switch Model - No-Reward Side History Coefficients')

        # second row: switch model with no reward side history and choice history as predictors
        ax = axes[1, 0]
        coef = result_switch_norwd_choice.params[1:max_lag+1]
        err = result_switch_norwd_choice.bse[1:max_lag+1]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Switch Model - No-Reward Side History Coefficients')

        ax = axes[1, 1]
        coef = result_switch_norwd_choice.params[max_lag+1:]
        err = result_switch_norwd_choice.bse[max_lag+1:]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Switch Model - Choice History Coefficients')
        # third row: switch model with reward side history and choice history as predictors
        ax = axes[2, 0]
        coef = result_switch_rwd_choice.params[1:max_lag+1]
        err = result_switch_rwd_choice.bse[1:max_lag+1]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Switch Model - Reward Side History Coefficients')
        ax = axes[2, 1]
        coef = result_switch_rwd_choice.params[max_lag+1:]
        err = result_switch_rwd_choice.bse[max_lag+1:]
        lags = np.arange(1, max_lag+1)
        ax.errorbar(lags, coef, yerr=err, fmt='-o', capsize=5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.set_xlabel('Lag (trials)')
        ax.set_ylabel('Coefficient')
        ax.set_title('Switch Model - Choice History Coefficients')
        plt.tight_layout()

    else:
        fig = None
    return results, fig

def fit_glm_animal(ani_focus, max_lag=5, plot=False):
    capsule_dirs = capsule_directories()
    dfs = [pd.read_csv(CAPSULE_ROOT + '/code/data_management/session_assets.csv'),
            pd.read_csv(CAPSULE_ROOT + '/code/data_management/hopkins_session_assets.csv'),
            pd.read_csv(CAPSULE_ROOT + '/code/data_management/hopkins_FP_session_assets.csv')]
    df = pd.concat(dfs)
    session_list = df['session_id'].values.tolist()
    ani_list = [str(session).split('_')[1] for session in session_list if str(session).startswith('behavior')]
    session_list = [session for session in session_list if str(session).startswith('behavior')]
    ani_session_df = pd.DataFrame({'animal': ani_list, 'session_id': session_list})

    ani_session_list = ani_session_df[ani_session_df['animal'] == ani_focus]['session_id'].values.tolist()
    results, fig = fit_glm_session_list(ani_session_list, max_lag=max_lag, plot=plot)
    if results is None:
        return None, None
    save_dir = os.path.join(capsule_dirs['derived_dir'], f'{ani_focus}/ani_combinded')
    if 'data' not in save_dir: # to avoid saving in the data directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(save_dir, 'glm_choice_history_model_results.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(results, f)
        if plot:
            fig.savefig(fname=os.path.join(save_dir, f'{ani_focus}_glm_choice_history_model_results.pdf'))
    return results, fig


def _build_hit_glm_session_df(session, t_Max, t_Max_hist):
    """Build the per-session design matrix for the hit/miss GLM.

    Returns a DataFrame with lagged reward and hit-history columns plus a
    binary `hit` column, ready to be concatenated across sessions.
    """
    session_tbl = get_session_tbl(session)

    hit = session_tbl['animal_response'].values != 2
    rewards = (
        (session_tbl['rewarded_historyL'].values == 1) |
        (session_tbl['rewarded_historyR'].values == 1)
    )

    hit_mtx = np.full((t_Max_hist, len(hit)), np.nan, dtype=float)
    for i in range(t_Max_hist):
        hit_mtx[i, i + 1:] = hit[:len(hit) - (i + 1)]

    rwd_mtx = np.full((t_Max, len(hit)), np.nan, dtype=float)
    for i in range(t_Max):
        rwd_mtx[i, i + 1:] = rewards[:len(hit) - (i + 1)]

    col_list = [f'rwd_{i+1}' for i in range(t_Max)]
    col_list += [f'hit_{i+1}' for i in range(t_Max_hist)]

    matrix = np.concatenate((rwd_mtx, hit_mtx), axis=0)
    reg_df = pd.DataFrame(matrix.T, columns=col_list)
    reg_df['hit'] = hit
    return reg_df


def hit_glm_all_trial(session_list, t_Max, t_Max_hist=0, n_jobs=-1, backend='loky', verbose=0):
    """Fit a population-level logistic GLM predicting hit vs. miss from reward history.

    Parameters
    ----------
    session_list : array-like of str
        Session IDs to include.
    t_Max : int
        Number of reward-history lags to include.
    t_Max_hist : int
        Number of hit-history lags to include (default 0).
    n_jobs : int
        Number of parallel workers (default -1 = all cores).
    backend : str
        joblib backend.
    verbose : int
        joblib verbosity.

    Returns
    -------
    dict with keys: rwd_coeffs, hit_coeffs, rwd_CI, hit_CI, model,
                    combined_reg_df, n_sessions, n_animals
    """
    reg_df_list = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_build_hit_glm_session_df)(session, t_Max, t_Max_hist)
        for session in session_list
    )

    combined_reg_df = pd.concat(reg_df_list, ignore_index=True)

    X = combined_reg_df.drop(columns=['hit']).values
    y = combined_reg_df['hit'].values.astype(float)
    X = sm.add_constant(X)

    nan_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[nan_mask]
    y = y[nan_mask]

    model = sm.Logit(y, X)
    result = model.fit()

    rwd_coeffs = result.params[1:t_Max + 1]
    hit_coeffs = result.params[t_Max + 1:t_Max + 1 + t_Max_hist]
    rwd_CI = result.conf_int()[1:t_Max + 1]
    hit_CI = result.conf_int()[t_Max + 1:t_Max + 1 + t_Max_hist]

    n_sessions = len(session_list)
    n_animals = len({parseSessionID(sess)[0] for sess in session_list})

    return {
        'rwd_coeffs': rwd_coeffs,
        'hit_coeffs': hit_coeffs,
        'rwd_CI': rwd_CI,
        'hit_CI': hit_CI,
        'model': result,
        'combined_reg_df': combined_reg_df,
        'n_sessions': n_sessions,
        'n_animals': n_animals,
    }
