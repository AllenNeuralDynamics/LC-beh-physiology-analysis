# %%
from codeocean.data_asset import DataAssetParams, DataAssetSearchParams,DataAssetAttachParams, Source, AWSS3Source
import pandas as pd
import os
os.sys.path.append('/root/capsule/code/beh_ephys_analysis')
from utils.beh_functions import parseSessionID
import os, sys
from codeocean import CodeOcean
client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("API_SECRET"))

# %%
# load and parse data ids
script_dir = os.path.dirname(os.path.abspath(__file__))
datalist_dir = os.path.join(script_dir, 'session_assets.csv')
data_df = pd.read_csv(datalist_dir)
data_df = data_df[data_df['session_id'].notna() & (data_df['session_id'] != "")]

# %%
col_to_attach = ['raw_data', 'sorted_curated', 'sorted', 'model_stan']

# %%
# Lists of strings for id and mount
all_ids = []
all_mounts = []

for curr_col in col_to_attach:
    valid_inds = [True if isinstance(s, str) and 30 < len(s) < 40 else False for s in data_df[curr_col].to_list()]
    session_ids = list(data_df[valid_inds]['session_id'].values)
    curr_ids = list(data_df[valid_inds][curr_col].values)
    curr_mount = [session_curr + '_' + curr_col for session_curr in session_ids]
    all_ids.extend(curr_ids)
    all_mounts.extend(curr_mount)
#%%
# all_ids = all_ids + ['f908dd4d-d7ed-4d52-97cf-ccd0e167c659']
# all_mounts = all_mounts + ['all_behavior']
all_ids = all_ids + ['c1a35fd0-c3aa-47a8-ba40-288b1e39a86a', 'ac7c7961-9178-4bf9-9d66-0a426cf3cc24', '1a8bede7-bdc1-4b41-8290-bc0bdafdf019', 'c712751d-f744-4fe8-9657-93a7084eab22', 'adf4f98c-5015-4d23-81e3-359a9a5b6ec1', 'c712751d-f744-4fe8-9657-93a7084eab22']
all_mounts = all_mounts + ['alignment_fix', 'dorsal_edges', 'merfish_data', 'LC_percentile_meshes', 'all_tongue_movements_16092025']

# Generate the list of DataAssetAttachParams objects
all_mounts_new = []
for id, mount in zip(all_ids, all_mounts):
    curr_mount = mount
    if 'stan' in mount:
        lists = mount.split('_model_stan')
        aniID, dateObj, raw_id = parseSessionID(lists[0])
        
        curr_mount = f'{aniID}_model_stan'
    all_mounts_new.append(curr_mount)   
        
all_mounts = all_mounts_new
data_assets = [DataAssetAttachParams(id, mount) for id, mount in zip(all_ids, all_mounts)]

# %%
computation_id = os.getenv("CO_COMPUTATION_ID")
results = client.computations.attach_data_assets(
    computation_id=computation_id,
    attach_params = data_assets,
)

print(f'Attached {len(all_ids)} data assets.')

for data_asset in results:
    print(f'{data_asset.id} mounted as {data_asset.mount}')

# %%
