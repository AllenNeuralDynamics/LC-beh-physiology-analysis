from tornado.web import url
from dask.dot import name
from datetime import datetime, timezone
import pandas as pd

from aind_data_schema.components.identifiers import Code, DataAsset
from aind_data_schema.core.metadata import Metadata, Procedures
from aind_data_schema.core.processing import (
    DataProcess,
    Processing,
    ProcessName,
    ProcessStage,
)
from aind_data_schema.utils.inheritance import derive_data_description
from codeocean import CodeOcean
import json
import os
from pathlib import Path

from aind_metadata_upgrader.upgrade import Upgrade

prefix_path = Path("/data/scratch_data")

meta_data_file = '/root/capsule/code/data_management/meta_data_processing_updated.xlsx'
session_meta = pd.read_excel(meta_data_file, sheet_name='sessions')
stan_meta = pd.read_excel(meta_data_file, sheet_name='animal_stan')
glm_meta = pd.read_excel(meta_data_file, sheet_name='animal_glm')
ccf_meta = pd.read_excel(meta_data_file, sheet_name='all_ccf')
dorsal_edge_meta = pd.read_excel(meta_data_file, sheet_name='all_dorsal_edge')

computation_params_file = '/root/capsule/code/data_management/processing_params.json'
with open(computation_params_file, 'r') as f:
    computation_params = json.load(f)


from aind_data_access_api.document_db import MetadataDbClient

docdb_api_client = MetadataDbClient(
    host="api.allenneuraldynamics.org",
    version="v2"
)
docdb_api_client_v1 = MetadataDbClient(
    host="api.allenneuraldynamics.org",
    version="v1"
)

session_data_files = ['/root/capsule/code/data_management/hopkins_session_assets.csv',
                        '/root/capsule/code/data_management/session_assets.csv',
                        '/root/capsule/code/data_management/hopkins_FP_session_assets.csv']
session_assets = pd.concat([pd.read_csv(file) for file in session_data_files], ignore_index=True)
# remove nan sessions
session_assets = session_assets.dropna(subset=['session_id'])
session_assets['animal_id'] = session_assets['session_id'].apply(lambda x: x.split("_")[1])
# attached_assets = pd.read_csv('/root/capsule/code/data_management/session_assets_sources.csv')

session_processing_check = pd.DataFrame(index=list(session_assets["session_id"]) + [f"{x}_combined" for x in session_assets["animal_id"].unique()],
 columns=list(session_meta["name"])+list(glm_meta["name"]), data=False)

client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("API_SECRET"))

def create_session_meta(session_id):
    # prepare data asset names
    aniID = session_id.split("_")[1]
    session_dict = session_assets[session_assets['session_id'] == session_id].to_dict(orient='records')[0]
    levels = ['raw_data', 'sorted_curated']
    session_data_assets = {level: client.data_assets.get_data_asset(data_asset_id=session_dict[level]).name for level in levels if pd.notna(session_dict[level])}
    data_assets = [DataAsset(name=name) for name in session_data_assets.values()]

    dp_list =[]
    asset_prefix = prefix_path/aniID
    if 'sorted_curated' in session_data_assets:
        curated_asset_name = session_data_assets['sorted_curated']
        data_time = curated_asset_name[-19:]
        t = datetime.strptime(data_time, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)
    else:
        t = datetime(2026, 3, 27, 00, 00, 00, tzinfo=timezone.utc)
        

    for row_ind, row in session_meta.iterrows():
        if not row["included in nwb"]:
            continue
        outputs = row['output'].format(aniID=aniID, session_id=session_id).split(',')
        if row['name'] in computation_params:
            params = computation_params[row['name']]
            # print(f"Using parameters for {row['name']}: {params}")
        else:
            params = None
        curr_code = Code(
            url="https://github.com/AllenNeuralDynamics/aind-beh-ephys-analysis",
            run_script=row['run_script'],
            version="v5.0",
            parameters=params,
            input_data=data_assets
        )
        for i, output in enumerate(outputs):
            output = f"{session_id}/{output.replace(" ","")}"
            # skip if not processed
            test_output = asset_prefix / output
            if not test_output.exists():
                print(f"Output file {output} does not exist, skipping {row['name']}")
                break
            
            session_processing_check.loc[session_id, row['name']] = True
            session_date = session_id[-19:]
            suffix = f"_{i}" if len(outputs) > 1 else ""
            curr_dp = DataProcess(
                        process_type=ProcessName.OTHER,
                        name=f"{row['name']}_{session_date}{suffix}",
                        experimenters=["Sue Su"],
                        stage=ProcessStage.ANALYSIS,
                        start_date_time=t,
                        end_date_time=t,
                        # output_path=output,
                        code=curr_code,
                        notes=row['name'] + '. ' + row['additional_note'] if pd.notna(row['additional_note']) else row['name'],
                        )
            dp_list.append(curr_dp)

    p = Processing.create_with_sequential_process_graph(
        data_processes=dp_list)
    return p, session_data_assets
# Tongue movement data
TONGUE_MOVEMENT_ASSET_NAME = "LC-ephys-tonguemovements_2026-08-28_00-00-00"
KEYPOINT_ASSET_NAME = "keypoint-tracking-bottomview-LCrecordings_2026-04-03_18-07-14"
TONGUE_MOVEMENT_DATA_DIR = Path('/root/capsule/data/all_tongue_movements')
KEYPOINT_TRACKING_DIR = Path('/root/capsule/data/keypoint_tracking_bottomview_LCrecordings_20260403')

def get_processing_subset(session_id, asset_path):
    p = Processing.model_validate_json((asset_path / "processing.json").read_text())
    dps = p.data_processes
    input_assets = [x for x in dps[0].code.input_data if x.name.startswith(session_id)]
    if not input_assets:
        print(f"No input assets found for session {session_id} in {asset_path}.")
        return None
    p.data_processes[0].code.input_data = input_assets
    return p

def packaging_processing(source_names):
    dp = DataProcess(
        process_type=ProcessName.OTHER,
        name="complete_nwb_packaging",
        experimenters=["Sue Su"],
        stage=ProcessStage.PROCESSING,
        start_date_time=datetime.now(),
        code=Code(
            url="https://github.com/AllenNeuralDynamics/aind-beh-ephys-analysis",
            run_script="code/data_management/build_merged_nwb.py",
            commit_hash="130d64a014df967a46e8cf67acc3b597af2753fe",
            input_data=[DataAsset(name=name) for name in source_names]
        )
    )
    return Processing(data_processes=[dp])

def write_session_metadata(session_id, include_tongue=True, include_keypoint=True):
    p, session_data_assets = create_session_meta(session_id)
    source_names = list(session_data_assets.values())

    session_id_new = session_data_assets["raw_data"]
    tongue_movements = get_processing_subset(session_id_new, TONGUE_MOVEMENT_DATA_DIR)
    if tongue_movements and include_tongue:
        p += tongue_movements
        source_names.append(TONGUE_MOVEMENT_ASSET_NAME)
    keypoint = get_processing_subset(session_id_new, KEYPOINT_TRACKING_DIR)
    if keypoint:
        p += keypoint
        source_names.append(KEYPOINT_ASSET_NAME)

    p += packaging_processing(source_names)

    if 'sorted_curated' in session_data_assets:
        base_asset_name = session_data_assets['sorted_curated']
    else:
        base_asset_name = session_data_assets['raw_data']

    # v2 metadata from query
    base_json = docdb_api_client.retrieve_docdb_records(
        filter_query=dict(name=base_asset_name),
    )[0]
    try:
        base_md = Metadata.model_validate(base_json)
    except Exception as e:
        base_json = docdb_api_client_v1.retrieve_docdb_records(
            filter_query=dict(name=base_asset_name),
        )[0]
        base_md = Upgrade(base_json).metadata

    # Create the derived metadata -- this applies all four inheritance rules
    # derived = Metadata.from_metadata(
    #     base_md,
    #     process_name="packaged_nwb",
    #     # isn't used and shouldn't be needed but may require a placeholder
    #     location="s3://my-bucket/derived-asset",
    #     new_processing=p,
    # )
    
    procedures_empty = Procedures(
        subject_id=base_md.procedures.subject_id,
    ).model_dump()

    process_name = "nwb-complete"
    derived_dd = derive_data_description(   
                base_md.data_description,
                process_name=process_name,
                source_data=source_names,
        )
    derived = Metadata(
            name=derived_dd.name,
            location=f"s3://aind-open-data/{derived_dd.name}",
            data_description=derived_dd,
            subject=base_md.subject,
            procedures=procedures_empty,
            instrument=base_md.instrument,
            acquisition=base_md.acquisition,
            processing=base_md.processing + p if base_md.processing else p,
            quality_control=base_md.quality_control,
        )
    # fill invalid procedures
    derived.procedures = base_md.procedures
    return derived

if __name__ == "__main__":
    example_sessions = [
    'behavior_ZS062_2021-05-06_15-46-14',
    'behavior_ZS059_2021-04-29_14-02-45',
    'behavior_ZS061_2021-04-08_18-01-30',
    'behavior_781166_2025-05-13_14-04-27',
    'behavior_754897_2025-03-12_12-23-15',
    'behavior_754897_2025-03-13_11-20-42',
    'behavior_754898_2025-01-01_20-40-03',
    'behavior_749472_2025-01-09_13-56-02',
    'behavior_754896_2025-01-03_17-20-19'
]
for id in example_sessions[:]:
    os.makedirs(f"/scratch/{id}", exist_ok=True)
    derived = write_session_metadata(id)
    derived.write_standard_files(output_directory=f"/scratch/{id}")
    derived.write_standard_file(output_directory=f"/scratch/{id}")