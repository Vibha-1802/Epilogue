import scipy.io as sio
import json
import numpy as np
from scipy.io.matlab import mat_struct

def convert_mat_to_dict(mat_obj):
    if isinstance(mat_obj, mat_struct):
        return {f: convert_mat_to_dict(getattr(mat_obj, f)) for f in mat_obj._fieldnames}
    elif isinstance(mat_obj, np.ndarray):
        return [convert_mat_to_dict(item) for item in mat_obj]
    else:
        return mat_obj

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            val = obj.item()
            if isinstance(val, bytes):
                return val.decode('utf-8', errors='ignore')
            return val
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

try:
    c = sio.loadmat(r'c:\Users\tina\epilogue\Epilogue\egovehicle_data\c_data.mat', squeeze_me=True, struct_as_record=False)
    camera_dict = convert_mat_to_dict(c['camera'])
    with open(r'c:\Users\tina\epilogue\Epilogue\egovehicle_data\c_data_readable.json', 'w') as f:
        json.dump(camera_dict, f, cls=NumpyEncoder, indent=4)
    print("Successfully exported c_data.mat to c_data_readable.json")
except Exception as e:
    print(f"Error exporting c_data.mat: {e}")

try:
    r = sio.loadmat(r'c:\Users\tina\epilogue\Epilogue\egovehicle_data\r_data.mat', squeeze_me=True, struct_as_record=False)
    radar_dict = convert_mat_to_dict(r['radar'])
    with open(r'c:\Users\tina\epilogue\Epilogue\egovehicle_data\r_data_readable.json', 'w') as f:
        json.dump(radar_dict, f, cls=NumpyEncoder, indent=4)
    print("Successfully exported r_data.mat to r_data_readable.json")
except Exception as e:
    print(f"Error exporting r_data.mat: {e}")
