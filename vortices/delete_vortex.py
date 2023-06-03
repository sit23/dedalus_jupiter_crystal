"""

Find folders and delete if they exist.

Try to use before every experiment so that frames and snapshots don't get used from previous experiments

"""

from pathlib import Path
import shutil

# remove frames directory
frames_path = Path('./vortices/vortex_frames')
if frames_path.exists() and frames_path.is_dir():
    shutil.rmtree(frames_path)


# remove snapshots directory
snapshots_path = Path('./vortices/vortex_snapshots') 
if snapshots_path.exists() and snapshots_path.is_dir():
    shutil.rmtree(snapshots_path)