"""

Find folders and delete if they exist.

Try to use before every experiment so that frames and snapshots don't get used from previous experiments

"""

from pathlib import Path
import shutil

# remove frames directory
frames_path = Path('./frames')
if frames_path.exists() and frames_path.is_dir():
    shutil.rmtree(frames_path)


# remove snapshots directory
snapshots_path = Path('./snapshots')
if snapshots_path.exists() and snapshots_path.is_dir():
    shutil.rmtree(snapshots_path)