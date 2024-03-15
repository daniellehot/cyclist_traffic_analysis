# 1. Install labelme
```bash
git clone https://github.com/labelmeai/labelme.git
cd labelme
pip install -e .
```

# QT Error - Cannot move to target thread 
Remove pip installation of opencv and install opencv with *apt*
```bash
pip3 uninstall opencv-python
sudo apt install libopencv-dev python3-opencv
```

# 2. Launch labelme 
```bash 
cd cyclist_traffic_analysis/annotating
labelme --nodata --labels labels.txt --labelflags labelflags.json  --keep-prev --autosave
```

# 3. Annotating 
0. Optional - Add a shortcut for Create AI polygon by:
- assigning a shortcut to the Create AI polygon action in *labelme/app.py*, 
- adding a shortcut to the template config file *labelme/config/default_config.yaml* , 
- adding a shortcut to the default config file *~/.labelmerc* 
1. Create a boudning box or a mask (whatever is easier)
2. Pick a label and corresponding flags if cyclist. 
3. Track each objet using the *group ID* option.