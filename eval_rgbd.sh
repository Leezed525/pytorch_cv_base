# eval depthtrack
cd Depthtrack_workspace
vot evaluate  LeeNet
vot analysis  LeeNet
vot report LeeNet
cd ..

# eval vot22-rgbd
cd VOT22RGBD_workspace
vot evaluate  LeeNet
vot analysis  LeeNet
vot report LeeNet
cd ..

