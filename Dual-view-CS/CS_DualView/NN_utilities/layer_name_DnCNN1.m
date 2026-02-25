addpath('/home/cz/Workspace/Matlab_Tools/npy-matlab/');
fileFolder_input = fullfile('/home/cz/Workspace/Trained_Weights/DnCNN/DnCNN_3x3_17/');
dir_name = dir(fullfile(fileFolder_input,'*.npz'));
dir_number = length(dir_name);

for j = 1:dir_number
tmp = readNPY(fullfile(fileFolder_input,dir_name(j).name));
end