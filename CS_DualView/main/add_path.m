function add_path()

addpath('../Enc');
addpath('../Dec');
addpath('../Utilities');
addpath('../Utilities/Measurements');
addpath('../Utilities/Measurements/mask');

addpath(genpath('../TVNLR'))
addpath(genpath('../../DLAMP_Toolbox/gampmatlab'));
addpath(genpath('../../DLAMP_Toolbox/Algorithms'));
addpath(genpath('../../DLAMP_Toolbox/Utils'));
addpath(genpath('../../DLAMP_Toolbox/Packages/BM3D'));
addpath(genpath('../../DLAMP_Toolbox/Packages/NonLocalMeansDenoising'));
addpath(genpath('../../DLAMP_Toolbox/Packages/TWSC-ECCV2018'));
% addpath(genpath('../Utilities'))
addpath(genpath('../../Matlab_Tools/MRI_lab'));

%% python-env (only keep what Restormer pipeline actually uses)
addpath('../../Trained_Weights/Restormer/');
addpath('../../Trained_Weights/MWCNN/');
addpath('../../Trained_Weights/sigma_estimate/');

if count(py.sys.path,'../../Trained_Weights/Restormer/') == 0
    insert(py.sys.path,int32(0),'../../Trained_Weights/Restormer/');
end
if count(py.sys.path,'../../Trained_Weights/MWCNN/') == 0
    insert(py.sys.path,int32(0),'../../Trained_Weights/MWCNN/');
end
if count(py.sys.path,'../../Trained_Weights/sigma_estimate/') == 0
    insert(py.sys.path,int32(0),'../../Trained_Weights/sigma_estimate/');
end


%%'C:\path\to\python.exe'
insert(py.sys.path,int32(0),'D:/Anaconda/envs/PointNet/');
setenv('PATH', 'D:/Anaconda/envs/PointNet/');
%%



end
