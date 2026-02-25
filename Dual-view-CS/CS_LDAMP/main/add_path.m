function add_path()

addpath('../Enc');
addpath('../Dec');
addpath('../channel');
addpath(genpath('../BCS'));
addpath('../NLR')
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
addpath(genpath('../../Matlab_Tools/MRI_lab'));
%rmpath('/home/cz/Workspace/DLAMP_Toolbox/DnCNN_TIP2017-master/utilities');
%rmpath('/home/cz/Workspace/CS_LDAMP/TVNLR/Utilities');
%rmpath('/home/cz/Workspace/DLAMP_Toolbox/Packages')

warning on

addpath('C:/Users/zanchen/Desktop/wb_PSNR/Matlab/Matlab_Tools/matconvnet-1.0-beta25/matlab');
addpath('C:/Users/zanchen/Desktop/wb_PSNR/Matlab/Matlab_Tools/matconvnet-1.0-beta25/matlab/simplenn');
addpath('C:/Users/zanchen/Desktop/wb_PSNR/Matlab/Matlab_Tools/matconvnet-1.0-beta25/matlab/mex');
addpath('../NoiseLevelEstimate1');
addpath(genpath('../../Trained_Weights'));
run('C:/Users/zanchen/Desktop/wb_PSNR/Matlab/Matlab_Tools/matconvnet-1.0-beta25/matlab/vl_setupnn.m');

%% python-env
addpath('..\..\Trained_Weights\Restormer\');
addpath('..\..\Trained_Weights\MWCNN\');
addpath('..\..\Trained_Weights\NCRCAN\');
addpath('..\..\Trained_Weights\RCAN1B\');
addpath('..\..\Trained_Weights\MWDNN\');
addpath('..\..\Trained_Weights\sigma_estimate_ablation\');
if count(py.sys.path,'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\Restormer\') == 0
    insert(py.sys.path,int32(0),'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\Restormer\');
end
if count(py.sys.path,'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\MWCNN\') == 0
    insert(py.sys.path,int32(0),'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\MWCNN\');
end
if count(py.sys.path,'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\NCRCAN\') == 0
    insert(py.sys.path,int32(0),'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\NCRCAN\');
end
if count(py.sys.path,'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\RCAN1B\') == 0
    insert(py.sys.path,int32(0),'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\RCAN1B\');
end
if count(py.sys.path,'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\sigma_estimate_ablation\') == 0
    insert(py.sys.path,int32(0),'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\sigma_estimate_ablation\');
end
if count(py.sys.path,'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\sigma_estimate_ablation\models\') == 0
    insert(py.sys.path,int32(0),'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\sigma_estimate_ablation\models\');
end
if count(py.sys.path,'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\DPIR\') == 0
    insert(py.sys.path,int32(0),'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\DPIR\');
end
if count(py.sys.path,'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\MWDNN\') == 0
    insert(py.sys.path,int32(0),'C:\Users\zanchen\Desktop\ht\Dual-view-CS\Trained_Weights\MWDNN\');
end

insert(py.sys.path,int32(0),'C:\ProgramData\Anaconda3\envs\WT\');
setenv('PATH', 'C:\ProgramData\Anaconda3\envs\WT\');
%%


py.importlib.reload(py.importlib.import_module('Restormer_denoise_matlab'));
% py.importlib.reload(py.importlib.import_module('MWCNN_matlab'));
% py.importlib.reload(py.importlib.import_module('ncrcan'));
% py.importlib.reload(py.importlib.import_module('rcan1b'));
% % py.importlib.reload(py.importlib.import_module('DPIR_matlab'));
% py.importlib.reload(py.importlib.import_module('denoiser_MWDNN'));
% py.importlib.reload(py.importlib.import_module('Sigma_hat'));


end
