
function Cudnn_path_check()
cudnn_lib = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp';
cudnn_bin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin';
cudnn_path = [cudnn_lib, ';',  cudnn_bin, ';'];

if isempty(getenv('PATH_BACKUP'))
    
    Env_PATH = getenv('PATH');
    setenv('PATH_BACKUP', [cudnn_path, getenv('PATH')]);
       
else
    
end

setenv('PATH', getenv('PATH_BACKUP'));

end