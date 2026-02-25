function [ denoised ] = denoise(noisy,sigma_hat,width,height,denoiser)
% function [ denoised ] = denoise(noisy,sigma_hat,width,height,denoiser)
%DENOISE takes a signal with additive white Guassian noisy and an estimate
%of the standard deviation of that noise and applies some denosier to
%produce a denoised version of the input signal
% Input:
%       noisy       : signal to be denoised
%       sigma_hat   : estimate of the standard deviation of the noise
%       width   : width of the noisy signal
%       height  : height of the noisy signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g.
%       denoiser='BM3D'
%Output:
%       denoised   : the denoised signal.

%To apply additional denoisers within the D-AMP framework simply add
%aditional case statements to this function and modify the calls to D-AMP
global global_time
% noisy=reshape(noisy,[width,height]);
noisy=reshape(noisy,[height,width]);

% =========================================================================
% 【新增】针对超大图像 (如 > 110万像素) 的分块保护逻辑
% 仅拦截 Urban100 中的 20.png (1280x960)，其他 1024x1024 图像直接放行
% =========================================================================
MAX_PIXELS_THRESHOLD = 1100000;
[h_curr, w_curr] = size(noisy);

if h_curr * w_curr > MAX_PIXELS_THRESHOLD
    % === 启用大尺寸分块 (512x512) ===
    patch_size = 512;
    step = 512; % 无重叠，避免插值担忧，保留原始像素处理
    
    output = zeros(size(noisy));
    count_map = zeros(size(noisy));
    
    for r = 1 : step : h_curr
        for c = 1 : step : w_curr
            % 1. 确定当前块坐标
            r_end = min(r + patch_size - 1, h_curr);
            c_end = min(c + patch_size - 1, w_curr);
            
            % 2. 提取分块
            patch_in = noisy(r:r_end, c:c_end);
            [p_h, p_w] = size(patch_in); % 获取当前块的实际尺寸
            
            % 3. 调用原有的去噪逻辑 (传入块的尺寸)
            patch_out = core_denoise_logic(patch_in, sigma_hat, p_w, p_h, denoiser);
            
            % 4. 填回结果
            output(r:r_end, c:c_end) = output(r:r_end, c:c_end) + patch_out;
            count_map(r:r_end, c:c_end) = count_map(r:r_end, c:c_end) + 1;
        end
    end
    % 处理可能的重叠（当前step=patch_size其实不需要，但为了代码健壮性保留）
    output = output ./ count_map;
    
else
    % === 正常情况 (<= 1024x1024) ===
    % 直接调用原有逻辑，不进行任何分块
    output = core_denoise_logic(noisy, sigma_hat, width, height, denoiser);
end

denoised=output(:);
save exampleScriptMAT
end


% =========================================================================
% 下面是你原本的所有代码，逻辑、注释完全未动，仅封装在子函数中
% =========================================================================
function output = core_denoise_logic(noisy, sigma_hat, width, height, denoiser)
    
    global global_time
    
    % 注意：传入子函数时 noisy 已经是 reshape 好的矩阵，
    % 但为了兼容下面 case 中可能存在的再次 reshape，这里保持原有变量名
    
    switch denoiser
        case 'NLMF'
            if min(height,width)==1
                %Scale signal from 0 to 1 for NLM denoiser
                max_in=max(noisy(:));
                min_in=min(noisy(:));
                range_in=max_in-min_in+eps;
                noisy=(noisy-min_in)/range_in;
                Options.filterstrength=sigma_hat/range_in*1.5;
                Options.kernelratio=5;
                Options.windowratio=10;
                output=range_in*NLMF(noisy,Options)+min_in;
            else
                if sigma_hat>200
                    Options.kernelratio=5;
                    Options.windowratio=17;
                    Options.filterstrength=sigma_hat/255*.9;
                elseif sigma_hat>150
                    Options.kernelratio=5;
                    Options.windowratio=17;
                    Options.filterstrength=sigma_hat/255*.8;
                elseif sigma_hat>100
                    Options.kernelratio=5;
                    Options.windowratio=17;
                    Options.filterstrength=sigma_hat/255*.6;
                elseif sigma_hat>=75
                    Options.kernelratio=5;
                    Options.windowratio=17;
                    Options.filterstrength=sigma_hat/255*.5;
                elseif sigma_hat>=45
                    Options.kernelratio=4;
                    Options.windowratio=17;
                    Options.filterstrength=sigma_hat/255*.6;
                elseif sigma_hat>=30
                    Options.kernelratio=3;
                    Options.windowratio=17;
                    Options.filterstrength=sigma_hat/255*.9;
                elseif sigma_hat>=15
                    Options.kernelratio=2;
                    Options.windowratio=10;
                    Options.filterstrength=sigma_hat/255*1;
                else
                    Options.kernelratio=1;
                    Options.windowratio=10;
                    Options.filterstrength=sigma_hat/255*2;
                end
                Options.nThreads=4;
                Options.enablepca=true;
                output=255*NLMF(noisy/255,Options);
            end
        case 'Gauss'
            h = fspecial('gaussian',5,sigma_hat);
            output=imfilter(noisy,h,'symmetric');
        case 'Bilateral'
            Options.kernelratio=0;
            Options.blocksize=256;
            if sigma_hat>200
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*5;
            elseif sigma_hat>150
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*4.5;
            elseif sigma_hat>100
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*3.5;
            elseif sigma_hat>=75
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*3;
            elseif sigma_hat>=45
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*2.5;
            elseif sigma_hat>=30
                Options.windowratio=17;
                Options.filterstrength=sigma_hat/255*2.2;
            elseif sigma_hat>=15
                Options.windowratio=10;
                Options.filterstrength=sigma_hat/255*2.2;
            else
                Options.windowratio=10;
                Options.filterstrength=sigma_hat/255*2;
            end
            Options.nThreads=4;
            Options.enablepca=false;
            output=255*NLMF(noisy/255,Options);
        case 'BLS-GSM'
            %Parameters are BLS-GSM default values
            PS = ones([width,height]);
            seed = 0;
            Nsc = ceil(log2(min(width,height)) - 4);
            Nor = 3;
            repres1 = 'uw';
            repres2 = 'daub1';
            blSize = [3 3];
            parent = 0;
            boundary = 1;
            covariance = 1;
            optim = 1;
            output = denoi_BLS_GSM(noisy, sigma_hat, PS, blSize, parent, boundary, Nsc, Nor, covariance, optim, repres1, repres2, seed);
        case 'BM3D'
            tic
            [NA, output]=BM3D(1,noisy,sigma_hat,'np',0);
            output=255*output;
            global_time = global_time+toc;
            
        case 'fast-BM3D'
            [NA, output]=BM3D(1,noisy,sigma_hat,'lc',0);
            output=255*output;
        case 'BM3D-SAPCA'
            output = 255*BM3DSAPCA2009(noisy/255,sigma_hat/255);
            
        case 'TWSC'
            output = denoise_TWSC_Sigma_RW(noisy/255);
            
        case 'DnCNN_20Layers_10cases'
            output=denoise_DnCNN_20Layers_10cases(noisy, sigma_hat);
            
        case 'DnCNN_17Layers_10cases'
            output=denoise_DnCNN_17Layers_10cases(noisy, sigma_hat);
            
        case 'DnCNN_20Layers_3x3_17cases'
            output=denoise_DnCNN_20Layers_3x3_17cases(noisy, sigma_hat);
            
        case 'EDSR_3x3_10cases'
            output = denoise_EDSR_3x3_10cases(noisy, sigma_hat);
            
        case 'EDSR_5x5_17cases'
            output = denoise_EDSR_5x5_17cases(noisy, sigma_hat);
            
        case {'RCAN_5x5_dilated_17cases','RCAN_5x5_dilated_17cases+'}
            output=denoise_RCAN_5x5_dilated_17cases(noisy, sigma_hat);
            
        case {'RCAN_5x5_dilated_woCA'}
            output=denoise_RCAN_5x5_dilated_woCA(noisy, sigma_hat);
            
        case 'RCAN_MS_5x5_dilated'
            output = denoise_RCAN_MS_5x5_dilated(noisy, sigma_hat);
            
        case 'RCAN_5x5_dilated_10cases'
            output=denoise_RCAN_5x5_dilated_10cases(noisy, sigma_hat);
            
        case 'RCAN_5x5_17cases'
            output=denoise_RCAN_5x5_17cases(noisy, sigma_hat);
            
        case 'RCAN_3x3_dilated_17cases'
            output=denoise_RCAN_3x3_dilated_17cases(noisy, sigma_hat);
            
        case 'RCAN_5x5_dilated_1case'
            output=denoise_RCAN_5x5_dilated_1case(noisy, sigma_hat);
            
        case 'RCAN_7x7_dilated_17cases'
            output=denoise_RCAN_7x7_dilated_17cases(noisy, sigma_hat);
            
        case 'RCAN_TEST'
            output=denoise_RCAN_TEST(noisy, sigma_hat);
            
        case 'PCLR'
            output = PCLR(noisy, sigma_hat);
            
        case 'NLM'
            patchSize = 10;
            decay_para = 1.0; %decay parameter
            searchWindowSize = 10;
            output = nonLocalMeans(noisy, sigma_hat, decay_para, patchSize, searchWindowSize);
            
        case 'WNNM'
            WNNM_Par   = WNNM_ParSet(sigma_hat);
            output = WNNM_DeNoising( noisy, WNNM_Par );
            
        case 'MWCNN17'
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            %noisy = py.numpy.fromstring(num2str(noisy(:)'), py.numpy.float64, int8(-1), char(' ')).reshape(int32(size(noisy)));
            output = double(py.MWCNN_matlab.denoise17(noisy,width,height,sigma_hat));
            output = reshape(output,width,height);
            output = output';
            
        case 'MWCNN24'
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            %noisy = py.numpy.fromstring(num2str(noisy(:)'), py.numpy.float64, int8(-1), char(' ')).reshape(float64(size(noisy)));
            output = double(py.MWCNN_matlab.denoise24(noisy,height,width,sigma_hat)); % Noting!!! The order of vectorisation is different between python and matlab!!!!!!
            output = reshape(output,width,height);
            output = output';
            
            
        case 'MWDNN'
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            output = double(py.denoiser_MWDNN.denoise17(noisy, height, width, sigma_hat));
            output = reshape(output,width,height);
            output = output';
            
        case 'SUNet' %SUnet
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            output = double(py.denoiser_SUNet.denoise124(noisy,sigma_hat));
            %         output = double(py.denoiser_SUNet2.main());
            output = reshape(output,height,width);
            output = output';
            
        case 'DPIR'
            noisy = reshape(noisy,1,height*width);
            output = double(py.DPIR_matlab.denoiser(noisy, width, height, sigma_hat));
            output = reshape(output,height,width);
            
        case 'TNRD' %浼犵粺鏂规硶锛屽緢鎱?
            [output] = denoise_TNRD(noisy, sigma_hat);
            
            %     case 'Restormer'
            %          if sigma_hat >= 150
            %              output=denoise_RCAN_3x3_dilated_17cases(noisy, sigma_hat);
            %         else
            %             noisy = reshape(noisy,height,width);
            %             noisy = noisy';
            %             noisy = reshape(noisy,1,height*width);
            %             output = double(py.Restormer_denoise_matlab.denoiser(noisy, height, width, sigma_hat));
            %             output = reshape(output,width,height);
            %             output = output';
            %          end
            
            %     case 'Restormer'
            %         % 准备数据：Restormer 和 MWCNN (Python版) 都需要这种格式
            %         % 将图像转置并拉成一行向量
            %         noisy_in = reshape(noisy, height, width);
            %         noisy_in = noisy_in';
            %         noisy_in = reshape(noisy_in, 1, height * width);
            %
            %         if sigma_hat >= 150
            %
            %             output = double(py.MWCNN_matlab.denoise24(noisy_in, width, height, sigma_hat));
            %
            %         else
            %             output = double(py.Restormer_denoise_matlab.denoiser(noisy_in, height, width, sigma_hat));
            %             output = reshape(output, width, height);
            %             output = output';
            %         end
        case 'Restormer'
            if sigma_hat>=150
                %output=denoise_RCAN_3x3_dilated_17cases(noisy, sigma_hat);
                %output=denoise_DnCNN_20Layers_3x3_17cases(noisy, sigma_hat);
                noisy = reshape(noisy,height,width);
                noisy = noisy';
                noisy = reshape(noisy,1,height*width);
                %noisy = py.numpy.fromstring(num2str(noisy(:)'), py.numpy.float64, int8(-1), char(' ')).reshape(float64(size(noisy)));
                output = double(py.MWCNN_matlab.denoise24(noisy,height,width,sigma_hat)); % Noting!!! The order of vectorisation is different between python and matlab!!!!!!
                output = reshape(output,width,height);
                output = output';
            else
                
                noisy = reshape(noisy,height,width);
                noisy = noisy';
                noisy = reshape(noisy,1,height*width);
                output = double(py.Restormer_denoise_matlab.denoiser(noisy, height, width, sigma_hat));
                output = reshape(output,width,height);
                output = output';
            end
            
        otherwise
            error('Unrecognized Denoiser')
    end
end