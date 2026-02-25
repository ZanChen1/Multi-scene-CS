function [x_hat_1, x_hat_2, PSNR_sum, MSE_sum] = Prox_Moment_DualView(y, iters, M_func_1, Mt_func_1, M_func_2, Mt_func_2, measure_1, measure_2, iterative_way)
% Prox_Moment_DualView_Final
% 1. 速度优化：无循环IO
% 2. 完美兼容：支持单/双/三/四视角
% 3. 智能CSV：解决了不同尺寸图片导致采样率波动从而生成多个文件的问题
%    - 文件名不再包含具体 Rate，只包含模式
%    - 具体 Rate 记录在 CSV 表格内部

    global global_time
    global_time = 0;
    
    randn('state', 0);
    rand('state', 0);

    %% ================= 1. 初始化与视角配置 =================
    num_views = 2; % 默认
    
    % 基础句柄
    M_ops{1} = @(x) M_func_1(x);  Mt_ops{1} = @(z) Mt_func_1(z);
    M_ops{2} = @(x) M_func_2(x);  Mt_ops{2} = @(z) Mt_func_2(z);
    measures{1} = measure_1; measures{2} = measure_2;
    
    % --- 动态检测视角数量 ---
    if isempty(measure_2), num_views = 1; end
    
    % Case 4/8: 三视角
    if iterative_way == 4 || iterative_way == 8
        num_views = 3;
        measures{3} = measure_2.measure_3;
        M_ops{3} = @(x) measure_2.A_3(x); Mt_ops{3} = @(z) measure_2.At_3(z);
        measures{2} = rmfield(measures{2}, {'measure_3', 'A_3', 'At_3'});
    end
    
    % Case 8: 四视角
    if iterative_way == 8
        num_views = 4;
        measures{4} = measure_2.measure_4;
        M_ops{4} = @(x) measure_2.A_4(x); Mt_ops{4} = @(z) measure_2.At_4(z);
        measures{2} = rmfield(measures{2}, {'measure_4', 'A_4', 'At_4'});
    end

    % 统一去噪句柄
    denoi_ops = cell(1, num_views);
    for v = 1:num_views
        denoi_ops{v} = @(noisy, sigma) denoise(noisy, sigma, measures{v}.image_width, measures{v}.image_height, measures{v}.denoize_name);
    end

    % 处理输入 y
    if iscell(y), y_vec = y; m_len = length(y{1}); else, y_vec = y; m_len = length(y); end

    %% ================= 2. 变量初始化 =================
    x_t = cell(num_views, 2); 
    v_t = cell(num_views, 1);
    sigma_hat = zeros(1, num_views); 
    eta = cell(1, num_views); 
    gamma_part = cell(1, num_views);
    
    % 列索引映射: View1->Col3, View2->Col4, View3->Col6, View4->Col8
    col_map = [3, 4, 6, 8]; 
    PSNR_sum = zeros(iters, 8); 
    MSE_sum  = zeros(iters, 8); 

    tic_init = tic;
    for v = 1:num_views
        if iterative_way == 3 && iscell(y_vec), x_t{v, 1} = Mt_ops{v}(y_vec{v}); else, x_t{v, 1} = Mt_ops{v}(y_vec); end
        
        sigma_hat(v) = SigEstmate_SigCNN(reshape(x_t{v, 1}, measures{v}.image_height, measures{v}.image_width));
        x_t{v, 2} = double(denoi_ops{v}(x_t{v, 1}, sigma_hat(v)));
        v_t{v} = zeros(measures{v}.length, 1); 
        eta{v} = randn(1, measures{v}.length);
    end
    global_time = global_time + toc(tic_init);

    PSNR_func = @(x_hat, ori_im) PSNR(abs(ori_im), abs(x_hat));
    alpha = 1; epsilon = 1;

    %% ================= 3. 主迭代循环 =================
    for i = 1:iters
        tic_iter = tic;
        
        % A. Gamma 计算
        gamma_total = 0;
        for v = 1:num_views
            diff = denoi_ops{v}(x_t{v, 1} + epsilon * eta{v}', sigma_hat(v)) - x_t{v, 2};
            m_curr = m_len; 
            if iterative_way == 3 && iscell(y_vec), m_curr = length(y_vec{v}); end
            gamma_part{v} = 1 / (m_curr * epsilon) .* eta{v} * diff;
            gamma_total = gamma_total + gamma_part{v};
        end

        % B. 残差 v_temp
        v_temp = y_vec; 
        if iterative_way ~= 3
            for v = 1:num_views, v_temp = v_temp - (M_ops{v}(x_t{v, 2}))'; end
        end

        % C. 更新 x
        switch iterative_way
            case {1, 4, 8, 7} % 共享 Gamma
                for v = 1:num_views, v_t{v} = gamma_total .* v_t{v} + Mt_ops{v}(v_temp); x_t{v, 1} = x_t{v, 2} + alpha .* v_t{v}; end
            case 5 % 独立 Gamma
                for v = 1:num_views, v_t{v} = gamma_part{v} .* v_t{v} + Mt_ops{v}(v_temp); x_t{v, 1} = x_t{v, 2} + alpha .* v_t{v}; end
            case 6 % 固定 Gamma
                for v = 1:num_views, v_t{v} = 1 .* v_t{v} + Mt_ops{v}(v_temp); x_t{v, 1} = x_t{v, 2} + alpha .* v_t{v}; end
            case 3 % 单图模式
                for v = 1:num_views, v_loc = y_vec{v} - (M_ops{v}(x_t{v, 2}))'; v_t{v} = gamma_part{v} .* v_t{v} + Mt_ops{v}(v_loc); x_t{v, 1} = x_t{v, 2} + alpha .* v_t{v}; end
        end
        global_time = global_time + toc(tic_iter);

        % D. 记录与去噪
        for v = 1:num_views
            sigma_hat(v) = SigEstmate_SigCNN(reshape(x_t{v, 1}, measures{v}.image_height, measures{v}.image_width));
            x_t{v, 2} = double(denoi_ops{v}(x_t{v, 1}, sigma_hat(v)));
            
            im_rec = reshape(x_t{v, 2}, measures{v}.image_height, measures{v}.image_width);
            [p_val, m_val] = PSNR_func(im_rec, measures{v}.ori_im);
            idx = col_map(v); 
            PSNR_sum(i, idx) = p_val;
            MSE_sum(i, idx)  = m_val;
        end
    end
    
    %% ================= 4. 输出与智能CSV写入 =================
    x_hat_1 = reshape(x_t{1, 2}, [measure_1.image_height, measure_1.image_width]);
    if num_views >= 2, x_hat_2 = reshape(x_t{2, 2}, [measure_2.image_height, measure_2.image_width]); else, x_hat_2 = []; end
    if num_views >= 3, x_hat_3 = reshape(x_t{3, 2}, [measures{3}.image_height, measures{3}.image_width]); save('par_three_pic.mat', 'x_hat_3'); end
    if num_views >= 4, x_hat_4 = reshape(x_t{4, 2}, [measures{4}.image_height, measures{4}.image_width]); save('par_four_pic.mat', 'x_hat_4'); end

    % === CSV 写入逻辑 ===
    try
        % 1. 计算多视角的平均收敛曲线
        sum_curve = zeros(iters, 1);
        for v = 1:num_views
            sum_curve = sum_curve + PSNR_sum(:, col_map(v));
        end
        avg_curve = sum_curve / num_views;
        
        % 2. 构建文件名（移除具体 Rate，防止生成多个文件）
        if isfield(measure_1, 'Test_set_name'), set_name = measure_1.Test_set_name; else, set_name = 'Convergence_Results'; end
        out_dir = ['../results/', set_name, '/Convergence_Data/'];
        if ~exist(out_dir, 'dir'), mkdir(out_dir); end
        
        % 文件名只包含：算法名 + 模式。例如：Restormer_Mode1.csv
        % 这样 Set11 中所有图片（无论尺寸导致 Rate 怎么变）都会写到这一个文件里
        csv_file = [out_dir, measure_1.denoize_name, '_Mode', num2str(iterative_way), '.csv'];
        
        fp = fopen(csv_file, 'a');
        
        % 3. 写入表头 (包含 Actual_Rate 列)
        if ftell(fp) == 0
            header = 'Image_Name,Actual_Rate'; % 增加 Rate 列
            for k = 1:iters
                header = [header, sprintf(',Iter_%d', k)];
            end
            fprintf(fp, '%s\n', header);
        end
        
        % 4. 写入数据 (包含当前图片的实际 Rate)
        fprintf(fp, '%s,%.6f', measure_1.Image_name_1, measure_1.rate);
        for k = 1:iters
            fprintf(fp, ',%.4f', avg_curve(k));
        end
        fprintf(fp, '\n');
        fclose(fp);
        
    catch ME
        disp(['CSV Write Warning: ' ME.message]);
    end

end