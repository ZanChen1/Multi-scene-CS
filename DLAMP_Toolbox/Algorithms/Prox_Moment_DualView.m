function [x_hat_1, x_hat_2, PSNR_sum, MSE_sum] = Prox_Moment_DualView(y, iters, M_func_1, Mt_func_1, M_func_2, Mt_func_2, measure_1, measure_2, iterative_way)

    global global_time
    global_time = 0;

    randn('state', 0);
    rand('state', 0);

    num_views = 2;

    M_ops{1} = @(x) M_func_1(x);  Mt_ops{1} = @(z) Mt_func_1(z);
    M_ops{2} = @(x) M_func_2(x);  Mt_ops{2} = @(z) Mt_func_2(z);
    measures{1} = measure_1; measures{2} = measure_2;

    if isempty(measure_2), num_views = 1; end

    if iterative_way == 4 || iterative_way == 8
        num_views = 3;
        measures{3} = measure_2.measure_3;
        M_ops{3} = @(x) measure_2.A_3(x); Mt_ops{3} = @(z) measure_2.At_3(z);
        measures{2} = rmfield(measures{2}, {'measure_3', 'A_3', 'At_3'});
    end

    if iterative_way == 8
        num_views = 4;
        measures{4} = measure_2.measure_4;
        M_ops{4} = @(x) measure_2.A_4(x); Mt_ops{4} = @(z) measure_2.At_4(z);
        measures{2} = rmfield(measures{2}, {'measure_4', 'A_4', 'At_4'});
    end

    denoi_ops = cell(1, num_views);
    for v = 1:num_views
        denoi_ops{v} = @(noisy, sigma) denoise(noisy, sigma, measures{v}.image_width, measures{v}.image_height, measures{v}.denoize_name);
    end

    if iscell(y), y_vec = y; m_len = length(y{1}); else, y_vec = y; m_len = length(y); end

    x_t = cell(num_views, 2);
    v_t = cell(num_views, 1);
    sigma_hat = zeros(1, num_views);
    eta = cell(1, num_views);
    gamma_part = cell(1, num_views);

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

    for i = 1:iters
        tic_iter = tic;

        gamma_total = 0;
        for v = 1:num_views
            diff = denoi_ops{v}(x_t{v, 1} + epsilon * eta{v}', sigma_hat(v)) - x_t{v, 2};
            m_curr = m_len;
            if iterative_way == 3 && iscell(y_vec), m_curr = length(y_vec{v}); end
            gamma_part{v} = 1 / (m_curr * epsilon) .* eta{v} * diff;
            gamma_total = gamma_total + gamma_part{v};
        end

        v_temp = y_vec;
        if iterative_way ~= 3
            for v = 1:num_views, v_temp = v_temp - (M_ops{v}(x_t{v, 2}))'; end
        end

        switch iterative_way
            case {1, 4, 8, 7}
                for v = 1:num_views
                    v_t{v} = gamma_total .* v_t{v} + Mt_ops{v}(v_temp);
                    x_t{v, 1} = x_t{v, 2} + alpha .* v_t{v};
                end
            case 5
                for v = 1:num_views
                    v_t{v} = gamma_part{v} .* v_t{v} + Mt_ops{v}(v_temp);
                    x_t{v, 1} = x_t{v, 2} + alpha .* v_t{v};
                end
            case 6
                for v = 1:num_views
                    v_t{v} = 1 .* v_t{v} + Mt_ops{v}(v_temp);
                    x_t{v, 1} = x_t{v, 2} + alpha .* v_t{v};
                end
            case 3
                for v = 1:num_views
                    v_loc = y_vec{v} - (M_ops{v}(x_t{v, 2}))';
                    v_t{v} = gamma_part{v} .* v_t{v} + Mt_ops{v}(v_loc);
                    x_t{v, 1} = x_t{v, 2} + alpha .* v_t{v};
                end
        end
        global_time = global_time + toc(tic_iter);

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

    x_hat_1 = reshape(x_t{1, 2}, [measure_1.image_height, measure_1.image_width]);
    if num_views >= 2, x_hat_2 = reshape(x_t{2, 2}, [measure_2.image_height, measure_2.image_width]); else, x_hat_2 = []; end

    % save outputs (keep behavior but make path configurable)
    if isfield(measure_1, 'results_root')
        save_root = measure_1.results_root;
    else
        save_root = '../results';
    end

    if num_views >= 3
        x_hat_3 = reshape(x_t{3, 2}, [measures{3}.image_height, measures{3}.image_width]);
        save(fullfile(save_root, 'par_three_pic.mat'), 'x_hat_3');
    end
    if num_views >= 4
        x_hat_4 = reshape(x_t{4, 2}, [measures{4}.image_height, measures{4}.image_width]);
        save(fullfile(save_root, 'par_four_pic.mat'), 'x_hat_4');
    end

    try
        sum_curve = zeros(iters, 1);
        for v = 1:num_views
            sum_curve = sum_curve + PSNR_sum(:, col_map(v));
        end
        avg_curve = sum_curve / num_views;

        if isfield(measure_1, 'Test_set_name')
            set_name = measure_1.Test_set_name;
        else
            set_name = 'Convergence_Results';
        end

        out_dir = fullfile(save_root, set_name, 'Convergence_Data');
        if ~exist(out_dir, 'dir'), mkdir(out_dir); end

        csv_file = fullfile(out_dir, [measure_1.denoize_name, '_Mode', num2str(iterative_way), '.csv']);

        fp = fopen(csv_file, 'a');
        if fp == -1
            warning('CSV write skipped: cannot open %s', csv_file);
            return;
        end

        if ftell(fp) == 0
            header = 'Image_Name,Actual_Rate';
            for k = 1:iters
                header = [header, sprintf(',Iter_%d', k)];
            end
            fprintf(fp, '%s\n', header);
        end

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