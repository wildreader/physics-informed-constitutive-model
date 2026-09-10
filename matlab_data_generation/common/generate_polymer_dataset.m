function output_file = generate_polymer_dataset(dataset_kind, architecture, make_plots)
%GENERATE_POLYMER_DATASET 生成与同一模型的三个顺序训练阶段匹配的数据集。
%
% output_file = generate_polymer_dataset(dataset_kind, architecture, make_plots)
%
% dataset_kind:
%   'train'  - 七工况训练集
%   'test'   - 五工况预测集
%
% architecture:
%   'free_energy'         - 无损伤、无体积支路
%   'free_energy_damage'  - 有损伤、无体积支路
%   'full'                - 有损伤、有体积支路
%
% 三种数据集使用相同的字段接口，使 Python 可以共用同一数据加载器。
% 未使用的字段仍然保留：例如 free_energy 数据仍保存历史最大伸长，
% 但该字段不会被对应的神经网络使用。

    if nargin < 3
        make_plots = true;
    end
    if nargin < 2
        error('必须指定 dataset_kind 和 architecture。');
    end

    dataset_kind = lower(strtrim(char(dataset_kind)));
    architecture = lower(strtrim(char(architecture)));
    valid_kinds = {'train', 'test'};
    valid_architectures = {'free_energy', 'free_energy_damage', 'full'};
    if ~any(strcmp(dataset_kind, valid_kinds))
        error('未知数据集类型：%s。可选 train 或 test。', dataset_kind);
    end
    if ~any(strcmp(architecture, valid_architectures))
        error(['未知模型架构：%s。可选 free_energy、', ...
               'free_energy_damage 或 full。'], architecture);
    end

    use_damage = ~strcmp(architecture, 'free_energy');
    use_volumetric = strcmp(architecture, 'full');

    % ================= 1. 公共参数 =================
    N_sph = 110;
    step_landa = 0.01;
    alpha = 0.5;

    % 前两阶段严格等体积：F=F_bar、J=1；只有 full 才加入 F_vol。
    % delta_J_ref 在所有 MAT 中保留，保证模型继承时参考尺度不变。
    lambda_J_ref = 9.0;
    delta_J_ref = 1.0e-4;
    k_vol = 1.0e6;

    common_dir = fileparts(mfilename('fullpath'));
    dataset_root = fileparts(common_dir);
    output_dir = fullfile(dataset_root, 'generated', architecture);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    lebedev = getLebedevSphere(N_sph);
    Dir = [lebedev.x(:), lebedev.y(:), lebedev.z(:)];
    Wt = lebedev.w(:) / (4*pi);
    if abs(sum(Wt) - 1.0) > 1.0e-10
        error('Lebedev 权重归一化失败：sum(Wt)=%.16g。', sum(Wt));
    end

    if strcmp(dataset_kind, 'train')
        [payload, plot_info] = generate_train_payload( ...
            architecture, use_damage, use_volumetric, ...
            Dir, Wt, step_landa, alpha, ...
            lambda_J_ref, delta_J_ref, k_vol);
        output_file = fullfile(output_dir, 'train_7cases.mat');
    else
        [payload, plot_info] = generate_test_payload( ...
            architecture, use_damage, use_volumetric, ...
            Dir, Wt, step_landa, alpha, ...
            lambda_J_ref, delta_J_ref, k_vol);
        output_file = fullfile(output_dir, 'test_5cases.mat');
    end

    save(output_file, '-struct', 'payload');
    print_dataset_summary(payload, dataset_kind, architecture, output_file);
    if make_plots
        plot_dataset_response(plot_info, dataset_kind, architecture);
    end
end


function [payload, plot_info] = generate_train_payload( ...
    architecture, use_damage, use_volumetric, ...
    Dir, Wt, step_landa, alpha, ...
    lambda_J_ref, delta_J_ref, k_vol)

    % 前六种训练工况：1->5->1->7->1->9->1。
    landa = [1:step_landa:5, 5-step_landa:-step_landa:1, ...
             1+step_landa:step_landa:7, 7-step_landa:-step_landa:1, ...
             1+step_landa:step_landa:9, 9-step_landa:-step_landa:1]';
    length_landa = length(landa);

    % 第七种工况：Z 预拉伸循环，然后进行 X 正式加载。
    landa_preZ = [1:step_landa:9, 9-step_landa:-step_landa:1]';
    landa_trainX = landa;
    landa_trainX_seamless = landa_trainX(2:end);
    len_Z = length(landa_preZ);

    results = cell(1, 7);
    for load_idx = 1:7
        if load_idx <= 6
            F_bar = build_isochoric_F_train(landa, load_idx);
            lambda_driver = landa;
        else
            F_bar = build_preZ_then_X_F( ...
                landa_preZ, landa_trainX_seamless);
            lambda_driver = [landa_preZ; landa_trainX_seamless];
        end

        results{load_idx} = evaluate_case( ...
            F_bar, lambda_driver, load_idx, ...
            use_damage, use_volumetric, Dir, Wt, alpha, ...
            lambda_J_ref, delta_J_ref, k_vol);
    end

    assembled = assemble_results(results);
    payload = struct();
    payload.X_La_curr_export = assembled.lambda_current;
    payload.X_La_max_export = assembled.lambda_maximum;
    payload.X_J_export = assembled.j_ratio;
    payload.X_C_export = assembled.c_vector;
    payload.X_F_export = assembled.f_vector;
    payload.Y_Stotal_export = assembled.s_total;
    payload.Y_p_export = assembled.pressure;
    payload.Y_sigma_total_export = assembled.sigma_total;
    payload.Dir = Dir;
    payload.Wt = Wt;
    payload.landa = landa;
    payload.J_path = results{1}.j_path;
    payload.length_landa = length_landa;
    payload.Train_Case_Lengths = assembled.case_lengths;
    payload.landa_preZ = landa_preZ;
    payload.landa_trainX = landa_trainX;
    payload.J_path_case7 = results{7}.j_path;
    payload.len_Z = len_Z;
    payload.lambda_J_ref = lambda_J_ref;
    payload.delta_J_ref = delta_J_ref;
    if use_volumetric
        payload.k_vol = k_vol;
    end
    payload.alpha = alpha;
    payload.model_architecture = architecture;
    payload.use_damage = use_damage;
    payload.use_volumetric = use_volumetric;
    payload.dataset_format_version = 2;

    plot_info = struct();
    plot_info.sigma_total = assembled.sigma_total;
    plot_info.case_lengths = assembled.case_lengths;
    plot_info.standard_stretch = landa;
    plot_info.special_stretch = landa_trainX;
    plot_info.special_case_index = 7;
    plot_info.special_start = len_Z;
    plot_info.load_names = { ...
        'Uniaxial X', 'Pure Shear X', 'Equibiaxial XY', ...
        'Uniaxial Y', 'Proportional Biaxial', ...
        'Rotated Uniaxial 45', 'Z Pre-stretch then X'};
end


function [payload, plot_info] = generate_test_payload( ...
    architecture, use_damage, use_volumetric, ...
    Dir, Wt, step_landa, alpha, ...
    lambda_J_ref, delta_J_ref, k_vol)

    % 第 1、2、3、5 种测试工况：1->4->1->6->1->8->1。
    landa_test1 = [1:step_landa:4, 4-step_landa:-step_landa:1, ...
                   1+step_landa:step_landa:6, 6-step_landa:-step_landa:1, ...
                   1+step_landa:step_landa:8, 8-step_landa:-step_landa:1]';

    % 第四种工况：Y 预拉伸循环，然后进行 X 正式加载。
    landa_preY = [1:step_landa:8, 8-step_landa:-step_landa:1]';
    landa_testX = landa_test1;
    landa_testX_seamless = landa_testX(2:end);
    len_Y = length(landa_preY);

    results = cell(1, 5);
    for load_idx = 1:5
        if load_idx ~= 4
            F_bar = build_isochoric_F_test(landa_test1, load_idx);
            lambda_driver = landa_test1;
        else
            F_bar = build_preY_then_X_F( ...
                landa_preY, landa_testX_seamless);
            lambda_driver = [landa_preY; landa_testX_seamless];
        end

        results{load_idx} = evaluate_case( ...
            F_bar, lambda_driver, load_idx, ...
            use_damage, use_volumetric, Dir, Wt, alpha, ...
            lambda_J_ref, delta_J_ref, k_vol);
    end

    assembled = assemble_results(results);
    payload = struct();
    payload.X_La_curr_test = assembled.lambda_current;
    payload.X_La_max_test = assembled.lambda_maximum;
    payload.X_J_test = assembled.j_ratio;
    payload.X_C_test = assembled.c_vector;
    payload.X_F_test = assembled.f_vector;
    payload.Y_Stotal_test = assembled.s_total;
    payload.Y_p_test = assembled.pressure;
    payload.Y_sigma_total_test = assembled.sigma_total;
    payload.Case_Lengths = assembled.case_lengths;
    payload.landa_test1 = landa_test1;
    payload.landa_preY = landa_preY;
    payload.landa_testX = landa_testX;
    payload.J_path_test1 = results{1}.j_path;
    payload.J_path_case4 = results{4}.j_path;
    payload.len_Y = len_Y;
    payload.lambda_J_ref = lambda_J_ref;
    payload.delta_J_ref = delta_J_ref;
    if use_volumetric
        payload.k_vol = k_vol;
    end
    payload.alpha = alpha;
    payload.model_architecture = architecture;
    payload.use_damage = use_damage;
    payload.use_volumetric = use_volumetric;
    payload.dataset_format_version = 2;

    plot_info = struct();
    plot_info.sigma_total = assembled.sigma_total;
    plot_info.case_lengths = assembled.case_lengths;
    plot_info.standard_stretch = landa_test1;
    plot_info.special_stretch = landa_testX;
    plot_info.special_case_index = 4;
    plot_info.special_start = len_Y;
    plot_info.load_names = { ...
        'Uniaxial Z', 'Equibiaxial YZ', 'Pure Shear Y', ...
        'Y Pre-stretch then X', ...
        'Proportional Biaxial YZ (1:0.5)'};
end


function result = evaluate_case( ...
    F_bar, lambda_driver, case_index, ...
    use_damage, use_volumetric, Dir, Wt, alpha, ...
    lambda_J_ref, delta_J_ref, k_vol)

    det_F_bar = determinant_pages(F_bar);
    max_iso_error = max(abs(reshape(det_F_bar, [], 1) - 1.0));
    if max_iso_error > 1.0e-10
        error('工况 %d 的 det(F_bar) 偏离 1，最大误差为 %.3e。', ...
              case_index, max_iso_error);
    end

    if use_volumetric
        J_path = build_J_path(lambda_driver, lambda_J_ref, delta_J_ref);
        F_vol = build_volumetric_F(J_path);
        F_dfm = pagemtimes(F_bar, F_vol);
        J = determinant_pages(F_dfm);
    else
        % 不调用体积变形公式，直接生成等体积路径。
        J_path = ones(size(lambda_driver));
        F_dfm = F_bar;
        J = ones(1, 1, size(F_bar, 3));
    end
    max_J_error = max(abs(reshape(J, [], 1) - J_path));
    if max_J_error > 1.0e-10
        error('工况 %d 的 det(F) 与指定 J(t) 不一致，最大误差为 %.3e。', ...
              case_index, max_J_error);
    end

    C_bar = pagemtimes(F_bar, 'transpose', F_bar, 'none');
    C_dfm = pagemtimes(F_dfm, 'transpose', F_dfm, 'none');
    if use_volumetric
        p = k_vol * (J - 1.0);
    else
        p = zeros(size(J));
    end

    [S_bar, La_Dir, La_Dir_max] = sphere_integration_matrix( ...
        C_bar, Dir, Wt, alpha, use_damage);
    S_Total = constitutive_model_pages(C_dfm, S_bar, p, J, use_volumetric);
    sigma_Total = pk2_to_cauchy_pages(F_dfm, S_Total, J);

    if ~use_volumetric
        if any(reshape(p, [], 1) ~= 0.0)
            error('无体积分支数据中出现了非零压力。');
        end
    end

    result = struct();
    result.lambda_current = La_Dir';
    result.lambda_maximum = La_Dir_max';
    result.j_ratio = reshape(J, [], 1);
    result.c_vector = tensor_pages_to_vec(C_dfm)';
    result.f_vector = tensor_pages_to_full_vec(F_dfm)';
    result.s_total = tensor_pages_to_vec(S_Total)';
    result.pressure = reshape(p, [], 1);
    result.sigma_total = tensor_pages_to_vec(sigma_Total)';
    result.j_path = J_path;
end


function assembled = assemble_results(results)
    case_count = numel(results);
    current_cells = cell(1, case_count);
    maximum_cells = cell(1, case_count);
    j_cells = cell(1, case_count);
    c_cells = cell(1, case_count);
    f_cells = cell(1, case_count);
    s_cells = cell(1, case_count);
    p_cells = cell(1, case_count);
    sigma_cells = cell(1, case_count);
    case_lengths = zeros(1, case_count);

    for idx = 1:case_count
        current_cells{idx} = results{idx}.lambda_current;
        maximum_cells{idx} = results{idx}.lambda_maximum;
        j_cells{idx} = results{idx}.j_ratio;
        c_cells{idx} = results{idx}.c_vector;
        f_cells{idx} = results{idx}.f_vector;
        s_cells{idx} = results{idx}.s_total;
        p_cells{idx} = results{idx}.pressure;
        sigma_cells{idx} = results{idx}.sigma_total;
        case_lengths(idx) = size(results{idx}.lambda_current, 1);
    end

    assembled = struct();
    assembled.lambda_current = vertcat(current_cells{:});
    assembled.lambda_maximum = vertcat(maximum_cells{:});
    assembled.j_ratio = vertcat(j_cells{:});
    assembled.c_vector = vertcat(c_cells{:});
    assembled.f_vector = vertcat(f_cells{:});
    assembled.s_total = vertcat(s_cells{:});
    assembled.pressure = vertcat(p_cells{:});
    assembled.sigma_total = vertcat(sigma_cells{:});
    assembled.case_lengths = case_lengths;
end


function J_path = build_J_path(lambda_driver, lambda_ref, delta_J_ref)
    if lambda_ref <= 1.0
        error('lambda_ref 必须大于 1。');
    end
    if delta_J_ref <= -1.0
        error('delta_J_ref 必须保证 J>0。');
    end

    xi = (lambda_driver - 1.0) / (lambda_ref - 1.0);
    if any(xi < -1.0e-12) || any(xi > 1.0 + 1.0e-12)
        error('lambda_driver 超出了 [1, lambda_ref] 范围。');
    end
    J_path = 1.0 + delta_J_ref * xi;
end


function F_bar = build_isochoric_F_train(lambda, load_idx)
    N = length(lambda);
    F_bar = zeros(3, 3, N);

    switch load_idx
        case 1
            F_bar(1,1,:) = lambda;
            F_bar(2,2,:) = lambda.^(-1/2);
            F_bar(3,3,:) = lambda.^(-1/2);
        case 2
            F_bar(1,1,:) = lambda;
            F_bar(2,2,:) = 1.0;
            F_bar(3,3,:) = lambda.^(-1);
        case 3
            F_bar(1,1,:) = lambda;
            F_bar(2,2,:) = lambda;
            F_bar(3,3,:) = lambda.^(-2);
        case 4
            F_bar(1,1,:) = lambda.^(-1/2);
            F_bar(2,2,:) = lambda;
            F_bar(3,3,:) = lambda.^(-1/2);
        case 5
            lambda_2 = 1.0 + 0.5*(lambda - 1.0);
            F_bar(1,1,:) = lambda;
            F_bar(2,2,:) = lambda_2;
            F_bar(3,3,:) = 1.0 ./ (lambda .* lambda_2);
        case 6
            theta = deg2rad(45);
            c = cos(theta);
            s = sin(theta);
            lambda_p = lambda;
            lambda_t = lambda.^(-1/2);
            F_bar(1,1,:) = lambda_p*c^2 + lambda_t*s^2;
            F_bar(2,2,:) = lambda_p*s^2 + lambda_t*c^2;
            F_bar(3,3,:) = lambda_t;
            F_bar(1,2,:) = (lambda_p - lambda_t)*s*c;
            F_bar(2,1,:) = (lambda_p - lambda_t)*s*c;
        otherwise
            error('未知训练工况编号：%d。', load_idx);
    end
end


function F_bar = build_isochoric_F_test(lambda, load_idx)
    N = length(lambda);
    F_bar = zeros(3, 3, N);

    switch load_idx
        case 1
            F_bar(1,1,:) = lambda.^(-1/2);
            F_bar(2,2,:) = lambda.^(-1/2);
            F_bar(3,3,:) = lambda;
        case 2
            F_bar(1,1,:) = lambda.^(-2);
            F_bar(2,2,:) = lambda;
            F_bar(3,3,:) = lambda;
        case 3
            F_bar(1,1,:) = 1.0;
            F_bar(2,2,:) = lambda;
            F_bar(3,3,:) = lambda.^(-1);
        case 5
            lambda_Y = lambda;
            lambda_Z = 1.0 + 0.5*(lambda - 1.0);
            F_bar(1,1,:) = 1.0 ./ (lambda_Y .* lambda_Z);
            F_bar(2,2,:) = lambda_Y;
            F_bar(3,3,:) = lambda_Z;
        otherwise
            error('未知测试工况编号：%d。', load_idx);
    end
end


function F_bar = build_preZ_then_X_F(lambda_preZ, lambda_X_seamless)
    len_Z = length(lambda_preZ);
    len_X = length(lambda_X_seamless);
    F_bar = zeros(3, 3, len_Z + len_X);

    F_bar(1,1,1:len_Z) = lambda_preZ.^(-1/2);
    F_bar(2,2,1:len_Z) = lambda_preZ.^(-1/2);
    F_bar(3,3,1:len_Z) = lambda_preZ;

    range_X = len_Z+1 : len_Z+len_X;
    F_bar(1,1,range_X) = lambda_X_seamless;
    F_bar(2,2,range_X) = lambda_X_seamless.^(-1/2);
    F_bar(3,3,range_X) = lambda_X_seamless.^(-1/2);
end


function F_bar = build_preY_then_X_F(lambda_preY, lambda_X_seamless)
    len_Y = length(lambda_preY);
    len_X = length(lambda_X_seamless);
    F_bar = zeros(3, 3, len_Y + len_X);

    F_bar(1,1,1:len_Y) = lambda_preY.^(-1/2);
    F_bar(2,2,1:len_Y) = lambda_preY;
    F_bar(3,3,1:len_Y) = lambda_preY.^(-1/2);

    range_X = len_Y+1 : len_Y+len_X;
    F_bar(1,1,range_X) = lambda_X_seamless;
    F_bar(2,2,range_X) = lambda_X_seamless.^(-1/2);
    F_bar(3,3,range_X) = lambda_X_seamless.^(-1/2);
end


function F_vol = build_volumetric_F(J_path)
    N = length(J_path);
    volume_stretch = reshape(J_path.^(1/3), 1, 1, N);
    F_vol = zeros(3, 3, N);
    F_vol(1,1,:) = volume_stretch;
    F_vol(2,2,:) = volume_stretch;
    F_vol(3,3,:) = volume_stretch;
end


function J = determinant_pages(F)
    J = F(1,1,:).*(F(2,2,:).*F(3,3,:) - F(2,3,:).*F(3,2,:)) ...
      - F(1,2,:).*(F(2,1,:).*F(3,3,:) - F(2,3,:).*F(3,1,:)) ...
      + F(1,3,:).*(F(2,1,:).*F(3,2,:) - F(2,2,:).*F(3,1,:));
end


function vec = tensor_pages_to_vec(T)
    vec = [squeeze(T(1,1,:))'; squeeze(T(2,2,:))'; ...
           squeeze(T(3,3,:))'; squeeze(T(1,2,:))'; ...
           squeeze(T(1,3,:))'; squeeze(T(2,3,:))'];
end


function vec = tensor_pages_to_full_vec(T)
    vec = [squeeze(T(1,1,:))'; squeeze(T(1,2,:))'; ...
           squeeze(T(1,3,:))'; squeeze(T(2,1,:))'; ...
           squeeze(T(2,2,:))'; squeeze(T(2,3,:))'; ...
           squeeze(T(3,1,:))'; squeeze(T(3,2,:))'; ...
           squeeze(T(3,3,:))'];
end


function [S_bar, La_Dir, La_Dir_max] = sphere_integration_matrix( ...
    C_bar, Dir, Wt, alpha, use_damage)

    N_sph = size(Dir, 1);
    sample_count = size(C_bar, 3);
    La_Dir = zeros(N_sph, sample_count);
    La_Dir_max = zeros(N_sph, sample_count);
    S_bar = zeros(3, 3, sample_count);

    for k = 1:sample_count
        C = C_bar(:,:,k);
        La_Dir(:,k) = sqrt(sum(Dir*C .* Dir, 2));

        LS = La_Dir(:,k);
        force_chain = (max(LS, 1.0) - 1.0).^3;
        if use_damage
            if k == 1
                La_Dir_max(:,k) = max(1.0, LS);
            else
                La_Dir_max(:,k) = max(La_Dir_max(:,k-1), LS);
            end
            LS_max = La_Dir_max(:,k);
            % 当前 MATLAB 公式得到的是剩余刚度/能量系数 (1-d)。
            local_survival = max((12.0 - max(LS_max, 1.0))/12.0, 0.0);
            average_survival = sum(Wt .* local_survival);
            survival_factor = (1.0 - alpha)*local_survival ...
                            + alpha*average_survival;
            Fc = survival_factor .* force_chain;
        else
            % 无损伤：不计算历史、D_local/D_avg/Dmg_chain，不乘损伤系数。
            La_Dir_max(:,k) = 1.0; % 仅保留统一 MAT 接口的占位字段。
            Fc = force_chain;
        end

        weighted_force = Wt .* Fc ./ LS;
        S_bar(:,:,k) = Dir' * (weighted_force .* Dir);
    end
end


function S_all = constitutive_model_pages(C_all, Sbar_all, p_all, J_all, use_volumetric)
    sample_count = size(C_all, 3);
    S_all = zeros(3, 3, sample_count);

    for k = 1:sample_count
        C = C_all(:,:,k);
        Sbar = Sbar_all(:,:,k);
        J = J_all(1,1,k);
        Cinv = C \ eye(3);
        CSbar = sum(sum(C .* Sbar));
        DevSbar = Sbar - (1.0/3.0)*CSbar*Cinv;
        S_iso = J^(-2/3) * DevSbar;
        if use_volumetric
            p = p_all(1,1,k);
            S_vol = J*p*Cinv;
            S_all(:,:,k) = S_iso + S_vol;
        else
            S_all(:,:,k) = S_iso;
        end
    end
end


function sigma_all = pk2_to_cauchy_pages(F_all, S_all, J_all)
    sample_count = size(F_all, 3);
    sigma_all = zeros(3, 3, sample_count);
    for k = 1:sample_count
        F = F_all(:,:,k);
        S = S_all(:,:,k);
        J = J_all(1,1,k);
        sigma_all(:,:,k) = (F*S*F') / J;
    end
end


function print_dataset_summary(payload, dataset_kind, architecture, output_file)
    fprintf('\n============================================================\n');
    fprintf('数据集生成完成\n');
    fprintf('模型架构: %s\n', architecture);
    fprintf('数据集类型: %s\n', dataset_kind);
    fprintf('保存路径: %s\n', output_file);

    if strcmp(dataset_kind, 'train')
        fprintf('X_La_curr_export      : [%d, %d]\n', ...
            size(payload.X_La_curr_export));
        fprintf('X_La_max_export       : [%d, %d]\n', ...
            size(payload.X_La_max_export));
        fprintf('X_J_export            : [%d, %d]\n', ...
            size(payload.X_J_export));
        fprintf('X_C_export            : [%d, %d]\n', ...
            size(payload.X_C_export));
        fprintf('X_F_export            : [%d, %d]\n', ...
            size(payload.X_F_export));
        fprintf('Y_sigma_total_export  : [%d, %d]\n', ...
            size(payload.Y_sigma_total_export));
        J_values = payload.X_J_export;
        p_values = payload.Y_p_export;
    else
        fprintf('X_La_curr_test        : [%d, %d]\n', ...
            size(payload.X_La_curr_test));
        fprintf('X_La_max_test         : [%d, %d]\n', ...
            size(payload.X_La_max_test));
        fprintf('X_J_test              : [%d, %d]\n', ...
            size(payload.X_J_test));
        fprintf('X_C_test              : [%d, %d]\n', ...
            size(payload.X_C_test));
        fprintf('X_F_test              : [%d, %d]\n', ...
            size(payload.X_F_test));
        fprintf('Y_sigma_total_test    : [%d, %d]\n', ...
            size(payload.Y_sigma_total_test));
        J_values = payload.X_J_test;
        p_values = payload.Y_p_test;
    end
    fprintf('J 范围                : [%.8f, %.8f]\n', ...
        min(J_values), max(J_values));
    fprintf('p 范围                : [%.8f, %.8f]\n', ...
        min(p_values), max(p_values));
    fprintf('============================================================\n');
end


function plot_dataset_response(plot_info, dataset_kind, architecture)
    component_names = {'\sigma_{11}', '\sigma_{22}', '\sigma_{33}', ...
                       '\sigma_{12}', '\sigma_{13}', '\sigma_{23}'};
    start_idx = 1;
    for idx = 1:numel(plot_info.case_lengths)
        end_idx = start_idx + plot_info.case_lengths(idx) - 1;
        sigma_case = plot_info.sigma_total(start_idx:end_idx, :);

        if idx == plot_info.special_case_index
            x_axis = plot_info.special_stretch;
            sigma_plot = sigma_case(plot_info.special_start:end, :);
        else
            x_axis = plot_info.standard_stretch;
            sigma_plot = sigma_case;
        end

        figure('Name', plot_info.load_names{idx}, ...
               'Position', [100, 100, 1200, 600], 'Color', 'w');
        for component_idx = 1:6
            subplot(2, 3, component_idx);
            plot(x_axis, sigma_plot(:, component_idx), ...
                 'r-', 'LineWidth', 1);
            title(component_names{component_idx}, ...
                  'FontSize', 12, 'FontWeight', 'bold');
            xlabel('Stretch ratio \lambda');
            ylabel('Cauchy stress');
            grid on;
        end
        sgtitle(sprintf('%s %s Case %d: %s', ...
            architecture, dataset_kind, idx, plot_info.load_names{idx}), ...
            'FontSize', 16, 'FontWeight', 'bold');
        start_idx = end_idx + 1;
    end
end
