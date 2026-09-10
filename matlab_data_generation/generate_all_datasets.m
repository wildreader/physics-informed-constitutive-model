clear; clc; close all;

% 一次生成三种架构的七工况训练集和五工况预测集。
% 批量生成默认不绘图；各架构子文件夹中的单独入口会绘图。
dataset_root = fileparts(mfilename('fullpath'));
addpath(fullfile(dataset_root, 'common'));

architectures = {'free_energy', 'free_energy_damage', 'full'};
for idx = 1:numel(architectures)
    generate_polymer_dataset('train', architectures{idx}, false);
    generate_polymer_dataset('test', architectures{idx}, false);
end

disp('三种架构的全部训练/预测数据集已生成。');

