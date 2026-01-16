clear;
close;

% Geometry
voxel_size = 0.2; % mm
pixel_size = 0.2;
SOD_mm = 410;
ODD_mm = 210;
D = (SOD_mm + ODD_mm) / voxel_size; % distance from source to center in pixels

fanSensorSpacing = pixel_size / voxel_size;  % 1 pixel
num_proj = 180;
rotationIncrement = 360 / num_proj;

path = '../output/sino/';
files = dir(fullfile(path, '*.nii'));
filenames = {files.name}';
for i = 1:length(filenames)
    sin_img = double(niftiread(strcat(path, filenames{i})));
    % Reconstruct
    recon = ifanbeam(sin_img, D, ...
    'FanSensorSpacing', fanSensorSpacing, ...
    'FanSensorGeometry', 'line', ...
    'FanRotationIncrement', rotationIncrement, ...
    'Filter', 'Shepp-Logan', ...
    'FanCoverage', 'cycle'); 
    niftiwrite(recon, '../output/img/recon.nii')
end
