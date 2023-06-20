% script to reconstruct simulate US data from random scenarios
% see simulate_full_data.m for the generation
%
% author: Felix Lucka
% date:        21.10.2021
% last update: 21.10.2021


clear
clc
%close all

startup

% set seed of random number generator
rng(1)

%% define all settings that will determine the type of simulation data used

%%% data simulation
n_ele        = 128;  % number of elements
pitch        = 3e-4; % pitch between elements
absorption   = 0; % [dB/(MHz^y cm)] acoustic absorption
i_sce        = 1; % number of random scenarios to reconstruct

% size of the computational domain used in forward computation in relation to the size of the transducer
x_fac_fwd        = 1.2; % x is lateral dimension, i.e., parallel to transducer
z_fac_fwd        = 1.6; % z is depth   dimension, i.e., perpendicular to transducer
pix_per_ele_fwd  = 4;  % number of pixel used to discretize the pitch (length between two transducer elements)

%%% ways to modulated original "perfect" data
source_modulation = 'none'; % 'none' or 'toneBurst'
tone_burst_freq   = 5e6; % [Hz]
tone_burst_cycles = 1;

snr           = inf; % signal-to-noise ratio at which to add noise
% add: band limitations by sensors
suppress_direct_signal = true;

%%% reconstruction techniques and setting of inverse computation grid
inv_method       = 'DAS'; % DAS or FKMig
% for FKMig, the data will be converted to plane wave data  
n_ang        = 1; % number of plane wave angles (choose odd to have 0 in set)
max_ang      = 16; % maximal steering angle

x_fac_inv        = 1.2;  % x is lateral dimension, i.e., parallel to transducer
z_fac_inv        = 1.6;% z is depth   dimension, i.e., perpendicular to transducer
pix_per_ele_inv  = 1;  % number of pixel used to discretize the pitch (length between two transducer elements)
dz_fac           = 1;  % upsampling in depth dimension

overwrite    = false;  % overwrite results that already exist for this scenario?
analyse_data = true;

%% generate a folder for the simulated data from the chosen scenario

data_name = ['Ne' int2str(n_ele) 'PPE' int2str(pix_per_ele_fwd) 'Pi' num2str(pitch, '%.2e')];
data_name = [data_name 'x' num2str(x_fac_fwd) 'z' num2str(z_fac_fwd)];
if(absorption > 0)
    data_name = [data_name 'A' num2str(absorption)];
end
full_storage_path = [storage_path 'SimulatedData' fs  data_name];
data_filename     = [full_storage_path fs 'scenario_' int2str(i_sce) '.mat'];
data_bgn_filename = [full_storage_path fs 'scenario_0.mat'];

%% load and pre-process the data

load(data_filename)

sos_bgn       = 1540; % [m/s] speed of sound
rho_bgn       =  985; % [kg/m^3] mass density
dt            = t_vec(2) - t_vec(1);
sampling_freq = 1/dt; % sampling frequency of the data

if(analyse_data)
    % compute power spectrum statistics of data
    [~, freq, Fs, power_spectrum_stats{1}] = spectralAnalysis(data, dt, 'decibel');
    scale = 1/1e6; 
    figure(123);
    plot(scale * freq, power_spectrum_stats{1}.mean); hold on
    grid on
    xlabel('Frequency  MHz');
    ylabel('Power Spectrum [dB]');
end

switch source_modulation
    case 'toneBurst'
        % the orginal data is a delta peak at time onset, low-pass filtered
        % to remove frequencies not supported by the k-Wave grid
        % we can generate tone burst signals by convolving in time
        new_swf = toneBurst(sampling_freq, tone_burst_freq, tone_burst_cycles);
        data    = convn(data, new_swf(:));
        source_wave_form = convn(source_wave_form(:), new_swf(:));
    case 'none'
        % nothing to do here
end
n_t           = size(data, 1);
n_ele         = size(data, 2);

% suppress direct signal
if(suppress_direct_signal)
    dist = (0:(n_ele-1)) * pitch;
    dist = abs(dist' - dist);
    data = suppressDirectSignal(data, dist, sos_bgn-100, dt, source_wave_form);
end

% pre-processing specific for the inverse method used
switch inv_method
    case 'DAS'

    case 'FKMig'
        % synthesise multi-source activation
        if(n_ang > 1)
            angles = linspace(-max_ang, max_ang, n_ang);
        else
            angles = 0;
        end
        data_PW  = zeros(n_t, n_ele, n_ang);
        % create an element index relative to the centre element of the transducer
        ele_ind = -(n_ele - 1)/2:(n_ele - 1)/2;
        
        for i_ang = 1:n_ang
            % use geometric beam forming to calculate the tone burst offsets for each
            % transducer element based on the element index
            ele_offset =   pitch * ele_ind * sin(angles(i_ang) * pi/180) / (sos_bgn * dt);
            % modify such that all delays are positive
            ele_offset = ele_offset - min(ele_offset);
            for i_ele = 1:n_ele
                % delay time series by interpolation
                delayed_data_i = data(:, :, i_ele);
                delayed_data_i = interp1((1:n_t), delayed_data_i, (1:n_t) - ele_offset(i_ele), 'linear', 0);
                % add
                data_PW(:,:,i_ang) = data_PW(:,:,i_ang) + delayed_data_i;
            end
        end
        data = data_PW; clear data_PW
end

% filter to mimic band-limited sensors

% add noise
if(~isinf(snr))
    
    % standard deviation is product of RMS
    sigma = sqrt(mean(data(:).^2)) * 10^(-snr/20);
    data  = data + sigma * randn(size(data));
    
end

if(analyse_data)
    % compute power spectrum statistics of data
    [~, freq, Fs, power_spectrum_stats{1}] = spectralAnalysis(data, dt, 'decibel');
    scale = 1/1e6; 
    figure(123);
    plot(scale * freq, power_spectrum_stats{1}.mean); hold on
    grid on
    xlabel('Frequency  MHz');
    ylabel('Power Spectrum [dB]');
end


%% setting for inverse reconstruction
% z = 0 is the vertical position of the transducer
% x = 0 is in the middle of the transducer

% set up computational grid for inverse computation
dx           = pitch / pix_per_ele_inv;
dz           = dx / dz_fac;
length_array = (n_ele - 1) * pitch;
Nx           = ceil(x_fac_inv * length_array/dx);
Nz           = ceil(z_fac_inv * length_array/dz);
Nzx          = [Nz, Nx];

x_vec_inv    = (1:Nx) * dx;
x_vec_inv    = x_vec_inv - mean(x_vec_inv);
z_vec_inv    = (0:(Nz-1)) * dz;
[Z, X]       = ndgrid(z_vec_inv, x_vec_inv);
x_ele_inv    = (0:(n_ele-1)) * pitch;
x_ele_inv    = x_ele_inv - mean(x_ele_inv);
z_ele_inv    = zeros(n_ele, 1);

%%

[~,latency_ind] = max(abs(source_wave_form));
        
switch inv_method
    
    case 'DAS'
        
        time_ind_matrix  = timeIndexFunction(sos_bgn, latency_ind/2 * dt, sampling_freq, x_ele_inv, z_ele_inv, X, Z);
        img              = delayAndSum(data, time_ind_matrix);
        
    case 'FKMig'
        
        % clip off start
        data = data(latency_ind:end,:,:);
        
        fk_para          = [];
        fk_para.TXangle  = deg2rad(angles(:));
        fk_para.pitch    = pitch;
        fk_para.t0       = 0;
        fk_para.c        = sos_bgn;
        fk_para.fs       = 1/dt;
        
        % call F-K migration
        [img, fk_para] = fkmig(data, fk_para);
        % hilbert transform along temporal dimension
        img = hilbert(img);
        img = abs(img);
        % interpolate onto inverse grid
        img = interpn(fk_para.z, fk_para.x - mean(fk_para.x) , img, z_vec_inv(:), x_vec_inv);
        
end

% post-processing

% absolut value 
img = abs(img);
% % SNR transform
img = 20*log10(img);
img = img - max(img(:), [], 'omitnan');
% clipping
img(img < -50) = -50;
img(img > 0) = 0;
% shift from negative values
img = (img + 50) / 50;
img(isnan(img)) = 1;

%% visualize

figure();
imagesc(z_vec_inv, x_vec_inv, img);
colormap(gray); axis square; xlabel('x'); ylabel('z');

% interpolate segmentation onto computational grid
[Z_fwd, X_fwd]      = ndgrid(z_vec, x_vec);
segmentation_interp = interpn(z_vec - z_ele(1), x_vec' - mean(x_ele), segmentation, z_vec_inv(:), x_vec_inv, 'nearest');

figure();
imagesc(segmentation_interp);
colormap(gray); axis square; xlabel('x'); ylabel('z');

