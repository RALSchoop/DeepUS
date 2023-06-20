% script to simulate US data from random scenarios
%
% author: Felix Lucka, Georgios Pilikos
% date:        19.10.2021
% last update: 29.10.2021


clear
clc
close all

% initializes the toolbox
startup

%% define all settings that will determine the type of simulation done here

n_ele        = 128;  % number of transducer elements
pitch        = 3e-4; % [mm] pitch (= distance) between elements
pix_per_ele  = 4;    % number of pixel used to discretize the pitch 
absorption   = 0;    % [dB/(MHz^y cm)] acoustic absorption

n_sce        = 100; % number of random scenarios to simulate

% size of the computational domain in relation to the size of the transducer
x_fac        = 1.2; % x is lateral dimension, i.e., parallel to transducer
z_fac        = 1.6; % z is depth   dimension, i.e., perpendicular to transducer

dryrun       = false; % don't simulate data, just check scenario generation
overwrite    = false;  % overwrite results that already exist for this scenario
code_version = 'CUDA'; % 'Matlab', 'C++' or 'CUDA'
use_gpu      = true;   % do computations on the GPU?

%% generate a folder for the simulated data from the chosen scenario

res_name = ['Ne' int2str(n_ele) 'PPE' int2str(pix_per_ele) 'Pi' num2str(pitch, '%.2e')];
res_name = [res_name 'x' num2str(x_fac) 'z' num2str(z_fac)];
if(absorption > 0)
    res_name = [res_name 'A' num2str(absorption)];
end
full_storage_path = [storage_path 'SimulatedData' fs  res_name]
makeDir(full_storage_path);

%% define basic characteristics of the random scenario generation

% background properties
sos_bgn       = 1540; % [m/s] speed of sound
rho_bgn       =  985; % [kg/m^3] mass density
sos_bgn_std   =   12; % [m/s] standard deviation of speed of sound in background
rho_bgn_std   =    8; % [kg/m^3] standard deviation of mass density in background

% define types of circular inclusions
regions = [];
% region 1: anechoic inclusions
regions(1).label       = 1; % label in the segmentation (0 is background)
regions(1).sos_mean    = sos_bgn; % [m/s] average speed of sound
regions(1).sos_std     =   0;     % [m/s] standard deviation of speed of sound
regions(1).rho_mean    = rho_bgn; % [kg/m^3] average mass density
regions(1).rho_std     =   0;     % [kg/m^3] standard deviation of mass density
regions(1).n_inc_mean  = 2; % average number of inclusions
regions(1).n_inc_min   = 1; % minimal number of inclusions
regions(1).inc_max_rad = 5e-3; % [m] maximal radium of inclusion
regions(1).inc_min_rad = 2e-3; % [m] minimal radium of inclusion
% region 2: highly scattering inclusions
regions(2).label       = 2;
regions(2).sos_mean    = 1565;
regions(2).sos_std     =   75;
regions(2).rho_mean    = 1043;
regions(2).rho_std     =   50;
regions(2).n_inc_mean  = 2;
regions(2).n_inc_min   = 1;
regions(2).inc_max_rad = 5e-3;
regions(2).inc_min_rad = 2e-3;
% region 3: nylon inclusions
regions(3).label       = 3;
regions(3).sos_mean    = 2620;
regions(3).sos_std     =   0;
regions(3).rho_mean    = 1140;
regions(3).rho_std     =   0;
regions(3).n_inc_mean  = 2;
regions(3).n_inc_min   = 1;
regions(3).inc_max_rad = 0.3e-3;
regions(3).inc_min_rad = 0.1e-3;

%% set up kWave simulation

if(~isEven(n_ele))
    error('n_ele needs to be even integer')
end

% create the computational grid; z is depth direction (perpendicular to
% transducer); x is lateral direction (parallel to transducer)
dx           = pitch / pix_per_ele; % grid size
length_array = (n_ele - 1) * pitch;
Nx           = intManipulation(x_fac * (length_array/dx + 1), 'evenUp'); % #pixels in vertical direction
Nz           = intManipulation(z_fac * (length_array/dx + 1), 'evenUp'); % #pixels in horizontal direction
Nzx          = [Nz, Nx];
kgrid        = kWaveGrid(Nz, dx, Nx, dx);
% make time based on min/max speeds of 1400 / 1700 m/s
kgrid.makeTime([1400, 1700]);
freq_max     = 1400 / (2 * dx); % maximal frequency that k-Wave propagates accurately (will be used for down-sampling)

% enlarge time array by 2 (enough time for a signal to travel the diagonal
% of the computational domain twice
kgrid.makeTime([1400, 1700], 0.3, 2 * kgrid.t_array(end));

% we will downsample the signal produced by k-Wave to 2*freq_max using a
% fourier interpolation technique
Nt_ds = ceil(2 * kgrid.t_array(end) / (1/freq_max));
dt_ds = kgrid.t_array(end)/(Nt_ds-1);
t_vec = (0:(Nt_ds-1)) * dt_ds;

% choose size of perfeclty matched layer automatically
pml_sz       = getOptimalPMLSize(Nzx, [20,100]);

% construct binary mask for the transducer
% start the with an offset of 1% of total depth
z_offset   = ceil(Nz * 0.01);
% place it in the center wrt lateral dimension.
x_offset   = max(1,ceil((Nx - (n_ele -1) * pix_per_ele)/2));
trs_mask   = zeros(Nzx);
trs_mask(z_offset, x_offset:pix_per_ele:(x_offset+(n_ele-1)*pix_per_ele)) = 1;
z_ele      = kgrid.x(trs_mask > 0);
x_ele      = kgrid.y(trs_mask > 0);


% construct source struct (mask will be constructed later)
source = [];
source.p_mask = trs_mask;

% we construct the source wave form as an approximation to a perfect
% "click". This involves filtering that introduces a delay
source_wave_form    = zeros(1, kgrid.Nt);
source_wave_form(1) = 1;
evalc('source_wave_form = filterTimeSeries(kgrid, struct(''sound_speed'', 1400), source_wave_form, ''ZeroPhase'', false);');

% define a sensor (same as source)
sensor      = [];
sensor.mask = trs_mask;

% define non-random part of medium
medium             = [];
if(absorption > 0)
    medium.alpha_coeff = absorption;
    medium.alpha_power = 1.5;
end

% additional arguments for k-Wave
if(use_gpu)
    data_cast  = 'gpuArray-single';
else
    data_cast  = 'single'; % 'double', 'single', 'gpuArray-double' or 'gpuArray-single'
end
kwave_args = {'PMLSize', pml_sz, 'PMLInside', false, 'DataCast', data_cast, ...
    'Smooth', true, 'PlotSim', false};

% grid vectors will be stored for later reference
z_vec        = kgrid.x_vec;
x_vec        = kgrid.y_vec;

%% generate random scenarios and simulate data

% loop over scenarios to simulate (0 = pure background)
for i_sce=0:n_sce
    
    % compose file name to store results
    res_filename = [full_storage_path fs 'scenario_' int2str(i_sce) '.mat'];
    
    cmp_sce = overwrite || ~exist(res_filename, 'file');
    
    if(cmp_sce) % we need to compute something
        
        % set random seed
        rng(i_sce);
        
        % initialize segmentation with background label
        segmentation = zeros(Nzx);
        
        % define background sos and density (random variations are coupled)
        bgd_rnd = randn(Nzx);
        bgd_rnd = min(max(bgd_rnd, -3), 3); % clip at 3 std deviations
        sos     = sos_bgn * ones(Nzx) + sos_bgn_std * bgd_rnd;
        rho     = rho_bgn * ones(Nzx) + rho_bgn_std * bgd_rnd;
        % reset sound speed and density up to transducers to baground values
        sos(1:z_offset, :) = sos_bgn;
        rho(1:z_offset, :) = rho_bgn;
        
        %%% loop over regions
        for i_reg = 1:length(regions)
            
            n_inc = poissrnd(regions(i_reg).n_inc_mean- regions(i_reg).n_inc_min) + regions(i_reg).n_inc_min;
            i_inc = 1;
            while (i_inc <= n_inc)
                % try to create a new inclusion
                inc_rad     =  rand() * (regions(i_reg).inc_max_rad - regions(i_reg).inc_min_rad) + regions(i_reg).inc_min_rad;
                inc_rad     =  max(inc_rad/dx, 1);
                dist_border =  ceil(inc_rad);
                cz          =  2*z_offset + dist_border;
                cz          =  cz + randi(Nz - (cz+dist_border), 1);
                cx          =  dist_border;
                cx          =  cx + randi(Nx - (cx+dist_border), 1);
                inc         =  makeDisc(Nz, Nx, cz, cx, inc_rad) > 0;
                if any(segmentation(inc)) % new inclusion overlaps with another one, try again
                    continue
                else
                    segmentation(inc) = regions(i_reg).label;
                    i_inc             = i_inc + 1;
                end
            end
            
            % assign sos and rho values
            region_mask      = segmentation == regions(i_reg).label;
            inc_rnd          = randn(nnz(region_mask), 1);
            inc_rnd          = min(max(inc_rnd, -3), 3); % clip at 3 std deviations
            sos(region_mask) = regions(i_reg).sos_mean + regions(i_reg).sos_std * inc_rnd;
            rho(region_mask) = regions(i_reg).rho_mean + regions(i_reg).rho_std * inc_rnd;
            
        end
        
        if(i_sce == 0)
            % scenario 0 is just  homogenous background
            sos(:)           = sos_bgn;
            rho(:)           = rho_bgn;
            segmentation(:)  = 0;
        end
        
        % construct medium struct for k-Wave
        medium.sound_speed = sos;
        medium.density     = rho;
                
        if(dryrun)
            
            % just plot scenario
            figure(1);
            subplot(1, 3, 1); imagesc(segmentation); axis ij
            subplot(1, 3, 2); imagesc(medium.sound_speed, [1450, 1650]); axis ij
            subplot(1, 3, 3); imagesc(medium.density); axis ij
            drawnow();
            pause(1);
            
        else
            
            fprintf(['simulating data for scenario ' int2str(i_sce)])
            data  = zeros(Nt_ds, n_ele, n_ele);
            
            clock_cmp = tic;
            for i_src = 1:n_ele
                
                
                % prepare single source activation
                source_i             = source;
                source_i.p           = zeros(n_ele, kgrid.Nt);
                source_i.p(i_src, :) = source_wave_form;
                % run k-Wave but suppress output
                switch code_version
                    case 'Matlab'
                        evalc('data_i_src = kspaceFirstOrder2D(kgrid, medium, source_i, sensor, kwave_args{:});');
                    case {'C++', 'CUDA'}
                        if(use_gpu)
                            evalc('data_i_src = kspaceFirstOrder2DG(kgrid, medium, source_i, sensor, kwave_args{:});');
                        else
                            evalc('data_i_src = kspaceFirstOrder2DC(kgrid, medium, source_i, sensor, kwave_args{:});');
                        end
                end
                
                data_i_src  = gather(data_i_src)';
                % downsample and add to data
                data(:, :, i_src)  = interpftn(data_i_src, [Nt_ds, n_ele]);
                
                fprintf('.')
                
            end
            disp(['done. Computation time:' convertSec(toc(clock_cmp))])
            
            % store simulated data
            res_filename = [full_storage_path fs 'scenario_' int2str(i_sce) '.mat'];
            save(res_filename, 'data', 'i_sce', 'medium', 'segmentation', 'trs_mask', ...
                'source_wave_form', 'regions', 'x_vec', 'z_vec', 't_vec', 'kwave_args',...
                'x_ele', 'z_ele', 'sos_bgn', 'rho_bgn', '-v7.3');
            
        end
        
    else
        disp(['scenario number ' int2str(i_sce) ' was already computed and will not be skipped.'])
    end
    
end
