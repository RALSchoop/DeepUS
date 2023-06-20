% script to study the first experimental data aquired by Gijs Hendriks
% <Gijs.Hendriks@radboudumc.nl> on 24 June 2021
%
% author: Felix Lucka, Georgios Pilikos, Ryan Schoop
% date:        01.09.2021
% last update: 20.06.2023


clear
clc
%close all

% initializes the toolbox
startup

% the script assumes that the data is stored in:
% storage_path/ExperimentalData/CIRS073_RUMC/, or
% storage_path/ExperimentalData/CIRS040GSE/

data_set_name = 'CIRS073_RUMC';
experimental_data_path = [storage_path 'ExperimentalData' fs data_set_name];

% this is where the prepared training data will be stored
training_data_path_root = [storage_path 'TrainingData' fs data_set_name];

%% settings of the reconstruction

n_ang_ss      = 1; % sub-sampled number of steering angles used in the reconstruction (1,...,75)
nz_cutoff     = 1500; % cut off in pixels depth direction (TODO: replace by proper interpolation)
%passband_fac  = 1; % relative filtering bandwidth

%% load the header

% the script tries to load the data header which contains all the
% information about the transducer settings

% Pick the header file based on the data_set chosen above.
header_file = [experimental_data_path fs 'USHEADER_20210624112712.mat'];
load(header_file);
USHEADER

n_ang = length(USHEADER.xmitAngles);
n_ele = size(USHEADER.xmitDelay, 2); % number of transducer elements

%% prepare the loop over all data sets

% get a list of all data files
data_files = rDir(experimental_data_path, 'USDATA_*.mat');
n_data     = length(data_files)

% make res dir for image
% training_data_path_img = [training_data_path_root fs 'Images' fs 'PB' num2str(passband_fac) fs 'nAng' int2str(n_ang_ss) fs];
training_data_path_img = [training_data_path_root fs 'Images' fs 'nAng' int2str(n_ang_ss) fs];
makeDir(training_data_path_img);
% make res dir for data
training_data_path_data = [training_data_path_root fs 'Data' fs 'nAng' int2str(n_ang_ss) fs];
makeDir(training_data_path_data);

% load first data set
load([data_files(2).folder fs data_files(2).name]);
    
% convert data to double and get rid of empty dimensions
USDATA = double(squeeze(USDATA));
    
% sub-sample in angle if we want to
if(n_ang_ss == 1)
    ang_ind = find(USHEADER.xmitAngles == 0);
else
    ang_ss  = linspace(USHEADER.xmitAngles(1), USHEADER.xmitAngles(end), n_ang_ss);
    ang_ind = zeros(n_ang_ss, 1);
    for i_ang = 1:n_ang_ss
        [~,ang_ind(i_ang)] = min(abs(USHEADER.xmitAngles - ang_ss(i_ang)));
    end
end
    
USDATA                    = USDATA(:,:,ang_ind);
USHEADER.xmitAngles       = USHEADER.xmitAngles(ang_ind);
USHEADER.xmitDelay        = USHEADER.xmitDelay(ang_ind,:);
USHEADER.xmitFocus        = USHEADER.xmitFocus(ang_ind);
USHEADER.xmitApodFunction = USHEADER.xmitApodFunction(ang_ind,:);
        
% sound speed used for computations
sos_bgn     = USHEADER.c;
lens_delay  = 96;

%% main loop
for i_data = 1:n_data
    
    fprintf(['reconstructing data set ' int2str(i_data) '...'])
    clock_cmp = tic;
    
    load([data_files(i_data).folder fs data_files(i_data).name]);
    
    % convert data to double and get rid of empty dimensions
    USDATA = double(squeeze(USDATA));
    USDATA = USDATA(:,:,ang_ind);

    
    % data pre-processing
    
    % correct general delay aka "lens correction"
    USDATA      = USDATA(lens_delay:end, :, :);
    n_t         = size(USDATA, 1);
    
    % correct angle-dependent delay
    dt        = 1/USHEADER.fs;
    delay_1   = USHEADER.xmitDelay(:,1) - (mean(USHEADER.xmitDelay(:,64:65),2));
    delay_2   = abs((n_ele-1)/2*USHEADER.pitch * sin(deg2rad(USHEADER.xmitAngles)) / sos_bgn);
    % choose a way to compute the delay
    delay     = delay_2;
    delay_ind = floor(delay./dt) + 1;
    for i_ang=1:n_ang_ss
        USDATA(1:n_t-delay_ind(i_ang)+1, :,i_ang)   = USDATA(delay_ind(i_ang):end, :,i_ang);
        USDATA(n_t-delay_ind(i_ang)+2:end, :,i_ang) = 0;
    end
    
    % correct channel 125
    USDATA(:,125,:) = 2*USDATA(:,125,:);
    
    % set up F-K migration and run it
    fk_para          = [];
    fk_para.TXangle  = deg2rad(USHEADER.xmitAngles(:));
    fk_para.pitch    = USHEADER.pitch;
    fk_para.t0       = 0;
    fk_para.c        = sos_bgn;
    fk_para.fs       = USHEADER.fs;
    %passband         = (USHEADER.fc + (USHEADER.fc * USHEADER.fbw * passband_fac*[-0.5,0.5])) / (USHEADER.fs/2);
    %fk_para.passband = passband;
    
    % call F-K migration
    [img_fkmig, fk_para] = fkmig(USDATA, fk_para);
    
    % image post processing
    img_pp = img_fkmig;
    % hilbert transform
    img_pp = abs(hilbert(img_pp));
    % % SNR transform
    img_pp = 20*log10(img_pp);
    img_pp = img_pp - max(img_pp(:));
    % clipping
    img_pp(img_pp < -70) = -70;
    img_pp(img_pp > 0) = 0;
    % shift from negative values
    img_pp = (img_pp + 70) / 70;
    
    % TODO: reintroduce the interpolation
    % interpolate images to isotropic grid twice as deep as wide
    %x_vec = fk_para.x(:)';
    %dx    = x_vec(2) - x_vec(1);
    %z_vec = dx * (0:(2*length(x_vec)-1));
    %img_fkmig = interpn(fk_para.z, fk_para.x, img_fkmig, z_vec(:), x_vec);
    %img_pp    = interpn(fk_para.z, fk_para.x, img_pp, z_vec(:), x_vec);
    img_fkmig = img_fkmig(1:nz_cutoff,:);
    img_pp    = img_pp(1:nz_cutoff,:);
    x_vec = fk_para.x(:)';
    z_vec = fk_para.z(1:nz_cutoff)';
    
    % save images
    res_filename_img = [training_data_path_img 'img_' int2str(i_data) '.mat'];
    save(res_filename_img, 'img_fkmig', 'img_pp', 'fk_para', 'x_vec', 'z_vec','-v7.3');
    
    % save data
    data = USDATA;
    res_filename_data = [training_data_path_data 'data_' int2str(i_data) '.mat'];
    save(res_filename_data, 'data', 'fk_para','-v7.3');
    
    disp(['done. Computation time:' convertSec(toc(clock_cmp))])
    
end

%% Extra metadata
if strcmp(data_set_name,'CIRS073_RUMC')
    LesionIdx.hyper_lesion1 = find(endsWith({data_files.folder},'hyper_lesion'));
    LesionIdx.hyper_lesion2 = find(contains({data_files.folder},'hyper_lesion2'));
    LesionIdx.hyper_lesion3 = find(contains({data_files.folder},'hyper_lesion3'));
    LesionIdx.hypo_lesion1 = find(endsWith({data_files.folder},'hypo_lesion'));
    LesionIdx.hypo_lesion2 = find(contains({data_files.folder},'hypo_lesion2'));
    LesionIdx.hypo_lesion3 = find(contains({data_files.folder},'hypo_lesion3'));
    LesionIdx.no_lesion1 = find(endsWith({data_files.folder},'no_lesion'));
    LesionIdx.no_lesion2 = find(contains({data_files.folder},'no_lesion2'));
elseif strcmp(data_set_name,'CIRS040GSE')
    LesionIdx.high_attenuation_hypoechoic = (1:5);
    LesionIdx.high_attenuation_wires = (6:10);
    LesionIdx.high_attenuation_plusdb = (11:15);
    LesionIdx.high_attenuation_minusdb = (16:20);
    LesionIdx.low_attenuation_hypoechoic = 20 + (1:5);
    LesionIdx.low_attenuation_wires = 20 + (6:10);
    LesionIdx.low_attenuation_plusdb = 20 + (11:15);
    LesionIdx.low_attenuation_minusdb = 20 + (16:20);
else
    error('Unknown data_set specified.')
end
LesionIdx.nIdx = n_data; % Don't forget this one!!

TargetInfo.nz_cutoff = nz_cutoff;

res_filename_metadata = [training_data_path_root fs 'metadata' '.mat'];

save(res_filename_metadata, 'USHEADER', 'TargetInfo', 'LesionIdx', '-v7.3');