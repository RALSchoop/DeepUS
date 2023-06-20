% script to study the first experimental data aquired by Gijs Hendriks
% <Gijs.Hendriks@radboudumc.nl> on 24 June 2021
%
% author: Felix Lucka, Georgios Pilikos
% date:        01.09.2021
% last update: 29.10.2021


clear
clc
%close all

% initializes the toolbox
startup

% the script assumes that the data is stored in
% storage_path/ExperimentalData/CIRS073_RUMC/, or
% storage_path/ExperimentalData/CIRS040GSE/
data_path = [storage_path 'ExperimentalData' fs 'CIRS073_RUMC'];

%% load the header

% the script tries to load the data header which contains all the
% information about the transducer settings
header_file = [data_path fs 'USHEADER_20210624112712.mat'];
load(header_file);
USHEADER

%% load the data set

lesion_type       = 'hypo_lesion'; % choose one of the subfolders
data_set          = 3; % numbering relative to how the OS lists files
lesion_data_path  = [data_path fs lesion_type fs];
lesion_data_files = dir([lesion_data_path, 'USDATA_*.mat']);
load([lesion_data_path lesion_data_files(data_set).name]);

% convert data to double and get rid of empty dimensions
USDATA = double(squeeze(USDATA));
n_t    = size(USDATA, 1); % number of time samples
n_ele  = size(USDATA, 2); % number of transducer elements
n_ang  = size(USDATA, 3); % number of steering angles

%% now we can sub-sample in angle if we want to 

n_ang_ss = 1; % change this to 1 to get single plane wave data
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
n_ang  = size(USDATA, 3);

%% data pre-processing

% sound speed used for computations
sos_bgn  = USHEADER.c;

% correct general delay aka "lens correction"
lens_delay  = 96;
USDATA      = USDATA(lens_delay:end, :, :);
n_t         = size(USDATA, 1);

% correct angle-dependent delay
dt        = 1/USHEADER.fs;
delay_1   = USHEADER.xmitDelay(:,1) - (mean(USHEADER.xmitDelay(:,64:65),2));
delay_2   = abs((n_ele-1)/2*USHEADER.pitch * sin(deg2rad(USHEADER.xmitAngles)) / sos_bgn);
% choose a way to compute the delay
delay     = delay_2;
delay_ind = floor(delay./dt) + 1;
for i_ang=1:n_ang
    USDATA(1:n_t-delay_ind(i_ang)+1, :,i_ang)   = USDATA(delay_ind(i_ang):end, :,i_ang);
    USDATA(n_t-delay_ind(i_ang)+2:end, :,i_ang) = 0;
end

% correct channel 125
USDATA(:,125,:) = 2*USDATA(:,125,:);

%% a bit of data analysis (clipping etc)

% power over time, angle and channel
figure();
subplot(1,3,1);
plot(sqrt(squeeze(sum(sum(USDATA.^2, 2), 3))));
subplot(1,3,2);
plot(sqrt(squeeze(sum(sum(USDATA.^2, 1), 3))));
subplot(1,3,3);
plot(sqrt(squeeze(sum(sum(USDATA.^2, 1), 2))));

[~, zero_ind] = min(abs(USHEADER.xmitAngles));
data_visu     = squeeze(USDATA(:,:,zero_ind));
data_visu     = data_visu / max(abs(data_visu(:)));
clip_off      = 0.3;
data_visu     = max(-clip_off, min(data_visu, clip_off));

figure();
imagesc(data_visu)

%% set up F-K migration and run it

fk_para          = [];
fk_para.TXangle  = deg2rad(USHEADER.xmitAngles(:));
fk_para.pitch    = USHEADER.pitch;
fk_para.t0       = 0;
fk_para.c        = sos_bgn;
fk_para.fs       = USHEADER.fs;
passband_fac     = 1;
passband         = (USHEADER.fc + (USHEADER.fc * USHEADER.fbw * passband_fac*[-0.5,0.5])) / (USHEADER.fs/2);
%fk_para.passband = passband;

% call F-K migration
[img_fkmig, fk_para] = fkmig(USDATA, fk_para);

%% image post processing

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

%% display image

% range of the displayed area
minX = -12.4/1000;
maxX =  12.4/1000;
minZ = 0;
maxZ = 30/1000;

figure();
imagesc(fk_para.x*1000, fk_para.z*1000, img_pp); 
colormap(gray);
axis equal manual;
axis([minX maxX minZ maxZ]*1000);
xlabel('x');
ylabel('z');
