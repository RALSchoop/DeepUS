% startup.m file for matlab part of the DeepUS toolbox
%
% author: Felix Lucka
% date:        21.10.2021
% last update: 21.10.2021
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp('DeepUS - toolbox for deep learning with ultrasonic data - matlab branch')
disp(' ')

% set global variables
global root_path

% Set userDir to root path from which kwave, felix_tools and storage of
% data can be found.
userDir = 'D:\Files\CWI Winter Spring 2022';
fs      = filesep();

% test if we are in the root folder of the toolbox
current_path = pwd;
if(length(current_path) < 26 || ~strcmp(current_path(end-5:end),'matlab') ...
          || ~strcmp(current_path(end-12:end-7),'DeepUS'))
    error('change to the root folder of the DeepUS toolbox')
else
    clear current_path
    root_path = pwd;
end

[~,computer_name] = system('hostname');
computer_name       = deblank(computer_name);

%%% here we set the paths for the toolboxes and data and results storage
%%% folders

kwave_path        = [userDir '\k-wave-toolbox-version-1.3\k-Wave\'];
felix_tools_path  = [userDir '\FelixMatlabTools\'];
storage_path      = [userDir '\Data\DeepUS\'];

% specify the paths for the different computers you use
% Not relevant if running on single machine.
switch computer_name
    case 'klamath.ci.cwi.nl'
        %storage_path = '/export/scratch3/felix/data/US/DeepUS/';
    case {'scan1.scilens.private', 'scan2.scilens.private', 'scan3.scilens.private',...
           'scan4.scilens.private', 'scan5.scilens.private', 'scan6.scilens.private',...
           'scan7.scilens.private', 'scan8.scilens.private',...
           'voxel1.scilens.private', 'voxel2.scilens.private', 'voxel3.scilens.private',...
           'voxel4.scilens.private', 'voxel5.scilens.private', 'voxel6.scilens.private',...
           'voxel7.scilens.private', 'voxel8.scilens.private'}
       
       storage_path = '/bigstore/felix/US/DeepUS/';
        
    otherwise

end

% add external software
addpath(genpath(felix_tools_path))
addpath(genpath(kwave_path))
addpath(genpath(root_path))




