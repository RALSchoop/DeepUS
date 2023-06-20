function [result, freq, Fs, statistics] = spectralAnalysis(f, dt, result_type)
%SPECTRALANALYSIS computes the amplitude spectrum of input signals along
%the temporal dimension
%
%
% DESCRIPTION:
%       spectralAnalysis computes the amplitude spectrum of an input
%       signal f 
% USAGE:
%       [psd, freq, Fs, statistics] = frequencyAnalysis(f, dt, 'decibel')
%
% INPUTS:
%       f           - a n_timesteps x n_sensors x n_sources array of acoustic measurements
%                     or a cell of such measurements
%       dt          - temporal sampling
%       result_type - 'amplitude', 'power' or 'decibel'
%
%
% OUTPUTS:
%       result     - amplitude or power spectrum of each channel or
%                    cell of such spectra
%       freq       - vector with corresponding frequencies
%       Fs         - sampling frequency
%       statistics - struct that summarizes statistics over channels with
%                    fields 'mean', 'median', 'max' and 'min' or cell with
%                    such structs
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 05.04.2017
%       last update     - 03.12.2021
%
% See also spect


% check user defined value for power, otherwise assign default value
if nargin < 3 || isempty(result_type)
    result_type = 'amplitude';
end


if(iscell(f))
    
    % compute quantities recursive over cell
    result     = cell(size(f));
    statistics = cell(size(f));
    for i_cell = 1:numel(f)
        if(~isempty(f{i_cell}))
            [result{i_cell}, freq, Fs, statistics{i_cell}] = ...
                spectralAnalysis(f{i_cell}, dt, result_type);
        end
    end
else
    
    % determine sampling frequency
    Fs  = 1 / dt;
    
    % reshape
    f = reshape(f, size(f,1), []);
    
    % call spect.m to compute amplitude spectrum
    [freq, result] = spect(f, Fs, 'Dim', 1);
    
    switch result_type
        case 'amplitude'
            result = abs(result);
        case {'power', 'decibel'}
            result = result.^2;
    end
    
    % compute statistics over channels
    statistics.mean   = mean(result, 2);
    statistics.median = median(result, 2);
    statistics.max    = max(result, [], 2);
    statistics.min    = min(result, [], 2);
    
    switch result_type
        case 'decibel'
            
            % convert results to decibel sclale (a reference can be subtracted later)
            result            = 10 * log10(result);
            statistics.mean   = 10 * log10(statistics.mean);
            statistics.median = 10 * log10(statistics.median);
            statistics.max    = 10 * log10(statistics.max);
            statistics.min    = 10 * log10(statistics.min);
    end
    
end

end