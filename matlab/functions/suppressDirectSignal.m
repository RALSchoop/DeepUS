function data = suppressDirectSignal(data, dist, sos_min, dt, source_wave_form)
%SUPPRESSDIRECTSIGNAL sets early signals in timeseries to zero
%
% DETAILS: 
%   suppressDirectSignal.m can be used to filter out signals whose travel
%   time is so small that they cannot come from reflection in the volume
%   but are either noise, artifacts, or in the case of simulated data, the
%   unrealistically large direct signal traveling from source to reciever. 
%
% USAGE:
%   data = suppressDirectSignal(data, dist, sos_min, dt, source_wave_form)
%
% INPUTS:
%   data - data in (Nt, Nrec, Nsrc) format
%   dist - distance matrix between sources and recievers of form (Nrec, Nsrc)
%   sos_min - lower bound for the speed of sound that is used to compute the 
%       direct arrival (the lower, the more time series will be zeored out.
%   dt   - time sampling interval (data is assumed to be sampled at 0, dt,
%   2*dt, 3dt,...
%   source_wave_form - source activation wave form, will be used to compute
%   the offset added to the travel time
%
% OUTPUTS:
%   data - processed data with early arriving signals zeroed out
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 17.11.2021
%       last update     - 17.11.2021
%
% See also

% find length of source wave form 
signal_duration = find(abs(source_wave_form) / max(source_wave_form) > 10^-6, 1, 'last');

for i_rec=1:size(data, 2)
    for i_src=1:size(data, 3)
        arrival_index = ceil((dist(i_rec,i_src) / sos_min)/dt);
        data(1:(signal_duration+arrival_index), i_rec, i_src) = 0;
    end
end

end