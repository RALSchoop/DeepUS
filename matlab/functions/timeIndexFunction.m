function timeIndexMatrix = timeIndexFunction(sos, offset, samplingFrequency, x_e, z_e, X, Z)
%TIMEINDEXFUNCTION prepares the delay-and-sum image formation by
%calculating the traveltimes from each transducer to each image point
%
% DETAILS: 
%   timeIndexFunction.m prepares the delay-and-sum image formation by
%   calculating the traveltimes from each transducer to each image point and
%   converting it into an index of the sampled time series. It assumes a
%   homogenous speed of sound throughout the volume, so it assumes the
%   signal travels on a straight ray
%
% USAGE:
%   timeMatrix = timeIndexFunction(sos, latency, samplingFrequency, x_e, z_e, X, Z)
%
% INPUTS:
%   sos - scalar speed of sound that the travel time computation is based
%   	upon
%   offset - time to add to the travel time computations, e.g., to correct
%   	for a delay between source activation and recording
%   samplingFrequency - [1/s] sampling frequency used to convert the travel
%   times into indices of the time series
%   x_e, z_e - (x,z) coordinates of the transducers 
%   X, Z     - (x,z) coordinates of the image grid points
%
% OUTPUTS:
%   timeIndexMatrix - volume of size (Nx, Nz, Ne) where
%   timeIndexMatrix(i,j,k) is the index in a time series 0, dt, 2*dt, 3*dt,...
%   where dt = 1/samplingFrequency that is closed to the travel time between 
%   transducer k and image point (i,j), that is between  (x_e(k), z_e(k)) 
%   and (X(i,j), Z(i,j))
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 17.11.2021
%       last update     - 17.11.2021
%
% See also delayAndSum

[Nz, Nx] = size(X);
Ne       = length(x_e);

% set up element-to-pixel distance matrix R
R = zeros(Nz, Nx, Ne);
for ie = 1:Ne
    R(:,:,ie) = sqrt((X - x_e(ie)).^2 + (Z - z_e(ie)).^2);
end

% convert into time index matrix
timeIndexMatrix  = round((R/sos + offset) * samplingFrequency);

end