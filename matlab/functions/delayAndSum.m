function image = delayAndSum(data, tIndMatSrc, tIndMatRec)
%DELAYANDSUM is simple implementation of the delay-and-sum type
%back-projection algorithm
%
% DETAILS: 
%   delayAndSum.m forms an image by computing the image intensity at 
%   a pixel p as 
%       Intensity(p) = sum_i sum_j data(travelTime(i,p) + travelTime(j,p))
%   where i and j run over the sources and receivers, respectively. 
%   
%
% USAGE:
%   image = delayAndSum(data, timeIndexMatrix, imageSize)
%
% INPUTS:
%   data - timeseries in format (Nt, Nrec, Nsrc) where Nt is the number of
%   time step, Nrec the number of recievers and Nsrc the number of sources
%   imageSize - size of 
%   tIndMatSrc - volume of size (Nz, Nx, Nsrc) that stores the travetime between 
%   image points and sources converted to indices in the time series (see
%   timeIndexFunction.m)
%
% OPTIONAL INPUTS:
%   tIndMatRec - the volume of size (Nz, Nx, Nrec) that stores the same
%   info as tIndMatSrc, just for the recievers. If not specified,
%   tIndMatRec = tIndMatSrc is assumed.

%
% OUTPUTS:
%   images - image of size Nz x Nx
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 17.11.2021
%       last update     - 17.11.2021
%
% See also

if(nargin < 3)
    % all transducers were used as sources and recievers
    tIndMatRec = tIndMatSrc;
end

[Nz,Nx,Nsrc] = size(tIndMatSrc);
Nrec         = size(tIndMatSrc, 3);

% assemble image 
image    = zeros(Nz*Nx,1);
for isrc = 1:Nsrc % loop over  source elements
    for jrec = 1:Nrec % loop over receiver elements
        % compute travel time for all pixel
        timeIndices = tIndMatSrc(:,:,isrc) + tIndMatRec(:,:,jrec) + 1;
        % delay to form image
        image_ij = data(timeIndices, jrec, isrc);
        % sum
        image    = image + image_ij;
    end
end

image = reshape(image, [Nz,Nx]);

end