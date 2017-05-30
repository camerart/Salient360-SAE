% Filename: save_saliency.m
% Author: Fred Qi
% Created: 2017-04-30 12:14:39(+0800)
%
% Last-Updated: 2017-05-30 12:26:41(+0800) [by Fred Qi]
%     Update #: 24

%% Arguments:
%    saliency_map: the saliency map to be saved.
%    output_path: the folder to save the saliency map.
%    image_name: name of the original image without path.
%    suffix: suffix and extension of the output saliency filename.


function [saliency_filename] = save_saliency(saliency_map, ...
                                             output_path, image_name, suffix)
    % disp(image_name);
    image_basename = strtok(image_name, '.');
    saliency_filename = fullfile(output_path, [image_basename, suffix]);
    % disp(saliency_filename);
    fid = fopen(saliency_filename, 'wb');
    fwrite(fid, saliency_map, 'double');
    fclose(fid);
end

% save_saliency.m ends here
