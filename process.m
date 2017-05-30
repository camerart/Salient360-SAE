%%% process.m ---
%%
%% Filename: process.m
%% Author: Fred Qi
%% Created: 2017-05-30 11:20:24(+0800)
%%
%% Last-Updated: 2017-05-30 14:14:53(+0800) [by Fred Qi]
%%     Update #: 100
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%% Commentary:
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%% Change Log:
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;

image_path = 'images';
output_path = 'saliency';
msg_head = 'Head saliency of %s has been written to %s.';
msg_head_eye = 'Head and eye saliency of %s has been written to %s.';

files_all = dir(fullfile(image_path, 'P*.jpg'));
for idx = 1:length(files_all)
    disp(' ');
    image_name = files_all(idx).name;
    image_pathname = fullfile(image_path, image_name);
    disp(['Processing ' image_pathname ' ...']);
    % read the input image
    imgIn = imread(image_pathname);
    
    % Task A: Head saliency
    sal_head = HeadSalMap(imgIn);
    % Save the saliency estimation result to a binary file.
    sal_filename = save_saliency(sal_head, ...
                                 output_path, image_name, '_head.bin');
    disp(sprintf(msg_head, image_pathname, sal_filename));

    % Task B: Head and eye saliency
    sal_head_eye = HeadEyeSalMap(imgIn);
    % Save the saliency estimation result to a binary file.
    sal_filename = save_saliency(sal_head_eye, ...
                                 output_path, image_name, '_head_eye.bin');
    disp(sprintf(msg_head_eye, image_pathname, sal_filename));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% process.m ends here
