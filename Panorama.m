function Panorama(dataset_name)
    % resize scale factor
    %% 
    resize_factor = 0.2;
    image5 = strcat(dataset_name,'/5.jpg');
    image4 = strcat(dataset_name,'/4.jpg');
    image3 = strcat(dataset_name,'/3.jpg');
    image2 = strcat(dataset_name,'/2.jpg');
    image1 = strcat(dataset_name,'/1.jpg');
    
    image_store = imageDatastore({image5,image4,image3,image2,image1});

    % Display images to be stitched
    %montage(image_store.Files)
    
    % Read the first image from the image set.
    I_pre = readimage(image_store, 1);
    
    % resize, original image too large
    I_pre = imresize(I_pre, resize_factor);

    % Initialize features for I(1)
    grayImage = rgb2gray(I_pre);
    
    % Initialize all the transforms to the identity matrix. Note that the
    % projective transform is used here because the building images are fairly
    % close to the camera. Had the scene been captured from a further distance,
    % an affine transform would suffice.
    numImages = numel(image_store.Files);
    tforms(numImages) = projective2d(eye(3));
    
    % Iterate over remaining image pairs
    for n = 2:numImages
        I = readimage(image_store, n);
        % resize, original image too large
        I = imresize(I, resize_factor);
      
        grayImage = rgb2gray(I);

        % get feature matching with SIFT
        [matchedPoints1, matchedPoints2] = FeatureMatching(I_pre, I, 10, 0.05);

        % Estimate the homography using RANSAC
        % set iteration num to be a very large number
        % iteration will stop if a consensus set of 100 is found 
            global_H = RansacComputeGlobalHomograhy(60000, I_pre, I,...
         matchedPoints2,matchedPoints1);
        
        tforms(n) = projective2d(global_H');
  
        % Compute T(n) * T(n-1) * ... * T(1)
        tforms(n).T = tforms(n).T * tforms(n-1).T;
        
        I_pre = I;
    end
    
    imageSize = size(I);  % all the images are the same size

    % Compute the output limits  for each transform
    for i = 1:numel(tforms)
        [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
    end
    
%     avgXLim = mean(xlim, 2);
% 
%     [~, idx] = sort(avgXLim);
% 
%     centerIdx = floor((numel(tforms)+1)/2);
% 
%     centerImageIdx = idx(centerIdx);

    centerImageIdx = 3;
    
    Tinv = invert(tforms(centerImageIdx));

    for i = 1:numel(tforms)
        tforms(i).T = tforms(i).T * Tinv.T;
    end
    
    for i = 1:numel(tforms)
        [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
    end
    

    % Find the minimum and maximum output limits
    xMin = min([1; xlim(:)]);
    xMax = max([imageSize(2); xlim(:)]);

    yMin = min([1; ylim(:)]);
    yMax = max([imageSize(1); ylim(:)]);

    % Width and height of panorama.
    width  = round(xMax - xMin);
    height = round(yMax - yMin);

    % Initialize the "empty" panorama.
    panorama = zeros([height width 3], 'like', I);
    
    blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

    % Create a 2-D spatial reference object defining the size of the panorama.
    xLimits = [xMin xMax];
    yLimits = [yMin yMax];
    panoramaView = imref2d([height width], xLimits, yLimits);

    % Create the panorama.
    for i = 1:numImages
        I = readimage(image_store, i);  
        % resize, original image too large
        I = imresize(I, resize_factor);
        % Transform I into the panorama.
        warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
        % Generate a binary mask.
        mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
        % Overlay the warpedImage onto the panorama.
        panorama = step(blender, panorama, warpedImage, mask);
    end

    figure
    imshow(panorama)
end
    
% estimate homography using RANSAC
function H = RansacComputeGlobalHomograhy(iteration_total, Ia, Ib, matchedPoints1, matchedPoints2)
    current_large_consensus = [];
    
    for x = 1:iteration_total
        [findPairs, points_a, points_b] = SelectRandomSubset(matchedPoints1, matchedPoints2, 4);
        if (findPairs == true)
            % compute local homography to help find the max consensus set
            Hi = ComputeHomography(points_a, points_b);
            consensus_idx_list = ComputeInliers(Hi, matchedPoints1, matchedPoints2);
            
            if (size(consensus_idx_list) > size(current_large_consensus))
                current_large_consensus = consensus_idx_list;
                % stop if consensus set size reach its threshold, so we
                % have enough point for stitching
                if (size(current_large_consensus, 2) > size(matchedPoints1, 2)/10)
                    break;
                end
            end
        end
    end
    % compute global homography
    consensus_set_points1 = matchedPoints1(:,current_large_consensus);
    consensus_set_points2 = matchedPoints2(:,current_large_consensus);
    
    % display matching after RANSAN (implies the consensus set)
    figure;
    showMatchedFeatures(Ia,Ib,consensus_set_points2',consensus_set_points1','montage');
    
    % get global homography using the consensus set
    H = ComputeHomography(consensus_set_points1', consensus_set_points2');
end

function [find4pairs, points_a, points_b] = SelectRandomSubset(matchedPoints1, matchedPoints2, select_number)
    points_a = zeros(select_number,2);
    points_b = zeros(select_number,2);
    find4pairs = false;
    
    if (size(matchedPoints1,2) < select_number &&...
        size(matchedPoints2,2) < select_number &&...
        size(matchedPoints1,2) ~= size(matchedPoints2,2))
    return;
    else
        random_list = randi(size(matchedPoints1,2),[select_number,1]);
        for i=1:length(random_list)
            points_a(i,:) = matchedPoints1(:,random_list(i));
            points_b(i,:) = matchedPoints2(:,random_list(i));
        end
        find4pairs = true;     
    end
end

function homography = ComputeHomography(points_a, points_b)
    A = zeros(size(points_a,1)*2,9);
    for i = 1:size(points_a,1)
        A(2*i-1,:) = [-points_a(i,1), -points_a(i,2),...
            -1, 0, 0, 0, ...
            points_a(i,1)*points_b(i,1), ...
            points_b(i,1)*points_a(i,2), ...
            points_b(i,1)];
        A(2*i, :) = [0, 0, 0, ...
            -points_a(i,1), -points_a(i,2),...
            -1, ...
            points_a(i,1)*points_b(i,2), ...
            points_b(i,2)*points_a(i,2),...
            points_b(i,2)];
    end
    
    [U, S, V] = svd(A);
    h = V(:,9);
    homography = reshape(h,3,3)';
end

function consensus_idx_list = ComputeInliers(homography, points_a, points_b)
    dist_threshold = 5;
    points_a_homogenous = padarray(points_a, [1,0], 1, 'post');
    points_b_homogenous = padarray(points_b, [1,0], 1, 'post');
    transformed_points_a = homography*points_a_homogenous;
    transformed_points_a = transformed_points_a./transformed_points_a(3,:);
    euclidean_dist_square = (sum((points_b_homogenous - transformed_points_a).^2, 1)).^(1/2);
    consensus_idx_list = find(euclidean_dist_square<dist_threshold);
end

% feature matching using SIFT - function from previous question with slight modification
function [matchedPoints1, matchedPoints2] = FeatureMatching(image1, image2, edgeth, peakth)
    Ia = im2double(image1) ;
    Ib = im2double(image2) ;
    
    Ia_gray = single(rgb2gray(image1));
    Ib_gray = single(rgb2gray(image2));
    
    [fa, da] = vl_sift(Ia_gray, 'edgethresh',edgeth, 'PeakThresh', peakth);
    [fb, db] = vl_sift(Ib_gray, 'edgethresh',edgeth, 'PeakThresh', peakth);
    
    [matches, scores] = vl_ubcmatch(da, db) ;

    matchedPoints1 = fa(1:2,matches(1,:));
    matchedPoints2 = fb(1:2,matches(2,:));

    figure();
    showMatchedFeatures(Ia,Ib,matchedPoints1',matchedPoints2','montage');
end


