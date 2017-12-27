function [matchedPoints1, matchedPoints2] =FeatureMatching(image1, image2, edgeth, peakth)
    Ia = imread(strcat(num2str(image1),'.jpg')) ;
    Ia = im2double(Ia) ;
    
    Ib = imread(strcat(num2str(image2), '.jpg')) ;
    Ib = im2double(Ib) ;
   
   
    Ia_gray = single(rgb2gray(Ia));
    Ib_gray = single(rgb2gray(Ib));
    
    [fa, da] = vl_sift(Ia_gray, 'edgethresh',edgeth, 'PeakThresh', peakth);
    [fb, db] = vl_sift(Ib_gray, 'edgethresh',edgeth, 'PeakThresh', peakth);
    
    ShowKeyPoints(Ia, fa, da, 350);
    ShowKeyPoints(Ib, fb, db, 350);
    
    [matches, scores] = vl_ubcmatch(da, db) ;

    matchedPoints1 = fa(1:2,matches(1,:));
    matchedPoints2 = fb(1:2,matches(2,:));

    figure();
    showMatchedFeatures(Ia,Ib,matchedPoints1',matchedPoints2','montage');
end


function ShowKeyPoints(image, frames, descriptors , key_num)
    figure;
    imshow(image) ;
    
    % find select number of keypoints and/or descriptors
    perm = randperm(size(frames,2)) ;
    sel = perm(1:key_num) ;
    h1 = vl_plotframe(frames(:,sel)) ;
    set(h1,'color','y','linewidth',3) ;
    
%     h3 = vl_plotsiftdescriptor(descriptors(:,sel),frames(:,sel)) ;
%     set(h3,'color','g') ;
end


