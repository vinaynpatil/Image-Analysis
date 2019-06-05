% HW4
% Name - Vinay Patil
% NetId - vpatil3
% Question 2

path = 'distributed';

% Getting all the file details from the distributed folder
files =  dir(sprintf('%s/img*.ppm',path));
num_files = length(files);
%Getting a referance image to get the image size details for furture use
ref_img = double(imread(sprintf('%s/%s',path, files(1).name)));

%Justification for selecting the below weights for the grayscale convertion

%Looking at the 3 channels of each images I realized that the blue channel
%contributes nothing(Except a dot occassionaly in the unhealthy retina)that
%will help us with the feature extraction.

%Similarly the red channel showed very low contrast when it comes to the
%features, but slightly better than the blue channel.

%The green channel turns out to be the best one among all as it provides
%the right contrast which makes our task of recognizing the features easy.

%Creating a vector to hold all the grayscaled images
gray_imgs = zeros([size(ref_img(:,:,1)),num_files]);

for i=1:num_files
    img_loop = double(imread(sprintf('%s/%s',path, files(i).name)));
    img = 0.1*img_loop(:,:,1) + 0.85*img_loop(:,:,2) + 0.05*img_loop(:,:,3);
    %Applying median filter to removing noise from the data and preserving
    %edges.
    img = medfilt2(img,[3,3]);
    gray_imgs(:,:,i) = img;
end


%One of the important things to do here is to mask out the optic disk from
%the retina images. The reason behind this is, the optic disk and the
%exudates share the similar level of intensities and color and hence will
%interfere in the feature detection.

%So the idea here is to detect the locaton of the optic circle using the
%circular hough, provided by you in class and later mask out that region 
%from the preprocessed retina image.

%This is the vector containing all the locations of the optic circles
%(Represented as 1's)
diskLocations = getOpticalDisk(gray_imgs);

%Taking negataion to change 0's to 1's and vice-versa and converting it to
%logical vector, so that it becomes easier to apply the mask.
diskLocations = ~diskLocations;

%Looking at the grayscaled images of the retina, it seems that windowing
%method would be suitable here to get the parts of the image I am most
%interested in
%But I am facing problem finding the min_val for the window, which changes
%across images. When I consider one image at a time, I can set the min_val
%by eyeballing and it works great, but since there is no one fit all value
%for all images, it causing some issues in the result. I hope to find a
%different method(which works independent of the images) for the final to 
%fix this issue

result = zeros(1,num_files);
for i=1:36
    img = gray_imgs(:,:,i);
    
    min_val = 0.80*max(img(:));
    max_val = max(img(:));
    slope = 255/(max_val - min_val);
    int = -1*min_val*255/(max_val - min_val);
    rng = img>min_val & img<max_val;
    img_wind = 0*img;
    img_wind(rng) = img(rng)*slope + int;

    %Computing element wise multiplication to set the recognized optical disk
    %to background - 0.
    img_wind = img_wind.*diskLocations(:,:,i);
    
    %Making use of dilate followed by erode to fill in the gaps and thin it
    %down for a improved component count
    img_wind = bwmorph(img_wind, 'dilate', 4);
    img_wind = bwmorph(img_wind, 'erode', 4);
    
    %Now since the contrast is fairly good(binary), we can make use of the
    %seed growing algorithm to count the no of components in the image.
    %Here I am making use of 8 connected objects(Thats what the bwlabel method defaults to)
    %in the method bwlabel. This can be changed by setting a diff value of N in bwlabel(img,N)
    %Also I noticed that it works best with the logical values.
    [labeledImage, numberOfObject] = bwlabel(img_wind);
    result(i) = numberOfObject;
end

figure
plot(1:num_files, result, 'bo')
xlabel('Image No', 'Fontsize', 20)
ylabel('Feature Magnitude', 'Fontsize', 20)


function [circle_img] = getOpticalDisk(imgs)
img_bin = zeros(size(imgs));
for i=1:size(imgs,3)
    img_loop = imgs(:,:,i);
    cut = quantile(img_loop(:), .98);
    img_bin(:,:,i) = img_loop>cut;
end

xrng = 1:size(img_loop, 2);
yrng = 1:size(img_loop, 1);
[xx,yy] = meshgrid(xrng,yrng);

num_radii = zeros(1, size(imgs,3));
circle_img = zeros(size(imgs));
for i=1:size(imgs,3)
    img_loop = img_bin(:,:,i);
    [center, radii] = imfindcircles(img_loop==1, [50 120],...
        'sensitivity', .98);
    num_radii(i) = radii(1);
    circle_img(:,:,i) = hypot(xx-center(1,1), yy-center(1,2))<=radii(1);
end

end
