% Final
% Name - Vinay Patil
% NetId - vpatil3

clear
clc

path = 'distributed';

% Getting all the file details from the distributed folder
files =  dir(sprintf('%s/img*.ppm',path));
num_files = length(files);
%Getting a referance image to get the image size details for furture use
ref_img = im2double(imread(sprintf('%s/%s',path, files(1).name)));
gray_imgs = zeros([size(ref_img(:,:,1)),num_files]);

% figure
% colormap('gray');
for i=1:num_files
    files(i).name
    img_loop = im2double(imread(sprintf('%s/%s',path, files(i).name)));
    img = 0.1*img_loop(:,:,1) + 0.85*img_loop(:,:,2) + 0.05*img_loop(:,:,3);
    %Applying median filter to removing noise from the data and preserving
    %edges.
    
    img = medfilt2(img,[3,3]);
    gray_imgs(:,:,i) = img;
    
    %     subplot(6,6,i)
    %     imagesc(img)
end

% figure
% colormap('gray');

colormap('gray');
result = zeros(1,num_files);
for i=1:36
    
    img = imcomplement(gray_imgs(:,:,i));
    
    bg = medfilt2(img,[25,25]);
    
    img = img - bg;
    img = adapthisteq(img);
    maxi = max(img(:));
    img = medfilt2(img,[5,5]);
    
    
    
    img = imbinarize(img);
    img = bwmorph(img, 'dilate', 4);
    subplot(6,6,i)
    imagesc(img)
    [labeledImage, numberOfObject] = bwlabel(img);
    result(i) = numberOfObject;
end

figure
scatter(1:num_files, result)
xlabel('Image No', 'Fontsize', 20)
ylabel('Feature Magnitude', 'Fontsize', 20)

result